import gc,sys,os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch,wandb
from sklearn.metrics import f1_score,roc_curve
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from .models.breast_clip_classifier import BreastClipClassifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Datasets.dataset_utils import get_single_gpu_dataloader_RSNA
from breastclip.scheduler import LinearWarmupCosineAnnealingLR
from metrics import pfbeta_binarized, pr_auc, compute_auprc, auroc, compute_accuracy_np_array
from utils import seed_all, AverageMeter, timeSince


def do_experiments(args, device):

    if 'efficientnetv2' in args.arch:
        args.model_base_name = 'efficientv2_s'
    elif 'efficientnet_b5_ns' in args.arch:
        args.model_base_name = 'efficientnetb5'
    else:
        args.model_base_name = args.arch

    args.data_dir = Path(args.data_dir)
    if args.dataset.lower() == "vindr" or args.dataset.lower() == "rsna":
        args.df = pd.read_csv(args.data_dir / args.csv_file)

    elif args.dataset.lower() == "csaw":
        args.df = pd.read_csv("/storage/csaw_cc_data/CSAW_final_balanced.csv")
    args.df = args.df.fillna(0)

    print(f"df shape: {args.df.shape}")
    #print(args.df.columns)
    oof_df = pd.DataFrame()
    for fold in range(args.start_fold, args.n_folds):
        args.cur_fold = fold
        print(f"fold: {args.cur_fold}")
        seed_all(args.seed)
        if args.dataset.lower() == "rsna":
            args.train_folds = args.df[
                (args.df['fold'] == 1) | (args.df['fold'] == 2)].reset_index(drop=True)
            args.valid_folds = args.df[args.df['fold'] == args.cur_fold].reset_index(drop=True)
            print(f"train_folds shape: {args.train_folds.shape}")
            print(f"valid_folds shape: {args.valid_folds.shape}")

        elif args.dataset.lower() == "vindr":
            args.train_folds = args.df[args.df['split'] == "training"].reset_index(drop=True)
            args.valid_folds = args.df[args.df['split'] == "test"].reset_index(drop=True)
        elif args.dataset.lower() == "csaw":
            args.train_folds = args.df[args.df['split'] == "train"].reset_index(drop=True)
            args.valid_folds = args.df[args.df['split'] == "val"].reset_index(drop=True)
        print(f"train_folds shape: {args.train_folds.shape}")
        print(f"valid_folds shape: {args.valid_folds.shape}")


        if args.inference_mode == 'y':
            _oof_df = inference_loop(args,device)
        elif args.inference_mode == 'n' or args.inference_mode == 'train_un_mammopretrained':
            _oof_df = train_loop(args, device)
        elif args.inference_mode == 'load':
            _oof_df = loaded_inference_loop(args, device)
        elif args.inference_mode == 'unpretrained_inf':
            _oof_df = unpretrained_inference_loop(args, device)

        oof_df = pd.concat([oof_df, _oof_df])

    if args.dataset.lower() == "rsna":
        oof_df = oof_df.reset_index(drop=True)
        oof_df['prediction_bin'] = oof_df['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
        oof_df_agg = oof_df[['patient_id', 'laterality', args.label, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()

        print(oof_df_agg.head(10))
        print('================ CV ================')
        aucroc = auroc(gt=oof_df_agg[args.label].values, pred=oof_df_agg['prediction'].values)

        oof_df_agg_cancer = oof_df_agg[oof_df_agg[args.label] == 1]
        oof_df_agg_cancer['prediction'] = oof_df_agg_cancer['prediction'].apply(lambda x: 1 if x >= 0.5 else 0)
        acc_cancer = compute_accuracy_np_array(oof_df_agg_cancer[args.label].values,
                                               oof_df_agg_cancer['prediction'].values)

        print(f'AUC-ROC: {aucroc}, acc +ve {args.label} patients: {acc_cancer * 100}')
        print('\n')
        print(oof_df.head(10))
        print(f"Results shape: {oof_df.shape}")
        print('\n')
        print(args.output_path)
        oof_df.to_csv(args.output_path / f'seed_{args.seed}_n_folds_{args.n_folds}_outputs.csv', index=False)


def train_loop(args, device):
    print(f'\n================== fold: {args.cur_fold} training ======================')
    print("labels:", args.label)
    if args.data_frac < 1.0:
        args.train_folds = args.train_folds.sample(frac=args.data_frac, random_state=1, ignore_index=True)

    if args.clip_chk_pt_path is not None:
        ckpt = torch.load(args.clip_chk_pt_path, map_location=device, weights_only=False)
        if ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        elif ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
            args.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        args.image_encoder_type = args.im_encoder
        ckpt = None
    if args.running_interactive:
        # test on small subsets of data on interactive mode
        args.train_folds = args.train_folds.sample(1000)
        args.valid_folds = args.valid_folds.sample(n=1000)

    train_loader, valid_loader,train_eval_loader = get_single_gpu_dataloader_RSNA(args)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    model = None
    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "cancer_birads":
        n_class = 5
    elif args.label.lower() == "birads":
        n_class = 3
    else:
        n_class = 1

    optimizer = None
    scheduler = None
    scalar = None
    mapper = None
    attr_embs = None
    if 'breast_clip' in args.arch:
        print(f"Architecture: {args.arch}")
        print(args.image_encoder_type)
        model = BreastClipClassifier(args, ckpt=ckpt, n_class=n_class)
        print("Model is loaded")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.warmup_epochs == 0.1:
            warmup_steps = args.epochs
        elif args.warmup_epochs == 1:
            warmup_steps = len(train_loader)
        else:
            warmup_steps = 10
        lr_config = {
            'total_epochs': args.epochs,
            'warmup_steps': warmup_steps,
            'total_steps': len(train_loader) * args.epochs
        }
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, **lr_config)
        scaler = torch.cuda.amp.GradScaler()

    model = model.to(device)
    #print(model)

    logger = SummaryWriter(args.tb_logs_path / f'fold{args.cur_fold}')

    if args.label.lower() == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to('cuda')
        print(f'pos_wt: {pos_wt}')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    best_aucroc = 0.
    best_acc = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        avg_loss = train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device
        )

        if (
                'efficientnetv2' in args.arch or 'efficientnet_b5_ns' in args.arch
                or 'efficientnet_b5_ns-detect' in args.arch or 'efficientnetv2-detect' in args.arch
        ):
            scheduler.step()

        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, args, device, epoch, mapper=mapper, attr_embs=attr_embs, logger=logger
        )
        args.valid_folds['prediction'] = predictions
        #do validation on train folds
        _, tr_predictions = valid_fn(
            train_eval_loader, model, criterion, args, device, epoch, mapper=mapper, attr_embs=attr_embs, logger=logger
        )
        args.train_folds['prediction'] = tr_predictions

        valid_agg = None
        train_agg = None
        if args.dataset.lower() == "vindr":
            valid_agg = args.valid_folds
            train_agg = args.train_folds
        elif args.dataset.lower() == "rsna":
            valid_agg = args.valid_folds[['patient_id', 'laterality', args.label, 'prediction', 'fold']].groupby(
                ['patient_id', 'laterality']).mean()
            train_agg = args.train_folds[['patient_id', 'laterality', args.label, 'prediction', 'fold']].groupby(
                ['patient_id', 'laterality']).mean()
        elif args.dataset.lower() == "csaw":
            valid_agg = args.valid_folds[['anon_filename', args.label, 'prediction']].groupby(
                ['anon_filename']).mean()
            train_agg = args.train_folds[['anon_filename', args.label, 'prediction']].groupby(
                ['anon_filename']).mean()

        if args.label.lower() == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
            correct_predictions = (valid_agg[args.label] == valid_agg['prediction']).sum()
            correct_tr_predictions = (train_agg[args.label] == train_agg['prediction']).sum()
            total_predictions = len(valid_agg)
            total_tr_predictions = len(train_agg)
            accuracy = correct_predictions / total_predictions
            tr_accuracy = correct_tr_predictions / total_tr_predictions
            valid_agg[args.label] = valid_agg[args.label].astype(int)
            train_agg[args.label] = train_agg[args.label].astype(int)
            valid_agg['prediction'] = valid_agg['prediction'].astype(int)
            train_agg['prediction'] = train_agg['prediction'].astype(int)
            f1 = f1_score(valid_agg[args.label], valid_agg['prediction'], average='macro')
            tr_f1= f1_score(train_agg[args.label], train_agg['prediction'], average='macro')

            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  '
                f'val accuracy: {accuracy * 100:.4f}   val f1: {f1:.4f}'f'train accuracy: {tr_accuracy * 100:.4f}   train f1: {tr_f1:.4f}'
            )
            logger.add_scalar(f'valid/{args.label}/accuracy', accuracy, epoch + 1)
            if best_acc < accuracy:
                best_acc = accuracy
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_{args.label}_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save Best val acc: {best_acc * 100:.4f} Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'accuracy': accuracy,
                        'f1': f1,
                    }, args.chk_pt_path #/ model_name
                )


        else:
            aucroc = auroc(valid_agg[args.label].values, valid_agg['prediction'].values)
            tr_aucroc= auroc(train_agg[args.label].values, train_agg['prediction'].values)
            elapsed = time.time() - start_time
            print(
                f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s'
            )
            print(f'Epoch {epoch + 1} - Val AUC-ROC Score: {aucroc:.4f} Train AUC-ROC Score: {tr_aucroc:.4f}')
            logger.add_scalar(f'valid/{args.label}/AUC-ROC', aucroc, epoch + 1)

            if best_aucroc < aucroc:
                best_aucroc = aucroc
                model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_{args.label}_ver{args.VER}.pth'
                print(f'Epoch {epoch + 1} - Save val aucroc: {best_aucroc:.4f} Model')
                torch.save(
                    {
                        'model': model.state_dict(),
                        'predictions': predictions,
                        'epoch': epoch,
                        'auroc': aucroc,
                    }, args.chk_pt_path
                )
        # Log metrics to wandb
        wandb_log = {
                    "epoch": epoch + 1,  # Current epoch
                    "train_loss": avg_loss,  # Average training loss
                    "valid_loss": avg_val_loss} # Average validation loss}

        if args.label.lower() == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_acc_cancer_ver{args.VER}.pth'
            print(f'[Fold{args.cur_fold}], Best val Accuracy: {best_acc * 100:.4f}')
            wandb_log.update({
                "train_accuracy": tr_accuracy,
                "train_f1_score": tr_f1,
                "valid_accuracy": accuracy,
                "valid_f1_score": f1,
            })
        else:
            wandb_log.update({
                "train_AUC-ROC": tr_aucroc,
                "valid_AUC-ROC": aucroc,
            })
            model_name = f'{args.model_base_name}_seed_{args.seed}_fold{args.cur_fold}_best_aucroc_ver{args.VER}.pth'
            print(f'[Fold{args.cur_fold}], val AUC-ROC Score: {best_aucroc:.4f}')
        predictions = torch.load(args.chk_pt_path, map_location=device,weights_only=False)['predictions']
        args.valid_folds['prediction'] = predictions
        # Send metrics to wandb
        wandb.log(wandb_log)

    print(f"Training complete. Best Val AUC-ROC: {best_aucroc:.4f}, Best Val Accuracy: {best_acc:.4f}")
    torch.cuda.empty_cache()
    gc.collect()
    return args.valid_folds
def inference_loop(args,device):
    print(f'================== fold: {args.cur_fold} validating ======================')
    print(args.valid_folds.shape)
    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "cancer_birads":
        n_class = 5
    elif args.label.lower() == "birads":
        n_class = 3
    else:
        n_class = 1

    if args.label.lower() == "density" or args.label.lower() == "birads":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to('cuda')
        print(f'pos_wt: {pos_wt}')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')


    # Load the pre-trained CLIP model configuration
    if args.clip_chk_pt_path is not None:
        clip_ckpt = torch.load(args.clip_chk_pt_path, map_location=device, weights_only=False)
        #print keys of the checkpoint
        print(f'clip_ckpt: {clip_ckpt.keys()}')
        config = clip_ckpt["config"]["model"]["image_encoder"]
        print("config of pre-trained: ", config)
        if clip_ckpt["config"]["model"]["image_encoder"]["model_type"] == "swin":
            args.image_encoder_type = clip_ckpt["config"]["model"]["image_encoder"]["model_type"]
        elif clip_ckpt["config"]["model"]["image_encoder"]["model_type"] == "cnn":
            args.image_encoder_type = clip_ckpt["config"]["model"]["image_encoder"]["name"]
    else:
        raise ValueError("Pre-trained CLIP model checkpoint path is required.")
    # Load the fine-tuned checkpoint
    finetuned_ckpt = torch.load(args.chk_pt_path, map_location=device, weights_only=False)
    print(f'finetuned_ckpt: {finetuned_ckpt.keys()}')
    print(f'finetuned_ckpt_model: {finetuned_ckpt["model"].keys()}')

    model = BreastClipClassifier(args, ckpt=clip_ckpt, n_class=n_class)
    print("Model is loaded")

    model = model.to(device)
    model.eval()
    _,valid_loader,_ = get_single_gpu_dataloader_RSNA(args)
    _, predictions = valid_fn(
        valid_loader, model, criterion, args, device, epoch=0)
    # Store predictions in the validation folds
    args.valid_folds['prediction'] = predictions

    """
    predictions = torch.load(
        args.chk_pt_path,
        map_location='cpu', weights_only=False)['predictions']
    print(f'predictions: {predictions.shape}', type(predictions))
    args.valid_folds['prediction'] = predictions
    """

    if args.label.lower() == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
        key = "density"
        valid_agg = args.valid_folds[['patient_id', 'laterality', key, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
        gt = valid_agg[key].values.astype(int)
        pred = valid_agg['prediction'].values.astype(int)
        accuracy = (gt == pred).mean()
        f1 = f1_score(gt, pred, average='macro')
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
    else:
        key = args.label
        valid_agg = args.valid_folds[['patient_id', 'laterality', key, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
        gt = valid_agg[key].values.astype(np.float32)
        pred = valid_agg['prediction'].values.astype(np.float32)
        aucroc = auroc(gt.astype(int), pred)
        print(f'AUC-ROC: {aucroc:.4f}')
    return args.valid_folds.copy()

def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, mapper, attr_embs, logger, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    start = end = time.time()
    print(f"Training for epoch {epoch + 1}/{args.epochs}...")

    progress_iter = tqdm(enumerate(train_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        if (
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
            inputs = inputs.squeeze(1)

        batch_size = inputs.size(0)
        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({'img': inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                scores = torch.nn.functional.normalize(scores, p=2, dim=1)
                inputs_dict = {'img': inputs, 'scores': scores}
                with torch.cuda.amp.autocast(enabled=args.apex):
                    y_preds = model(inputs_dict)
        else:
            with torch.cuda.amp.autocast(enabled=args.apex):
                y_preds = model(inputs)
        if args.label == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # batch scheduler
        # scheduler.step()
        if 'breast_clip' in args.arch:
            scheduler.step()
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]['lr']],
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}'
                  .format(epoch + 1, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))

        if step % args.log_freq == 0 or step == (len(train_loader) - 1):
            index = step + len(train_loader) * epoch
            logger.add_scalar('train/epoch', epoch, index)
            logger.add_scalar('train/iter_loss', losses.avg, index)
            logger.add_scalar('train/iter_lr', optimizer.param_groups[0]['lr'], index)

    return losses.avg


def valid_fn(valid_loader, model, criterion, args, device, epoch=1, mapper=None, attr_embs=None, logger=None):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = time.time()

    progress_iter = tqdm(enumerate(valid_loader), desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                         total=len(valid_loader))
    for step, data in progress_iter:
        inputs = data['x'].to(device)
        batch_size = inputs.size(0)
        if (
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b5_period_n_lp" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_ft" or
                args.arch.lower() == "upmc_breast_clip_det_b2_period_n_lp" or
                args.arch.lower() == "upmc_vindr_breast_clip_det_b2_period_n_lp"
        ):
            inputs = inputs.squeeze(1).permute(0, 3, 1, 2)
        elif args.arch.lower() == 'swin_tiny_custom_norm' or args.arch.lower() == 'swin_base_custom_norm':
            inputs = inputs.squeeze(1)

        if mapper is not None:
            with torch.cuda.amp.autocast(enabled=args.apex):
                pred = mapper({'img': inputs})
                img_embs = torch.nn.functional.normalize(pred["region_proj_embs"].float(), dim=2)
                if args.label.lower() == "mass":
                    img_emb = img_embs[:, 0, :]
                    txt_emb = attr_embs[0, :]
                elif args.label.lower() == "suspicious_calcification":
                    img_emb = img_embs[:, 1, :]
                    txt_emb = attr_embs[1, :]
                scores = img_emb @ txt_emb
                scores = scores.view(batch_size, -1)
                inputs_dict = {'img': inputs, 'scores': scores}
                with torch.no_grad():
                    y_preds = model(inputs_dict)
        else:
            with torch.no_grad():
                y_preds = model(inputs)

        if args.label == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
            labels = data['y'].to(torch.long).to(device)
            loss = criterion(y_preds, labels)
        else:
            labels = data['y'].float().to(device)
            loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))

        losses.update(loss.item(), batch_size)

        if args.label == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
            _, predicted = torch.max(y_preds, 1)
            preds.extend(predicted.cpu().numpy())
        else:
            preds.append(y_preds.squeeze(1).sigmoid().cpu().numpy())


        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))

        if (step % args.log_freq == 0 or step == (len(valid_loader) - 1)) and logger is not None:
            index = step + len(valid_loader) * epoch
            logger.add_scalar('valid/iter_loss', losses.avg, index)

    if args.label == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
        predictions = np.array(preds)
    else:
        predictions = np.concatenate(preds)
    return losses.avg, predictions
def loaded_inference_loop(args,device):
    print(f'================== fold: {args.cur_fold} validating ======================')
    print(args.valid_folds.shape)

    predictions = torch.load(
        args.chk_pt_path,
        map_location='cpu', weights_only=False)['predictions']
    print(f'predictions: {predictions.shape}', type(predictions))
    args.valid_folds['prediction'] = predictions


    if args.label.lower() == "density" or args.label.lower() == "birads":
        key = "density"
        valid_agg = args.valid_folds[['patient_id', 'laterality', key, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
        gt = valid_agg[key].values.astype(int)
        pred = valid_agg['prediction'].values.astype(int)
        accuracy = (gt == pred).mean()
        f1 = f1_score(gt, pred, average='macro')
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        wandb.log({"valid/accuracy": accuracy, "valid/f1_score": f1})
    elif args.label.lower() == "cancer_birads":
        key = "cancer_birads"
        valid_agg = args.valid_folds[['patient_id', 'laterality', key, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
        gt = valid_agg[key].values.astype(int)
        pred = valid_agg['prediction'].values.astype(int)
        accuracy = (gt == pred).mean()
        f1 = f1_score(gt, pred, average='macro')
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        wandb.log({"valid/accuracy": accuracy, "valid/f1_score": f1})
    else:
        key = args.label
        valid_agg = args.valid_folds[['patient_id', 'laterality', key, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
        gt = valid_agg[key].values.astype(np.float32)
        print(f"Class distribution: {np.bincount(gt.astype(int))}")
        pred = valid_agg['prediction'].values.astype(np.float32)
        aucroc = auroc(gt.astype(int), pred)
        print(f'AUC-ROC: {aucroc:.4f}')
        wandb.log({"valid/AUC-ROC": aucroc})
        # Find the optimal threshold using the ROC curve
        fpr, tpr, thresholds = roc_curve(gt.astype(int), pred)
        optimal_idx = (tpr - fpr).argmax()  # Maximize TPR - FPR
        optimal_threshold = thresholds[optimal_idx]
        print(f'Optimal Threshold: {optimal_threshold:.4f}')
        # Identify misclassified examples using the optimal threshold
        misclassified = valid_agg[(gt.astype(int) != (pred > optimal_threshold).astype(int))]
        num_misclassified = len(misclassified)
        print(f'Number of misclassified examples: {num_misclassified}')
        # Find false negatives (gt = 1, pred = 0 based on optimal threshold)
        false_negatives = valid_agg[(gt.astype(int) == 1) & ((pred > optimal_threshold).astype(int) == 0)]
        num_false_negatives = len(false_negatives)
        print(f'Number of false negatives: {num_false_negatives}')
        # Find false positives (gt = 0, pred = 1 based on optimal threshold)
        false_positives = valid_agg[(gt.astype(int) == 0) & ((pred > optimal_threshold).astype(int) == 1)]
        num_false_positives = len(false_positives)
        print(f'Number of false positives: {num_false_positives}')
        # Print IDs of false negatives and false positives
        print('False Negatives IDs:')
        print(false_negatives.index.tolist())
        print('False Positives IDs:')
        print(false_positives.index.tolist())
        false_negatives.to_csv('false_negatives.csv')
        false_positives.to_csv('false_positives.csv')
    return args.valid_folds.copy()

def unpretrained_inference_loop(args,device):
    print(f'================== fold: {args.cur_fold} validating ======================')
    print(args.valid_folds.shape)
    if args.label.lower() == "density":
        n_class = 4
    elif args.label.lower() == "cancer_birads":
        n_class = 5
    elif args.label.lower() == "birads":
        n_class = 3
    else:
        n_class = 1

    if args.label.lower() == "density" or args.label.lower() == "birads":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.weighted_BCE == "y":
        pos_wt = torch.tensor([args.BCE_weights[f"fold{args.cur_fold}"]]).to('cuda')
        print(f'pos_wt: {pos_wt}')
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_wt)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    args.image_encoder_type = args.im_encoder
    model = BreastClipClassifier(args, ckpt=None, n_class=n_class)
    print("Model is loaded without pre-trained configuration")
    model = model.to(device)
    model.eval()
    _,valid_loader,_ = get_single_gpu_dataloader_RSNA(args)
    _, predictions = valid_fn(
        valid_loader, model, criterion, args, device, epoch=0)
    # Store predictions in the validation folds
    args.valid_folds['prediction'] = predictions

    if args.label.lower() == "density" or args.label.lower() == "birads" or args.label.lower() == "cancer_birads":
        key = "density"
        valid_agg = args.valid_folds[['patient_id', 'laterality', key, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
        gt = valid_agg[key].values.astype(int)
        pred = valid_agg['prediction'].values.astype(int)
        accuracy = (gt == pred).mean()
        f1 = f1_score(gt, pred, average='macro')
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')
        # Log metrics to wandb
        wandb.log({"valid/accuracy": accuracy, "valid/f1_score": f1})
    else:
        key = args.label
        valid_agg = args.valid_folds[['patient_id', 'laterality', key, 'prediction', 'fold']].groupby(
            ['patient_id', 'laterality']).mean()
        gt = valid_agg[key].values.astype(np.float32)
        pred = valid_agg['prediction'].values.astype(np.float32)
        aucroc = auroc(gt.astype(int), pred)
        print(f'AUC-ROC: {aucroc:.4f}')
        # Log metrics to wandb
        wandb.log({"valid/AUC-ROC": aucroc})
    return args.valid_folds.copy()