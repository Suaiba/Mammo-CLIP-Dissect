import warnings

import torch

from Classifiers.single_gpu_experiments import do_experiments
from utils import get_Paths, seed_all

warnings.filterwarnings("ignore")
import argparse
import os,wandb
import pickle


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard-path', metavar='DIR',
                        default='/storage2/log',
                        help='path to tensorboard logs')
    parser.add_argument('--checkpoints', metavar='DIR',
                        default='/restricted/projectnb/batmanlab/shawn24/PhD/Mammo-CLIP/checkpoints',
                        help='path to checkpoints')
    parser.add_argument('--output_path', metavar='DIR',default='/storage2/outputs',help='path to output logs')
    parser.add_argument(
        "--data-dir",
        default="/restricted/projectnb/batmanlab/shawn24/PhD/RSNA_Breast_Imaging/Dataset",
        type=str, help="Path to data file"
    )
    parser.add_argument(
        "--img-dir", default="RSNA_Cancer_Detection/train_images_png", type=str, help="Path to image file"
    )

    parser.add_argument("--clip_chk_pt_path", default=None, type=str, help="Path to Mammo-CLIP chkpt")
    parser.add_argument("--csv-file", default="RSNA_Cancer_Detection/final_rsna.csv", type=str,
                        help="data csv file")
    parser.add_argument("--dataset", default="RSNA", type=str, help="Dataset name? (RSNA or VinDr or CSAW)")
    parser.add_argument("--data_frac", default=1.0, type=float, help="Fraction of data to be used for training")
    parser.add_argument(
        "--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str,
        help="For b5 classification, [upmc_breast_clip_det_b5_period_n_lp for linear probe and  upmc_breast_clip_det_b5_period_n_ft for finetuning]. "
             "For b2 classification, [upmc_breast_clip_det_b2_period_n_lp for linear probe and  upmc_breast_clip_det_b2_period_n_ft for finetuning].")
    parser.add_argument("--label", default="cancer", type=str,
                        help="cancer for RSNA or cancer_birads,Mass, Suspicious_Calcification, density for VinDr")
    parser.add_argument("--detector-threshold", default=0.1, type=float)
    parser.add_argument("--swin_encoder", default="microsoft/swin-tiny-patch4-window7-224", type=str)
    parser.add_argument("--pretrained_swin_encoder", default="y", type=str)
    parser.add_argument("--swin_model_type", default="y", type=str)
    parser.add_argument("--im_encoder", default="tf_efficientnetv2-detect", type=str,help="Image encoder type for unpretrained breastclip classifier model")
    parser.add_argument("--VER", default="084", type=str)
    parser.add_argument("--epochs-warmup", default=0, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--alpha", default=10.0, type=float)
    parser.add_argument("--sigma", default=15.0, type=float)
    parser.add_argument("--p", default=1.0, type=float)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--focal-alpha", default=0.6, type=float)
    parser.add_argument("--focal-gamma", default=2.0, type=float)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_folds", default=4, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=9, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--print-freq", default=5000, type=int)
    parser.add_argument("--log-freq", default=1000, type=int)
    parser.add_argument("--running-interactive", default='n', type=str)
    parser.add_argument("--inference-mode", default='n', type=str)
    parser.add_argument('--model-type', default="Classifier", type=str)
    parser.add_argument("--weighted-BCE", default='n', type=str)
    parser.add_argument("--balanced-dataloader", default='n', type=str)

    return parser.parse_args()


def main(args):
    # Initialize wandb
    wandb.init(
        project="MammoCLIP_classifier_finetune",  # Replace with your project name
        config=args,  # Log all args as config
    )
    seed_all(args.seed)
    # get paths
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.root = f"lr_{args.lr}_epochs_{args.epochs}_weighted_BCE_{args.weighted_BCE}_balanced_dataloader_{args.balanced_dataloader}_{args.label}_data_frac_{args.data_frac}_post_miccai"
    args.root = f"lr_{args.lr}_epochs_{args.epochs}_weighted_BCE_{args.weighted_BCE}_{args.label}_data_frac_{args.data_frac}"
    args.apex = True if args.apex == "y" else False
    args.pretrained_swin_encoder = True if args.pretrained_swin_encoder == "y" else False
    args.swin_model_type = True if args.swin_model_type == "y" else False
    args.running_interactive = True if args.running_interactive == "y" else False

    chk_pt_path, output_path, tb_logs_path = get_Paths(args)
    if args.inference_mode == "n":
        #args.chk_pt_path = os.path.join(chk_pt_path, f"{args.arch}_seed_{args.seed}_fold{args.start_fold}_best_acc_cancer_ver{args.VER}.pth")
        args.chk_pt_path = os.path.join(chk_pt_path, f"{args.arch}_seed_{args.seed}_fold{args.start_fold}_best_acc_{args.label}_ver{args.VER}.pth")
        #args.chk_pt_path = chk_pt_path
    elif args.inference_mode == "y":
        args.chk_pt_path = "/scratch/project_465001915/salahudd/MammoCLIP_classfier_checkpoints/single_gpu/VinDr/Classifier//"

    elif  args.inference_mode == "load":
        if args.arch=="upmc_breast_clip_det_b5_period_n_ft":
            if args.label.lower()=="mass":
                print("Loading VinDr mass model")
                args.chk_pt_path = "/scratch/project_465001915/salahudd/MammoCLIP_classfier_checkpoints/single_gpu/VinDr/Classifier/upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_30_weighted_BCE_y_Mass_data_frac_1.0/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_Mass_ver084.pth"
            elif args.label.lower()=="suspicious_calcification":
                print("Loading VinDr calc model")
                args.chk_pt_path = "/scratch/project_465001915/salahudd/MammoCLIP_classfier_checkpoints/single_gpu/VinDr/Classifier/upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_30_weighted_BCE_y_Suspicious_Calcification_data_frac_1.0/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_Suspicious_Calcification_ver084.pth"
            elif args.label.lower()=="density":
                args.chk_pt_path = "/scratch/project_465001915/salahudd/MammoCLIP_classfier_checkpoints/single_gpu/VinDr/Classifier/upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_30_weighted_BCE_y_density_data_frac_1.0/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_density_ver084.pth"
            elif args.label.lower()=="cancer_birads":
                args.chk_pt_path = "/scratch/project_465001915/salahudd/MammoCLIP_classfier_checkpoints/single_gpu/VinDr/Classifier/upmc_breast_clip_det_b5_period_n_ft/lr_5e-05_epochs_30_weighted_BCE_y_cancer_birads_data_frac_1.0/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_cancer_birads_ver084.pth"
            else:
                raise ValueError("Invalid label for VinDr dataset. Choose from 'mass', 'suspicious_calcification', 'density', 'cancer_birads'.")
        elif args.arch=="upmc_breast_clip_det_b2_period_n_ft":
            args.chk_pt_path = "/scratch/project_465001915/salahudd/MammoCLIP_classfier_checkpoints/single_gpu/VinDr/Classifier/upmc_breast_clip_det_b2_period_n_ft/"
            if args.label.lower()=="mass":
                args.chk_pt_path += ""
            elif args.label.lower()=="suspicious_calcification":
                args.chk_pt_path += ""
            elif args.label.lower()=="density":
                args.chk_pt_path += ""
            elif args.label.lower()=="cancer_birads":
                args.chk_pt_path += ""

    elif args.inference_mode == "unpretrained_inf":
        args.chk_pt_path = None
    elif args.inference_mode == "train_un_mammopretrained":
        args.clip_chk_pt_path=None
        args.chk_pt_path = os.path.join(chk_pt_path,f"Un_mammo_pretrained_{args.arch}_seed_{args.seed}_fold{args.start_fold}_best_acc_{args.label}_ver{args.VER}.pth")
    #args.chk_pt_path= "/cephfs/Downstream_evalualtion_b5_fold0/classification/Models/Classifier/fine_tune/mass/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_cancer_ver084.pth"
    #args.chk_pt_path="/cephfs/Downstream_evalualtion_b5_fold0/classification/Models/Classifier/fine_tune/calcification/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_aucroc_ver084.pth"
    #args.chk_pt_path="/cephfs/Downstream_evalualtion_b5_fold0/classification/Models/Classifier/fine_tune/density/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_cancer_ver084.pth"
    args.output_path = output_path
    args.tb_logs_path = tb_logs_path

    os.makedirs(chk_pt_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(tb_logs_path, exist_ok=True)
    print("====================> Paths <====================")
    print(f"checkpoint_path: {chk_pt_path}")
    print(f"output_path: {output_path}")
    print(f"tb_logs_path: {tb_logs_path}")
    print('device:', device)
    print('torch version:', torch.__version__)
    print("====================> Paths <====================")

    pickle.dump(args, open(os.path.join(output_path, f"seed_{args.seed}_train_configs.pkl"), "wb"))
    torch.cuda.empty_cache()

    if args.weighted_BCE == "y" and args.dataset.lower() == "rsna" and args.label.lower() == "cancer":
        args.BCE_weights = {
            "fold0": 46.48148148148148,
            "fold1": 46.01830663615561,
            "fold2": 46.41339491916859,
            "fold3": 46.05747126436781
        }
    elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "mass":
        args.BCE_weights = {
            "fold0": 15.573306370070778,
            "fold1": 15.573306370070778,
            "fold2": 15.573306370070778,
            "fold3": 15.573306370070778
        }
    elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "suspicious_calcification":
        args.BCE_weights = {
            "fold0": 37.296728971962615,
            "fold1": 37.296728971962615,
            "fold2": 37.296728971962615,
            "fold3": 37.296728971962615,
        }

    elif args.weighted_BCE == "y" and args.dataset.lower() == "vindr" and args.label.lower() == "focal_asymmetry":
        args.BCE_weights = {
            "fold0": 74.88425925925925,
            "fold1": 74.88425925925925,
            "fold2": 74.88425925925925,
            "fold3": 74.884259259259255,
        }

    if args.balanced_dataloader == "y":
        args.sampler_weights = {
            "fold0": {"pos_wt": 0.003401360544217687, "neg_wt": 7.469375560203167e-05},
            "fold1": {"pos_wt": 0.0035211267605633804, "neg_wt": 7.503001200480192e-05},
            "fold2": {"pos_wt": 0.003424657534246575, "neg_wt": 7.48839299086416e-05},
            "fold3": {"pos_wt": 0.003472222222222222, "neg_wt": 7.419498441905327e-05},
        }

    do_experiments(args, device)


if __name__ == "__main__":
    args = config()
    main(args)