import os
import argparse
import datetime
import json, wandb
import pandas as pd
import torch

import og_utils
import similarity
from Classifiers.models.breast_clip_classifier import BreastClipClassifier

# Initialize wandb
wandb.init(project="clip-dissect", name="mammo_pre_clip_broader_describe_neurons", config={})
parser = argparse.ArgumentParser(description='CLIP-Dissect')

parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                    choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                    help="Which CLIP-model to use")
parser.add_argument("--num_class", type=int, default=1, help="Number of classes for classification task")
parser.add_argument("--target_model", type=str, default="resnet50",
                    help=""""Which model to dissect, supported options are pretrained imagenet models from
                        torchvision and resnet18_places""")
parser.add_argument("--target_layers", type=str, default="conv1,layer1,layer2,layer3,layer4",
                    help="""Which layer neurons to describe. String list of layer names to describe, separated by comma(no spaces). 
                          Follows the naming scheme of the Pytorch module used""")
parser.add_argument("--d_probe", type=str, default="vindr",
                    choices=["imagenet_broden", "cifar100_val", "imagenet_val", "broden", "imagenet_broden", "vindr", "imagenet_subsets",
                             "csaw", "csaw_all_splits"])
parser.add_argument("--concept_set", type=str, default="data/20k.txt", help="Path to txt file containing concept set")
parser.add_argument("--batch_size", type=int, default=200, help="Batch size when running CLIP/target model")
parser.add_argument("--device", type=str, default="cuda", help="whether to use GPU/which gpu")
parser.add_argument("--activation_dir", type=str, default="saved_activations", help="where to save activations")
parser.add_argument("--result_dir", type=str, default="/storage2/Broader_clip_dissect", help="where to save results")
parser.add_argument("--pool_mode", type=str, default="avg", help="Aggregation function for channels, max or avg")
parser.add_argument("--similarity_fn", type=str, default="soft_wpmi", choices=["soft_wpmi", "wpmi", "rank_reorder",
                                                                               "cos_similarity",
                                                                               "cos_similarity_cubed"])
parser.add_argument("--Breast_clip_chkpt", type=str, default=None,
                    help="Path to Breast-CLIP checkpoint prettrained on UPMC & VinDr")
# parser.add_argument("--finetuned_img_classifier_chkpt", type=str, default="/cephfs/Downstream_evalualtion_b5_fold0/classification/Models/Classifier/fine_tune/density/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_cancer_ver084.pth", help="Path to fine-tuned Breast-CLIP image classifier checkpoint")
parser.add_argument("--finetuned_img_classifier_chkpt", type=str, default=None,
                    help="Path to fine-tuned Breast-CLIP image classifier checkpoint")
parser.add_argument(
    "--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str,
    help="For b5 classification, [upmc_breast_clip_det_b5_period_n_lp for linear probe and  upmc_breast_clip_det_b5_period_n_ft for finetuning]. "
         "For b2 classification, [upmc_breast_clip_det_b2_period_n_lp for linear probe and  upmc_breast_clip_det_b2_period_n_ft for finetuning].")
parser.parse_args()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    print("Running CLIP-Dissect")
    args = parser.parse_args()
    args.target_layers = args.target_layers.split(",")
    wandb.config.update(vars(args))
    """
    if args.Breast_clip_chkpt is not None:
        clip_ckpt = torch.load(args.Breast_clip_chkpt, map_location=args.device, weights_only=False)
    if args.finetuned_img_classifier_chkpt is not None:
        finetuned_ckpt = torch.load(args.finetuned_img_classifier_chkpt, map_location=args.device, weights_only=False)
    model = BreastClipClassifier(args, ckpt=clip_ckpt, n_class=4, finetuned_ckpt=finetuned_ckpt)
    #print the model layer names
    print(model.state_dict().keys())
    exit()
    """
    similarity_fn = eval("similarity.{}".format(args.similarity_fn))

    og_utils.save_activations(clip_name=args.clip_model, target_name=args.target_model,
                           target_layers=args.target_layers, d_probe=args.d_probe,
                           concept_set=args.concept_set, batch_size=args.batch_size,
                           device=args.device, pool_mode=args.pool_mode,
                           save_dir=args.activation_dir, breast_clip_ckh=args.Breast_clip_chkpt,
                           fine_tuned_ckh=args.finetuned_img_classifier_chkpt, args=args)

    print("Activations saved")
    wandb.log({"status": "activations_saved"})

    outputs = {"layer": [], "unit": [], "description": [], "similarity": [], "images": []}
    with open(args.concept_set, 'r') as f:
        words = (f.read()).split('\n')

    for target_layer in args.target_layers:
        print("Target layer: {}".format(target_layer))
        save_names = og_utils.get_save_names(clip_name=args.clip_model, target_name=args.target_model,
                                          target_layer=target_layer, d_probe=args.d_probe,
                                          concept_set=args.concept_set, pool_mode=args.pool_mode,
                                          save_dir=args.activation_dir)
        target_save_name, clip_save_name, text_save_name = save_names
        target_save_name = f"{args.activation_dir}/clip_dissector_{args.target_model}_{args.d_probe}_small_not_mammo_pretrained_{target_save_name}"
        clip_save_name = f"{args.activation_dir}/clip_dissector_{args.target_model}_{args.d_probe}_small_not_mammo_pretrained_{clip_save_name}"
        text_save_name = f"{args.activation_dir}/clip_dissector_{args.target_model}_{args.d_probe}_small_not_mammo_pretrained_{text_save_name}"

        similarities, target_feats = og_utils.get_similarity_from_activations(
            target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=True,
            device=args.device, d_probe=args.d_probe)

        print("Similarity calculated for layer: {}".format(target_layer))
        wandb.log({"layer": target_layer, "similarity_calculated": True})
        #vals, ids = torch.max(similarities, dim=1)
        vals, ids = torch.topk(similarities, k=10, dim=1)
        _, top_ids = torch.topk(target_feats, k=5, dim=0)

        del similarities, target_feats
        torch.cuda.empty_cache()

        #descriptions = [words[int(idx)] for idx in ids]
        descriptions = []

        print(f"Length of words: {len(words)}")
        for id in ids:
            print(f"id: {id}")
            descriptions.append([words[int(idx)] for idx in id])

        outputs["unit"].extend([i for i in range(len(vals))])
        outputs["layer"].extend([target_layer] * len(vals))
        outputs["description"].extend(descriptions)
        outputs["similarity"].extend(vals.cpu().numpy())
        outputs["images"].extend(top_ids.T.cpu().numpy())

        del top_ids, vals, ids
        torch.cuda.empty_cache()
    print("All layers processed, saving results")
    wandb.log({"status": "all_layers_processed"})
    df = pd.DataFrame(outputs)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    save_path = "{}/{}_{}".format(args.result_dir, args.target_model,
                                  datetime.datetime.now().strftime("%y_%m_%d_%H_%M"))
    os.mkdir(save_path)
    if args.Breast_clip_chkpt is not None:
        if args.finetuned_img_classifier_chkpt is not None:
            save_csv_text = "clip_dissector_vindr_den_finetuned_breast_clip_classifier_descriptions.csv"
            save_txt = "clip_dissector_vindr_den_finetuned_breast_clip_classifier_descriptions_args.txt"
        else:
            if args.d_probe == "vindr":
                save_csv_text = "clip_dissector_vindr_mammo_pretrained_breast_clip_classifier_descriptions.csv"
                save_txt = "clip_dissector_vindr_mammo_pretrained_breast_clip_classifier_descriptions_args.txt"
            elif args.d_probe == "imagenet_subsets":
                save_csv_text = "clip_dissector_imagenet_subsets_small_mammo_pretrained_breast_clip_classifier_descriptions.csv"
                save_txt = "clip_dissector_imagenet_subsets_small_mammo_pretrained_breast_clip_classifier_descriptions_args.txt"
    else:
        if args.d_probe == "vindr":
            save_csv_text = "clip_dissector_vindr_not_mammo_pretrained_breast_clip_descriptions.csv"
            save_txt = "clip_dissector_vindr_not_mammo_pretrained_breast_clip_descriptions_args.txt"
        elif args.d_probe == "imagenet_subsets":
            save_csv_text = "clip_dissector_clip_target_imagenet_subsets_small_not_mammo_pretrained_clip_descriptions.csv"
            save_txt = "clip_dissector_clip_target_imagenet_subsets_small_not_mammo_pretrained_breast_clip_descriptions_args.txt"
    df.to_csv(os.path.join(save_path, save_csv_text), index=False)
    with open(os.path.join(save_path, save_txt), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    wandb.log({"status": "results_saved"})
    wandb.finish()
    print("Dissection done! Results saved to {}".format(save_path))