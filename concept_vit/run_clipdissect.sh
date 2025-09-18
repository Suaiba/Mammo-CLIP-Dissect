#!/bin/bash
#BREAST CLIP



python concept_vit/describe_broad_neurons.py --target_model "breastclip" --target_layers "image_encoder._blocks[0],image_encoder._blocks[1],image_encoder._blocks[2],image_encoder._blocks[3],image_encoder._blocks[4],image_encoder._blocks[5],image_encoder._blocks[6],image_encoder._blocks[7],image_encoder._blocks[8],image_encoder._blocks[9],image_encoder._blocks[10],image_encoder._blocks[11], \
 image_encoder._blocks[12],image_encoder._blocks[13],image_encoder._blocks[14],image_encoder._blocks[15],image_encoder._blocks[16],image_encoder._blocks[17],image_encoder._blocks[18],image_encoder._blocks[19],image_encoder._blocks[20],image_encoder._blocks[21],image_encoder._blocks[22],image_encoder._blocks[23], \
 image_encoder._blocks[24],image_encoder._blocks[25],image_encoder._blocks[26],image_encoder._blocks[27],image_encoder._blocks[28],image_encoder._blocks[29],image_encoder._blocks[30],image_encoder._blocks[31],image_encoder._blocks[32],image_encoder._blocks[33],image_encoder._blocks[34],image_encoder._blocks[35], \
 image_encoder._blocks[36],image_encoder._blocks[37],image_encoder._blocks[38]" --d_probe "vindr" --concept_set "/storage/MammoCLIP_copy/src/Concepts/Specific_concepts_sorted.txt" --Breast_clip_chkpt "/storage/BreastCLIP_models/b5-model-best-epoch-7.tar"


#BREAST CLIP CLASSIFIER: DENSITY
#python /storage/MammoCLIP_copy/src/concept_vit/describe_broad_neurons.py --target_model "breastclip_classifier" --target_layers "image_encoder._blocks[0],image_encoder._blocks[1],image_encoder._blocks[2],image_encoder._blocks[3],image_encoder._blocks[4],image_encoder._blocks[5],image_encoder._blocks[6],image_encoder._blocks[7],image_encoder._blocks[8],image_encoder._blocks[9],image_encoder._blocks[10],image_encoder._blocks[11], \
 #image_encoder._blocks[12],image_encoder._blocks[13],image_encoder._blocks[14],image_encoder._blocks[15],image_encoder._blocks[16],image_encoder._blocks[17],image_encoder._blocks[18],image_encoder._blocks[19],image_encoder._blocks[20],image_encoder._blocks[21],image_encoder._blocks[22],image_encoder._blocks[23], \
 #image_encoder._blocks[24],image_encoder._blocks[25],image_encoder._blocks[26],image_encoder._blocks[27],image_encoder._blocks[28],image_encoder._blocks[29],image_encoder._blocks[30],image_encoder._blocks[31],image_encoder._blocks[32],image_encoder._blocks[33],image_encoder._blocks[34],image_encoder._blocks[35], \
 #image_encoder._blocks[36],image_encoder._blocks[37],image_encoder._blocks[38]" --d_probe "vindr" --concept_set "/storage/MammoCLIP_copy/src/Specific_concepts_sorted.txt" --Breast_clip_chkpt "/storage/BreastCLIP_models/b5-model-best-epoch-7.tar" --finetuned_img_classifier_chkpt "/storage/BreastCLIP_models/New_finetuned/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_density_ver084.pth" --num_class 4


#BREAST CLIP CLASSIFIER: CALC
#python /storage/MammoCLIP_copy/src/concept_vit/describe_broad_neurons.py --target_model "breastclip_classifier" --target_layers "image_encoder._blocks[0],image_encoder._blocks[1],image_encoder._blocks[2],image_encoder._blocks[3],image_encoder._blocks[4],image_encoder._blocks[5],image_encoder._blocks[6],image_encoder._blocks[7],image_encoder._blocks[8],image_encoder._blocks[9],image_encoder._blocks[10],image_encoder._blocks[11], \
 #image_encoder._blocks[12],image_encoder._blocks[13],image_encoder._blocks[14],image_encoder._blocks[15],image_encoder._blocks[16],image_encoder._blocks[17],image_encoder._blocks[18],image_encoder._blocks[19],image_encoder._blocks[20],image_encoder._blocks[21],image_encoder._blocks[22],image_encoder._blocks[23], \
 #image_encoder._blocks[24],image_encoder._blocks[25],image_encoder._blocks[26],image_encoder._blocks[27],image_encoder._blocks[28],image_encoder._blocks[29],image_encoder._blocks[30],image_encoder._blocks[31],image_encoder._blocks[32],image_encoder._blocks[33],image_encoder._blocks[34],image_encoder._blocks[35], \
 #image_encoder._blocks[36],image_encoder._blocks[37],image_encoder._blocks[38]" --d_probe "vindr" --concept_set "/storage/MammoCLIP_copy/src/Specific_concepts_sorted.txt" --Breast_clip_chkpt "/storage/BreastCLIP_models/b5-model-best-epoch-7.tar" --finetuned_img_classifier_chkpt "/storage/BreastCLIP_models/New_finetuned/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_Suspicious_Calcification_ver084.pth" --num_class 1

#BREAST CLIP CLASSIFIER: MASS
#python /storage/MammoCLIP_copy/src/concept_vit/describe_broad_neurons.py --target_model "breastclip_classifier" --target_layers "image_encoder._blocks[0],image_encoder._blocks[1],image_encoder._blocks[2],image_encoder._blocks[3],image_encoder._blocks[4],image_encoder._blocks[5],image_encoder._blocks[6],image_encoder._blocks[7],image_encoder._blocks[8],image_encoder._blocks[9],image_encoder._blocks[10],image_encoder._blocks[11], \
 #image_encoder._blocks[12],image_encoder._blocks[13],image_encoder._blocks[14],image_encoder._blocks[15],image_encoder._blocks[16],image_encoder._blocks[17],image_encoder._blocks[18],image_encoder._blocks[19],image_encoder._blocks[20],image_encoder._blocks[21],image_encoder._blocks[22],image_encoder._blocks[23], \
 #image_encoder._blocks[24],image_encoder._blocks[25],image_encoder._blocks[26],image_encoder._blocks[27],image_encoder._blocks[28],image_encoder._blocks[29],image_encoder._blocks[30],image_encoder._blocks[31],image_encoder._blocks[32],image_encoder._blocks[33],image_encoder._blocks[34],image_encoder._blocks[35], \
 #image_encoder._blocks[36],image_encoder._blocks[37],image_encoder._blocks[38]" --d_probe "vindr" --concept_set "/storage/MammoCLIP_copy/src/Specific_concepts_sorted.txt" --Breast_clip_chkpt "/storage/BreastCLIP_models/b5-model-best-epoch-7.tar" --finetuned_img_classifier_chkpt "/storage/BreastCLIP_models/New_finetuned/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_Mass_ver084.pth" --num_class 1


#BREAST CLIP CLASSIFIER: CANCER
#python /storage/MammoCLIP_copy/src/concept_vit/describe_broad_neurons.py --target_model "breastclip_classifier" --target_layers "image_encoder._blocks[0],image_encoder._blocks[1],image_encoder._blocks[2],image_encoder._blocks[3],image_encoder._blocks[4],image_encoder._blocks[5],image_encoder._blocks[6],image_encoder._blocks[7],image_encoder._blocks[8],image_encoder._blocks[9],image_encoder._blocks[10],image_encoder._blocks[11], \
 #image_encoder._blocks[12],image_encoder._blocks[13],image_encoder._blocks[14],image_encoder._blocks[15],image_encoder._blocks[16],image_encoder._blocks[17],image_encoder._blocks[18],image_encoder._blocks[19],image_encoder._blocks[20],image_encoder._blocks[21],image_encoder._blocks[22],image_encoder._blocks[23], \
 #image_encoder._blocks[24],image_encoder._blocks[25],image_encoder._blocks[26],image_encoder._blocks[27],image_encoder._blocks[28],image_encoder._blocks[29],image_encoder._blocks[30],image_encoder._blocks[31],image_encoder._blocks[32],image_encoder._blocks[33],image_encoder._blocks[34],image_encoder._blocks[35], \
 #image_encoder._blocks[36],image_encoder._blocks[37],image_encoder._blocks[38]" --d_probe "vindr" --concept_set "/storage/MammoCLIP_copy/src/Specific_concepts_sorted.txt" --Breast_clip_chkpt "/storage/BreastCLIP_models/b5-model-best-epoch-7.tar" --finetuned_img_classifier_chkpt "/storage/BreastCLIP_models/New_finetuned/upmc_breast_clip_det_b5_period_n_ft_seed_10_fold0_best_acc_cancer_birads_ver084.pth" --num_class 5




