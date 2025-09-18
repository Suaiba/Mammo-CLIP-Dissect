import os,sys
import torch
import pandas as pd
#from PyQt5.QtCore.QTextCodec import kwargs
from torchvision import datasets, transforms, models
from transformers import AutoModelForImageClassification, AutoImageProcessor, AutoModelForPreTraining
#from datasets import load_dataset, concatenate_datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.clip import BreastClip
from transformers import BertTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer, AddedToken
from data import DataModule, CustomImageDatasetFromTxt, EMBED_Dataset,CombinedDataset,EMBED_marker_Dataset
from data.dataset.CSAW_dataset import CSAWDataset,CSAWDataset_all_splits
from Classifiers.models.breast_clip_classifier import BreastClipClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
DATASET_ROOTS = {"imagenet_subsets": "/storage/ImageNet_subsets","imagenet_val": "your_path_to_imagenet_validation",
                "broden": "your_path_to_broden1_224",
                }

MODELS = {

    "vit": "google/vit-base-patch16-224-in21k",
    "dino": "facebook/dinov2-base",
    "clip": "openai/clip-vit-base-patch16",
    "resnet": "microsoft/resnet-50",
    "mae": "facebook/vit-mae-base", 
    "dino-cub": "teresas/dinov2-base-cub",
    "vit-cub": "teresas/vit-base-patch16-224-cub",
    "clip-cub": "teresas/clip-vit-base-patch16-cub",
    "resnet-cub": "teresas/resnet-50-cub",
    "dino-bloodmnist": "teresas/dinov2-base-bloodmnist",
    "vit-bloodmnist": "teresas/vit-base-patch16-224-bloodmnist",
    "clip-bloodmnist": "teresas/clip-vit-base-patch16-bloodmnist",
    "resnet-bloodmnist": "teresas/resnet-50-bloodmnist",
}

def get_target_model(target_name, device, args=None, ckpt=None, n_class=None, finetuned_ckpt=None):
    """
    returns target model in eval mode and its preprocess function
    target_name: supported options - {resnet18_places, resnet18, resnet34, resnet50, resnet101, resnet152}
                 except for resnet18_places this will return a model trained on ImageNet from torchvision
                 
    To Dissect a different model implement its loading and preprocessing function here
    """
    if target_name == "breastclip":
        model_config={'name': 'clip_custom', 'temperature': 0.07, 'image_encoder': {'source': 'cnn', 'name': 'tf_efficientnet_b5_ns-detect', 'pretrained': True, 'model_type': 'cnn'}, 'text_encoder': {'source': 'huggingface', 'name': 'emilyalsentzer/Bio_ClinicalBERT', 'pretrained': True, 'gradient_checkpointing': False, 'pooling': 'eos', 'cache_dir': '/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/outputs/huggingface/', 'trust_remote_code': True, 'mlm_head': True}, 'projection_head': {'name': 'linear', 'dropout': 0.1, 'proj_dim': 512}}
        loss_config={'breast_clip': {'label_smoothing': 0.0, 'i2i_weight': 1.0, 't2t_weight': 0.5, 'loss_ratio': 1.0}}
        tokenizer=BertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        target_model = BreastClip(model_config, loss_config, tokenizer)
        target_model=target_model.to(device)
        preprocess=None
    elif target_name == "breastclip_classifier":
        if args is None or ckpt is None or n_class is None:
            raise ValueError("Arguments `args`, `ckpt`, and `n_class` must be provided for BreastClipClassifier.")
        target_model = BreastClipClassifier(args, ckpt=ckpt, n_class=n_class)
        if finetuned_ckpt is not None:
            print("Loading finetuned Breast-CLIP checkpoint from {}".format(finetuned_ckpt))
            target_model.load_state_dict(finetuned_ckpt['model'])
        target_model = target_model.to(device)
        preprocess = None

    elif target_name in MODELS.keys(): # for models from huggingface
        name = MODELS[target_name]
        if target_name == "mae":
            target_model = AutoModelForPreTraining.from_pretrained(name).to(device)
        else:   
            target_model = AutoModelForImageClassification.from_pretrained(name).to(device)
        preprocess = get_resnet_imagenet_preprocess()
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif "vit_b" in target_name:
        target_name_cap = target_name.replace("vit_b", "ViT_B")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
    elif "resnet" in target_name:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        preprocess = weights.transforms()
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))

    target_model.eval()

    return target_model, preprocess

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
    elif dataset_name == "imagenet_subsets":
        txt_file = "/storage/imagenet_subsets.txt"
        # Load the imagenet_subsets dataset
        target_mean = [0.485, 0.456, 0.406]
        target_std = [0.229, 0.224, 0.225]
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
        data = CustomImageDatasetFromTxt(txt_file=txt_file, transform=preprocess)
        return data
    elif dataset_name == "vindr":

        datamodule = DataModule(
            data_config={'vindr': {'name': 'vindr', 'data_type': 'image_classification_zs', 'data_dir': '/storage', 'img_dir': 'VinDR_MammoCLIP/images_png', 'data_path': 'VinDR-data/vindr_detection_v1_folds.csv', 'text_max_length': 256}},
            dataloader_config={'train': {'pin_memory': True, 'shuffle': True, 'drop_last': True, 'num_workers': 0, 'batch_size': 4}, 'valid': {'pin_memory': True, 'shuffle': False, 'drop_last': False, 'num_workers': 0, 'batch_size': 4}, 'test': {'pin_memory': True, 'shuffle': False, 'drop_last': False, 'num_workers': 4, 'batch_size': 6}},
            tokenizer_config={'source': 'huggingface', 'pretrained_model_name_or_path': 'emilyalsentzer/Bio_ClinicalBERT', 'cache_dir': '/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/outputs/huggingface/tokenizers'},
            transform_config={'train': {'Resize': {'size_h': 1520, 'size_w': 912}, 'transform': {'affine_transform_degree': 20, 'affine_translate_percent': 0.1, 'affine_scale': [0.8, 1.2], 'affine_shear': 20, 'elastic_transform_alpha': 10, 'elastic_transform_sigma': 15, 'p': 1.0}}, 'valid': {'Resize': {'size_h': 1520, 'size_w': 912}}, 'test': {'Resize': {'size_h': 1520, 'size_w': 912}}},
            mean=0.3089279,
            std=0.25053555408335154,
            image_encoder_type="tf_efficientnet_b5_ns-detect",
            cur_fold= 0
        )
        test_dataloader_dict = datamodule.valid_dataloader()
        assert len(test_dataloader_dict) > 0
        dataloader = test_dataloader_dict['vindr']
        #data = datamodule.datasets['valid'][0]
        return dataloader
    elif dataset_name == "vindr_alt":
        datamodule = DataModule(
            data_config={'vindr': {'name': 'vindr', 'data_type': 'image_classification_zs', 'data_dir': '/storage',
                                   'img_dir': 'VinDR_MammoCLIP/images_png',
                                   'data_path': 'VinDR-data/vindr_detection_v1_folds.csv', 'text_max_length': 256}},
            dataloader_config={
                'train': {'pin_memory': True, 'shuffle': True, 'drop_last': True, 'num_workers': 0, 'batch_size': 4},
                'valid': {'pin_memory': True, 'shuffle': False, 'drop_last': False, 'num_workers': 0, 'batch_size': 4},
                'test': {'pin_memory': True, 'shuffle': False, 'drop_last': False, 'num_workers': 4, 'batch_size': 6}},
            tokenizer_config={'source': 'huggingface',
                              'pretrained_model_name_or_path': 'emilyalsentzer/Bio_ClinicalBERT',
                              'cache_dir': '/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/outputs/huggingface/tokenizers'},
            transform_config={'train': {'Resize': {'size_h': 1520, 'size_w': 912},
                                        'transform': {'affine_transform_degree': 20, 'affine_translate_percent': 0.1,
                                                      'affine_scale': [0.8, 1.2], 'affine_shear': 20,
                                                      'elastic_transform_alpha': 10, 'elastic_transform_sigma': 15,
                                                      'p': 1.0}}, 'valid': {'Resize': {'size_h': 1520, 'size_w': 912}},
                              'test': {'Resize': {'size_h': 1520, 'size_w': 912}}},
            mean=0.3089279,
            std=0.25053555408335154,
            image_encoder_type="tf_efficientnet_b5_ns-detect",
            cur_fold=0
        )
        test_dataloader_dict = datamodule.valid_dataloader()
        assert len(test_dataloader_dict) > 0
        dataloader = test_dataloader_dict['vindr']
        dataset = dataloader.dataset  # Extract the dataset from the dataloader
        return dataset  # Return the dataset instead of the dataloader
    elif dataset_name == "combined":
        # Get the individual datasets
        data_imagenet = get_data("imagenet_subsets")
        data_vindr = get_data("vindr_alt")
        # Combine the datasets
        combined_dataset = CombinedDataset(data_imagenet, data_vindr)
        #print("imgnet bit:",combined_dataset[0])  # Should print (image, label)
        #print("vindr bit:",combined_dataset[len(data_imagenet)])  # Should print (image, label) from vindr
        return combined_dataset

    elif dataset_name == "embed_png":
        # Define the root directory containing the PNG images
        #root_dir = "/storage2/Embed_subset/EMBED_SUBSET/"
        root_dir = "/storage2/Embed_subset/EMBED_84/"
        # Define the path to the CSV file containing filenames and labels
        csv_file = "/storage2/Embed_subset/latest_implant_nonimplant_files.csv"
        # Define any transformations to apply to the images
        transform = transforms.Compose([
            transforms.Resize((1520, 912)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        # Create the dataset
        data = EMBED_Dataset(root_dir=root_dir, csv_file=csv_file, transform=transform)

    elif dataset_name == "embed_marker_84":
        # Define the root directory containing the PNG images
        #root_dir = "/storage2/Embed_subset/EMBED_SUBSET/"
        root_dir = "/storage2/Embed_subset/EMBED_marker/"
        # Define the path to the CSV file containing filenames and labels
        csv_file = "/storage2/Embed_subset/latest_marker_or_not_files.csv"
        # Define any transformations to apply to the images
        transform = transforms.Compose([
            transforms.Resize((1520, 912)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        # Create the dataset
        data = EMBED_marker_Dataset(root_dir=root_dir, csv_file=csv_file, transform=transform)

    elif dataset_name == "embed_implant":
        # Define the root directory containing the PNG images
        #root_dir = "/storage2/Embed_subset/EMBED_SUBSET/"
        root_dir = "/storage2/Embed_subset/EMBED_84/"
        # Define the path to the CSV file containing filenames and labels
        csv_file = "/storage2/Embed_subset/latest_only_implant_files.csv"
        # Define any transformations to apply to the images
        transform = transforms.Compose([
            transforms.Resize((1520, 912)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        # Create the dataset
        data = EMBED_Dataset(root_dir=root_dir, csv_file=csv_file, transform=transform)

    elif dataset_name == "embed_marker_only":
        # Define the root directory containing the PNG images
        #root_dir = "/storage2/Embed_subset/EMBED_SUBSET/"
        root_dir = "/storage2/Embed_subset/EMBED_marker/"
        # Define the path to the CSV file containing filenames and labels
        csv_file = "/storage2/Embed_subset/latest_only_marker_files.csv"
        # Define any transformations to apply to the images
        transform = transforms.Compose([
            transforms.Resize((1520, 912)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        # Create the dataset
        data = EMBED_marker_Dataset(root_dir=root_dir, csv_file=csv_file, transform=transform)


    elif dataset_name == "embed_non_implant":
        # Define the root directory containing the PNG images
        #root_dir = "/storage2/Embed_subset/EMBED_SUBSET/"
        root_dir = "/storage2/Embed_subset/EMBED_84/"
        # Define the path to the CSV file containing filenames and labels
        csv_file = "/storage2/Embed_subset/latest_only_nonimplant_files.csv"
        # Define any transformations to apply to the images
        transform = transforms.Compose([
            transforms.Resize((1520, 912)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        # Create the dataset
        data = EMBED_Dataset(root_dir=root_dir, csv_file=csv_file, transform=transform)

    elif dataset_name == "embed_non_implant_100":
        # Define the root directory containing the PNG images
        root_dir = "/storage2/EMBED_NON_IMPLANTS/EMBED_NO_IMPLANTS_100"
        # Define the path to the CSV file containing filenames and labels
        csv_file = "/storage2/EMBED_NON_IMPLANTS/only_nonimplant_100_files.csv"
        # Define any transformations to apply to the images
        transform = transforms.Compose([
            transforms.Resize((1520, 912)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ])
        # Create the dataset
        data = EMBED_Dataset(root_dir=root_dir, csv_file=csv_file, transform=transform)


    elif dataset_name == "csaw":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = CSAWDataset(
            root_folder="/storage/",
            annotation_csv="/storage/csaw_cc_data/CSAW_final_balanced.csv",
            imagefolder_path="/storage2/csaw/csaw_mammo_clip_alt_images",
            split="test",  # change if needed
            transform_config=transform,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader
    elif dataset_name == "csaw_all_splits":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = CSAWDataset_all_splits(
            root_folder="/storage/",
            annotation_csv="/storage/csaw_cc_data/CSAW_final_balanced.csv",
            imagefolder_path="/storage2/csaw/csaw_mammo_clip_alt_images",
            transform_config=transform,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    elif dataset_name == "imagenet_broden":
        data = torch.utils.data.ConcatDataset([datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess), 
                                                     datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess)])

    return data


def get_places_id_to_broden_label():
    with open("data/categories_places365.txt", "r") as f:
        places365_classes = f.read().split("\n")
    
    broden_scenes = pd.read_csv('data/broden1_224/c_scene.csv')
    id_to_broden_label = {}
    for i, cls in enumerate(places365_classes):
        name = cls[3:].split(' ')[0]
        name = name.replace('/', '-')
        
        found = (name+'-s' in broden_scenes['name'].values)
        
        if found:
            id_to_broden_label[i] = name.replace('-', '/')+'-s'
        if not found:
            id_to_broden_label[i] = None
    return id_to_broden_label
    
def get_cifar_superclass():
    cifar100_has_superclass = [i for i in range(7)]
    cifar100_has_superclass.extend([i for i in range(33, 69)])
    cifar100_has_superclass.append(70)
    cifar100_has_superclass.extend([i for i in range(72, 78)])
    cifar100_has_superclass.extend([101, 104, 110, 111, 113, 114])
    cifar100_has_superclass.extend([i for i in range(118, 126)])
    cifar100_has_superclass.extend([i for i in range(147, 151)])
    cifar100_has_superclass.extend([i for i in range(269, 281)])
    cifar100_has_superclass.extend([i for i in range(286, 298)])
    cifar100_has_superclass.extend([i for i in range(300, 308)])
    cifar100_has_superclass.extend([309, 314])
    cifar100_has_superclass.extend([i for i in range(321, 327)])
    cifar100_has_superclass.extend([i for i in range(330, 339)])
    cifar100_has_superclass.extend([345, 354, 355, 360, 361])
    cifar100_has_superclass.extend([i for i in range(385, 398)])
    cifar100_has_superclass.extend([409, 438, 440, 441, 455, 463, 466, 483, 487])
    cifar100_doesnt_have_superclass = [i for i in range(500) if (i not in cifar100_has_superclass)]
    
    return cifar100_has_superclass, cifar100_doesnt_have_superclass