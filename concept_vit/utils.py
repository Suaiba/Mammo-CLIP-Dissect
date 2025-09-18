import os,sys
import math
import numpy as np
import torch
import clip
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.clip import BreastClip
from transformers import BertTokenizerFast
from torchvision import transforms
from transformers.tokenization_utils import PreTrainedTokenizer, AddedToken
from tqdm import tqdm
from torch.utils.data import DataLoader
import data_utils,argparse
from torchinfo import summary
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PM_SUFFIX = {"max":"_max", "avg":""}


def encode_image(model,image: torch.Tensor,device):
    with torch.no_grad():
        img_emb = model.encode_image(image.to(device))
        print(f"Encoded image features shape: {img_emb.shape}")
        img_emb = model.image_projection(img_emb) if model.projection else img_emb
        print(f"Projected image features shape: {img_emb.shape}")
        img_emb = img_emb / torch.norm(img_emb, dim=1, keepdim=True)
    return img_emb.detach().cpu().numpy()
def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc or ViT neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if type(output) is tuple:
                output = output[0]
            #print("Output shape in hook:", output.shape)
            if len(output.shape)==4: #CNN layers
                outputs.append(output.mean(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                print("No pooling for FC layers")
                outputs.append(output.detach())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4: #CNN layers
                outputs.append(output.amax(dim=[2,3]).detach())
            elif len(output.shape)==3: #ViT
                outputs.append(output[:, 0].clone())
            elif len(output.shape)==2: #FC layers
                outputs.append(output.detach())
    return hook

def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                             PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name
def save_combined_target_activations(target_model, dataset, save_name, target_layers=["layer4"], batch_size=20,
                                     device="cuda", pool_mode='avg'):
    """
    Save activations for a target model.
    Args:
        save_name: Save file path, should include {} which will be formatted by layer names.
    """
    print("Saving target activations")
    _make_save_dir(save_name)
    save_names = {target_layer: save_name.format(target_layer) for target_layer in target_layers}
    if _all_saved(save_names):
        return
    all_features = {target_layer: [] for target_layer in target_layers}
    # Register hooks for the target layers
    hooks = {}
    for target_layer in target_layers:
        print(f"Registering hook for target layer: {target_layer}")
        command = f"target_model.{target_layer}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))"
        hooks[target_layer] = eval(command)
    try:
        with torch.no_grad():
            for batch in tqdm(DataLoader(dataset, batch_size=50, num_workers=1)):
                # Check if the batch is a dictionary (vindr format)
                if isinstance(batch, dict):
                    images = batch["images"]
                    #print(f"Shape of images before squeeze: {images.shape}")
                    if images.dim() == 5:  # Check if there is an extra dimension
                        # Handle shape [batch_size, 1, height, width, channels]
                        images = images.squeeze(1).permute(0, 4, 1,
                                                           2)  # Permute to [batch_size, channels, height, width]
                    else:
                        # Handle shape [batch_size, height, width, channels]
                        images = images.permute(0, 3, 1, 2)  # Permute to [batch_size, channels, height, width]
                else:
                    # Otherwise, assume it's a tuple (images, labels)
                    images, _ = batch
                    if images.dim() == 5:  # Check if there is an extra dimension
                        images = images.squeeze(1)  # Remove the extra dimension
                        images = images.permute(0, 3, 1, 2)
                    images = images.to(device)
                # Forward pass through the target model
                target_model.encode_image(images)
        # Save activations for each target layer
        for target_layer in target_layers:
            print(f"Saving activations for layer: {target_layer}")
            torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
            hooks[target_layer].remove()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Free memory
        del all_features
        torch.cuda.empty_cache()
    return
def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    print("Saving target activations")
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=batch_size, num_workers=1)):
            features = target_model.encode_image(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_target_vindr_activations(target_model, dataloader, save_name, target_layers=["layer4"], batch_size=1000,
                            device="cuda", pool_mode='avg',target_name="breastclip"):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    print("Saving target activations")
    _make_save_dir(save_name)
    save_names = {}
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)

    if _all_saved(save_names):
        return

    all_features = {target_layer: [] for target_layer in target_layers}

    hooks = {}
    for target_layer in target_layers:
        print("Registering hook for target layer:", target_layer)
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(
            target_layer)
        hooks[target_layer] = eval(command)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch["images"] = batch["images"].squeeze(1).permute(0, 3, 1, 2)

            if target_name == "breastclip":
                features = target_model.encode_image(batch["images"].to(device))
            elif target_name == "breastclip_classifier":
                features = target_model(batch["images"].to(device))

            else:
                raise ValueError(f"Unknown target model name: {target_name}")

            #features = encode_image(model=target_model, image=batch["images"].to(device), device=device)

    for target_layer in target_layers:
        print("target_layer is:",target_layer)
        #print shape
        print("Shape of all features for target layer {}: {}".format(target_layer, all_features[target_layer][0].shape))
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        # check if all_features[target_layer] is empty
        if len(all_features[target_layer]) == 0:
            print(f"Warning: No features saved for layer {target_layer}. Check if the layer name is correct.")
        hooks[target_layer].remove()
    # free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_csaw_target_activations(target_model, dataloader, save_name, target_layers=["layer4"], batch_size=1000,
                            device="cuda", pool_mode='avg',target_name="breastclip"):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    print("Saving target activations")
    _make_save_dir(save_name)
    save_names = {}
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)

    if _all_saved(save_names):
        return

    all_features = {target_layer: [] for target_layer in target_layers}

    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(
            target_layer)
        hooks[target_layer] = eval(command)

    with torch.no_grad():
        for image,_ in tqdm(dataloader):
            images = image
            if target_name == "breastclip":
                features = target_model.encode_image(images.to(device))
            elif target_name == "breastclip_classifier":
                features = target_model(images.to(device))
            else:
                raise ValueError(f"Unknown target model name: {target_name}")

            #features = encode_image(model=target_model, image=batch["images"].to(device), device=device)

    for target_layer in target_layers:
        print("target_layer is:",target_layer)
        #print("hi:",(all_features[target_layer]))
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    # free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_combined_image_features(model, dataset, save_name, batch_size=20, device="cuda"):
    print("Saving image features")
    if os.path.exists(save_name):
        print("File already exists")
        return
    _make_save_dir(save_name)
    all_features = []
    try:
        with torch.no_grad():
            for batch in tqdm(DataLoader(dataset, batch_size=50, num_workers=1)):
                # Check if the batch is a dictionary (vindr format)
                if isinstance(batch, dict):
                    images = batch["images"]
                    #print(f"Shape of images before squeeze: {images.shape}")
                    if images.dim() == 5:  # Check if there is an extra dimension
                        images = images.squeeze(1)  # Remove the extra dimension
                    #print(f"Shape of images after squeeze: {images.shape}")
                    images = images.permute(0, 3, 1, 2)  # Permute to [batch_size, channels, height, width]
                else:
                    # Otherwise, assume it's a tuple (images, labels)
                    images, _ = batch
                    if images.dim() == 5:  # Check if there is an extra dimension
                        images = images.squeeze(1)  # Remove the extra dimension
                        images = images.permute(0, 3, 1, 2)
                    #print(f"Shape of images after permute: {images.shape}")


                # Encode images
                images = images.to(device)
                features = model.encode_image(images)
                features = model.image_projection(features) if model.projection else features
                all_features.append(features.cpu())
        # Save all features
        torch.save(torch.cat(all_features), save_name)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Free memory
        del all_features
        torch.cuda.empty_cache()
    return

def save_clip_image_features(model, dataset, save_name, batch_size , device = "cuda"):
    print("Saving image features")
    if os.path.exists(save_name):
        print("File already exists")
        return
    _make_save_dir(save_name)
    all_features = []
    try:
        with torch.no_grad():
            for images, labels in tqdm(DataLoader(dataset, batch_size=20,num_workers=1)):
                #permute images to (batch_size, channels, height, width) whereas now it is (batch_size, channels,width,height)
                images = images.permute(0, 1, 3, 2)
                #print("Image shape non vindr:", images.shape)
                images = images.to(device)
                #print("Image shape after permute and to device:", images.shape)
                features = model.encode_image(images)
                features = model.image_projection(features) if model.projection else features
                all_features.append(features.cpu())
        torch.save(torch.cat(all_features), save_name)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Free memory
        del all_features
        torch.cuda.empty_cache()
    return

def save_clip_vindr_image_features(model, dataloader, save_name, batch_size=1000 , device = "cuda"):
    print("Saving image features")

    if os.path.exists(save_name):
        print("File already exists")
        return

    _make_save_dir(save_name)
    all_features = []
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch["images"] = batch["images"].squeeze(1).permute(0, 3, 1, 2)
                #print(f"Batch image shape: {batch['images'].shape}")
                features = model.encode_image(batch["images"].to(device))
                #print(f"Encoded image features shape: {features.shape}")
                features = model.image_projection(features) if model.projection else features
                #print(f"Projected image features shape: {features.shape}")
                #print(f"Features shape: {features.shape}")
                all_features.append(features)
        torch.save(torch.cat(all_features), save_name)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Free memory
        del all_features
        torch.cuda.empty_cache()
    return

def save_clip_csaw_image_features(model, dataloader, save_name, batch_size=1000 , device = "cuda"):
    print("Saving image features")

    if os.path.exists(save_name):
        print("File already exists")
        return
    _make_save_dir(save_name)
    all_features = []
    try:
        with torch.no_grad():
            for image,_ in tqdm(dataloader):
                #print("image shape:",image.shape)
                images = image
                #images = image.permute(0, 3, 1, 2)
                #print(f"Batch image shape: {batch['images'].shape}")
                features = model.encode_image(images.to(device))
                #print(f"Encoded image features shape: {features.shape}")
                features = model.image_projection(features) if model.projection else features
                #print(f"Projected image features shape: {features.shape}")
                #print(f"Features shape: {features.shape}")
                all_features.append(features)
        torch.save(torch.cat(all_features), save_name)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Free memory
        del all_features
        torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    print("Saving text features")
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return


def save_clip_vindr_text_features(model, text, save_name, batch_size=1000, device="cuda"):
    print("Saving text features")
    if os.path.exists(save_name):
        print("File already exists")
        return
    # Ensure text is a dictionary when passed
    _make_save_dir(save_name)
    text_features = []
    # Move the text input to the same device as the model
    text = {key: value.to(device) for key, value in text.items()}

    # Pass text directly as a dictionary
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text['input_ids']) / batch_size))):  # Use 'input_ids' from the dictionary
            batch_text = {key: value[batch_size * i:batch_size * (i + 1)] for key, value in
                          text.items()}  # Slice batches
            encoded_text = model.encode_text(batch_text)  # Pass the batch as a dictionary
            projected_text_fts = model.text_projection(encoded_text) if model.projection else encoded_text
            text_features.append(projected_text_fts)  # Pass the batch as a dictionary

    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return
def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            encoded_features=model.encode_text(text[batch_size*i:batch_size*(i+1)])
            #print(f"Encoded text features shape: {encoded_features.shape}")
            projected_fts = model.text_projection(encoded_features) if model.projection else encoded_features
            #print(f"Projected text features shape: {projected_fts.shape}")
            text_features.append(projected_fts)
    text_features = torch.cat(text_features, dim=0)
    return text_features

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir,breast_clip_ckh=None,fine_tuned_ckh=None,args=None):
    #clip_model, clip_preprocess = clip.load(clip_name, device=device)
    print("Saving activations")
    model_config = {'name': 'clip_custom', 'temperature': 0.07,
                    'image_encoder': {'source': 'cnn', 'name': 'tf_efficientnet_b5_ns-detect', 'pretrained': True,
                                      'model_type': 'cnn'},
                    'text_encoder': {'source': 'huggingface', 'name': 'emilyalsentzer/Bio_ClinicalBERT',
                                     'pretrained': True, 'gradient_checkpointing': False, 'pooling': 'eos',
                                     'cache_dir': '/ocean/projects/asc170022p/shg121/PhD/Breast-CLIP/src/codebase/outputs/huggingface/',
                                     'trust_remote_code': True, 'mlm_head': True},
                    'projection_head': {'name': 'linear', 'dropout': 0.1, 'proj_dim': 512}}
    print("Loading model")
    loss_config = {'breast_clip': {'label_smoothing': 0.0, 'i2i_weight': 1.0, 't2t_weight': 0.5, 'loss_ratio': 1.0}}
    tokenizer = BertTokenizerFast.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    print("Going into clip_model")
    clip_model = BreastClip(model_config, loss_config, tokenizer)




    if breast_clip_ckh is not None:
        ckpt = torch.load(breast_clip_ckh, map_location=device, weights_only=False)
        print("Loading pre-trained Breast-CLIP checkpoint")
        clip_model.load_state_dict(ckpt["model"], strict=False)
        print(" Pre-trained Breast-CLIP model loaded")

        if fine_tuned_ckh is not None:
            print("Finetuned checkpoint provided")
            finetuned_ckpt = torch.load(fine_tuned_ckh, map_location=device, weights_only=False)
            save_text="/newest_{}_cancer_finetuned_".format(d_probe)
        else:
            finetuned_ckpt = None
            save_text="/latest_{}_mammo_pretrained_".format(d_probe)
    else:
        finetuned_ckpt = None
        print("No pre-trained or finetuned checkpoint provided")

        save_text="/Latest_{}_not_mammo_pretrained_".format(d_probe)

    clip_model = clip_model.to(device)

    clip_preprocess = None
    if target_name == "breastclip_classifier":
        clip_ckpt = torch.load(breast_clip_ckh, map_location=device, weights_only=False)
        print("Loading finetuned Breast-CLIP classifier")
        target_model, target_preprocess = data_utils.get_target_model(target_name, device,args=args, ckpt=clip_ckpt, n_class=args.num_class, finetuned_ckpt=finetuned_ckpt)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
        if breast_clip_ckh is not None:
            target_ckpt = torch.load(breast_clip_ckh, map_location=device, weights_only=False)
            print("Loading pre-trained target checkpoint")
            target_model.load_state_dict(target_ckpt["model"], strict=False)
            print("Pre-trained target model loaded")
    #print len of target layers
    #print(len(target_model.image_encoder._blocks),flush=True)

    #setup data
    print('Getting data')
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    #print len of data
    #print("Length of data_c:",len(data_c))
    #print("Length of data_t:",len(data_t))
    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    #ignore empty lines
    words = [i for i in words if i!=""]
    #print("Untokenized example:",words[:5])
    print('Tokenizing text')
    
    #text = clip_model.tokenize(["{}".format(word) for word in words]).to(device)
    text = clip_model.tokenize(["{}".format(word) for word in words])
    #print some bits of text
    #print("tokenized example:",text[:5])

    
    save_names = get_save_names(clip_name = clip_name, target_name = target_name,
                                target_layer = '{}', d_probe = d_probe, concept_set = concept_set,
                                pool_mode=pool_mode, save_dir = save_dir)

    target_save_name, clip_save_name, text_save_name = save_names
    #prefix to save_name
    target_save_name = save_dir + save_text + target_save_name
    clip_save_name = save_dir + save_text + clip_save_name
    text_save_name = save_dir + save_text + text_save_name
    print("Name of saved activations:")
    print("target_save_name:",target_save_name)
    print("clip_save_name:",clip_save_name)
    print("text_save_name:",text_save_name)


    if d_probe!='vindr' and d_probe!='imagenet_subsets' and d_probe!='combined' and d_probe!="embed_png" and d_probe!="embed_implant" and d_probe!="embed_non_implant" and d_probe!="embed_non_implant_100" and d_probe!="embed_marker_84" and d_probe!="embed_marker_only":
        #print("Probe is not VinDr")
        if d_probe=='csaw' or d_probe=='csaw_all_splits':
            print("Probe is CSAW")
            text=text.to(device)
            save_clip_vindr_text_features(clip_model, text, text_save_name, batch_size)
            save_clip_csaw_image_features(clip_model, data_c, clip_save_name, batch_size, device)
            save_csaw_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)
        else:
            print("Probe is not VinDR or CSAW or ImageNet subsets or combined")
            text=text.to(device)
            save_clip_text_features(clip_model, text, text_save_name, batch_size)
            save_clip_csaw_image_features(clip_model, data_c, clip_save_name, batch_size, device)
            save_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)
    else:
        if d_probe=="vindr":
            print("Probe is VinDr")
        elif d_probe=="imagenet_subsets":
            print("Probe is Imagenet subsets")
        elif d_probe=="combined":
            print("Probe is combination of VinDr and Imagenet subsets")
        elif d_probe=="embed_png" or d_probe=="embed_implant" or d_probe=="embed_non_implant" or d_probe=="embed_non_implant_100" or d_probe=="embed_marker_84" or d_probe=="embed_marker_only":
            print("Probe is EMBED based")
        save_clip_vindr_text_features(clip_model, text, text_save_name, batch_size)
        if d_probe=="vindr":
            save_clip_vindr_image_features(clip_model, data_c, clip_save_name, batch_size, device)
            save_target_vindr_activations(target_model, data_t, target_save_name, target_layers,
                            batch_size, device, pool_mode,target_name)
        elif d_probe=="imagenet_subsets" or d_probe=="embed_png" or d_probe=="embed_implant" or d_probe=="embed_non_implant" or d_probe=="embed_non_implant_100" or d_probe=="embed_marker_84" or d_probe=="embed_marker_only":
            save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
            save_target_activations(target_model, data_t, target_save_name, target_layers,
                                    batch_size, device, pool_mode)
        elif d_probe=="combined":
            save_combined_image_features(clip_model, data_c, clip_save_name, batch_size, device)
            save_combined_target_activations(target_model, data_t, target_save_name, target_layers,
                                    batch_size, device, pool_mode)



    return
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True, device="cuda", d_probe="vindr",top_k=100):
    print("clip_save_name:",clip_save_name)
    print("text_save_name:",text_save_name)
    image_features = torch.load(clip_save_name, map_location='cpu', weights_only=True).float()
    text_features = torch.load(text_save_name, map_location='cpu', weights_only=True).float()
    if d_probe == 'vindr' or d_probe == 'csaw' or d_probe == 'csaw_all_splits':
        # Move image features and text features to the same device
        image_features = image_features.to(device)
        text_features = text_features.to(device)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")


        # Example: If they have different sizes, project image features to match the text feature dimension (512).
        #if d_probe == 'vindr':
            # Project image features from 2048 to 512
            #weight = torch.randn(2048, 512).to(device)  # Correct the dimensions
            #image_features = torch.matmul(image_features, weight)
            # Project text features from 768 to 512
            #weight_text = torch.randn(768, 512).to(device)
            #text_features = torch.matmul(text_features, weight_text)
            #image_features = torch.nn.functional.linear(image_features,
                                                    #torch.randn(2048, 512).to(device))  # Linear projection

        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    print(f"Clip features shape: {clip_feats.shape}")
    #put clip_feats on device
    clip_feats = clip_feats.to(device)
    target_feats = torch.load(target_save_name, map_location=device, weights_only=True)
    print(f"Target features shape: {target_feats.shape}")
    similarity = similarity_fn(clip_feats, target_feats, device=device,top_k=top_k)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
def get_dist_from_target_activations(target_save_name, device="cuda",neuron_idx=None):
    target_feats = torch.load(target_save_name, map_location=device, weights_only=True)
    neuron_activations = target_feats[:, neuron_idx] if neuron_idx is not None else target_feats
    print(f"Neuron activations shape: {neuron_activations.shape}")
    return neuron_activations
def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))

def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


def get_clip_feats(clip_save_name, text_save_name, device="cuda", d_probe="vindr"):
    """
    Computes and returns the CLIP features (similarity matrix) between image features and text features.
    Args:
        clip_save_name (str): Path to the saved image features.
        text_save_name (str): Path to the saved text features.
        device (str): Device to load the features onto (default: "cuda").
        d_probe (str): Dataset probe type (default: "vindr").
    Returns:
        torch.Tensor: The computed CLIP features (similarity matrix).
    """
    print("clip_save_name:", clip_save_name)
    print("text_save_name:", text_save_name)
    # Load image and text features
    image_features = torch.load(clip_save_name, map_location='cpu', weights_only=True).float()
    text_features = torch.load(text_save_name, map_location='cpu', weights_only=True).float()
    # Move features to the specified device if necessary
    if d_probe in ['vindr', 'csaw', 'csaw_all_splits']:
        image_features = image_features.to(device)
        text_features = text_features.to(device)
    # Normalize features
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        # Compute CLIP features (similarity matrix)
        clip_feats = image_features @ text_features.T
    # Free up memory
    del image_features, text_features
    torch.cuda.empty_cache()
    print(f"Clip features shape: {clip_feats.shape}")
    return clip_feats
    