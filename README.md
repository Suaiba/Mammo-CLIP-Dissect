## Mammo-CLIP Dissect

**Mammo-CLIP Dissect** is a concept-based explainability framework for analysing how deep learning (DL) vision models, including mammography-specific Vision–Language Models (VLMs), capture clinically meaningful concepts at different network layers. It adapts and extends concept-based interpretability methods to the mammography domain, enabling systematic investigation of neuron-level capture of text concepts.

This repository contains the concept set, code, scripts, and instructions to reproduce the experiments described in our paper:

> *Mammo-CLIP Dissect: A Framework for Analysing Mammography Concepts in Vision-Language Models.

## 🚧 Work in Progress  

This repository is currently **under active development**.  
Documentation, examples, and additional details will be added soon.  
Please check back for updates or open an issue if you have questions.  

## Features  

- Concept-based dissection of mammography models at different layers  
- Supports multiple setups, including mammography-specific and ImageNet-pretrained models  
- Flexible configuration for \(F_{target}\), \(F_{dissector}\), and \(D_{probe}\) datasets  
- Built on and extends the CLIP-Dissect pipeline to mammography  

---

## Getting Started  

### Prerequisites  

- Python 3.8+  
- PyTorch 1.10+  
- CUDA-enabled GPU (recommended)  
- Dependencies listed in `requirements.txt`  

Install required packages with:  

```bash
pip install -r requirements.txt
```
## Running the Mammo-CLIP Dissect framework
The main entry point for the framework is the run_clipdissect.sh script:
```bash
concept_vit/run_clipdissect.sh
```

- This script sets up and runs Mammo-CLIP Dissect for different experimental configurations.
- Several setups are supported, including G-Mammo-CLIP Dissect, M-Mammo-CLIP Dissect and C-Mammo-CLIP Dissect(as described in the paper).
- Options for setups and experiment variations are indicated in the comments within the scripts.
- Uncomment the relevant lines to run different parts of the pipeline as needed.

The following setups can be run by uncommenting the relevant lines
- G-Mammo-CLIP Dissect: Line 3-6 under "#BREAST CLIP" commenting out the field --Breast_clip_chkpt. This is to use ImageNet pre-trained checkpoints for the MammoCLIP dissector and target models.
- M-Mammo-CLIP Dissect: Line 3-6 under "#BREAST CLIP" WITH out the field --Breast_clip_chkpt included, this loads the Mammography pre-trained checkpoint for the MammoCLIP dissector and target models.
- C-Mammo-CLIP Dissect: Lines 16-33 cover the four variants of  C-Mammo-CLIP Dissect corresponding to the four downstream classification tasks. These have the additional field finetuned_img_classifier_chkpt to input the finetuned classifier checkpoint. Uncomment and run for the task of your choice as needed!
- 
## Fine-tuning the Mammo-CLIP classifier
To finetune the Mammo-CLIP classifier we provide a shell file called 
```
Finetune/MammoCLIP.sh
```
This was run on ROC-M based GPU cluster which uses a SLURM based system. Please modify this file as needed, with special focus on the capitalized folders named **YOUR REPOSITORY HERE**

To fine-tune Mammo-CLIP based classifier for one of the four tasks as described in our paper, set the field --inference-mode to "n" signifying you wish to train and not just run inference. To select task on which to finetune change the field --label.

## Evaluating the finetuned Mammo-CLIP classifier
To evaluate the performance of a finetuned classifier --inference-mode to "load" to load from your own repository with fine-tuned model checkpoints.
## Acknowledgements  

This repository builds on and extends ideas and code from several key projects:  

- **[Mammo-CLIP](https://github.com/batmanlab/Mammo-CLIP)** — the mammography-specific Vision–Language Model we use as a base as presented in the paper **[Mammo-CLIP: A Vision Language Foundation Model to Enhance Data Efficiency and Robustness in Mammography.](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_59)**    
- **[CLIP-Dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect)** — the original CLIP-based dissection framework that inspired our implementation of concept-level neuron analysis from the paper **[CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks.](https://arxiv.org/abs/2204.10965)**  
- **[From Colours to Classes](https://github.com/teresa-sc/concepts_in_ViTs)** — which introduced layer-wise dissection of ViTs using CLIP-dissect in the paper **[From Colors to Classes: Emergence of Concepts in Vision Transformers.](https://arxiv.org/abs/2503.24071)**  

We gratefully acknowledge the authors of these papers and repositories for making their work available to the community.  
