# AmGANet: Attribute-Modulated Geometric Alignment Attention for Language-Guided Medical Image Segmentation
Official implementation of **AmGANet**.

> **Manuscript status**  
> This work is currently **under review** at *Information Fusion*.


> **Publication note**  
> If the manuscript is accepted for publication, this repository and any related materials will be updated to reflect the final publication information and applicable sharing policies.

## Overview
![main figure](assets\3D.png)
> **<p align="justify"> Abstract:** *Medical text has recently been introduced into medical image segmentation to incorporate expert knowledge and alleviate the limitations of purely vision-based models. However, existing methods often rely on paired image-text inputs and underexploit attribute-level spatial cues, resulting in insufficient geometric guidance and limited cross-modal interaction. To address these issues, we propose the Attribute-Modulated Geometric Alignment Language-guided Attention Network (AmGANet). Specifically, the Instance-Aware Cross-Modal Adaptation module enables structured semantic guidance without requiring paired image-text inputs at inference time through task-aware visual adaptation, image-conditioned textual modulation, and decoupled retrieval of lesion count and location attributes. Meanwhile, the proposed Attribute-Modulated Geometric Alignment Attention transforms attribute-level textual priors into a global affine field and adaptively incorporates it into image-driven local deformation modeling for more accurate spatial localization. In addition, a Synergistic Semantic Decoding Block hierarchically refines semantic and spatial representations through bidirectional modulation, promoting deeper visual-textual alignment and more effective cross-modal interaction. Experiments on four public datasets demonstrate the effectiveness of AmGANet and its strong and consistent performance over existing methods.* </p>


## Method
 
<p float="left">
  <img src="assets\overview.png" width="100%" />
</p>

1) **a novel Attribute-Level Language-Guided Medical Image Segmentation framework**: The method overcomes the rigid constraints of conventional static image-text pairing, achieving instance-aware dynamic adaptation of textual guidance and explicit geometric alignment of lesions, thereby fundamentally addressing the limitations of textual flexibility and spatial geometric localization in multimodal segmentation.
2) **Instance-Aware Cross-Modal Adaptation (IACMA)** enables structured attribute-guided report-free inference through task-aware visual adaptation, image-conditioned textual modulation, and decoupled count/location retrieval.
3) **Attribute-Modulated Geometric Alignment Attention (AGA)** transforms attribute-level textual priors into a global affine field and adaptively integrates it with image-driven local deformation modeling, thereby alleviating geometric ambiguity and localization drift in language-guided segmentation.
4) **Synergistic Semantic Decoding Block (SSD)** bridges the modality gap between semantics and vision. Through a layer-wise anchoring and bidirectional refinement strategy, it progressively refines semantic and spatial feature representations, deeply mining implicit correlations beyond independent modalities.



## Requirements

This project is tested with the following environment:

- Python 3.10.18
- torch==2.0.1+cu118
- torchvision==0.15.2+cu118

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Then downgrade `setuptools` to version `59.5.0` by uninstalling the default `setuptools==82.0.1` included in Python 3.10.18:

```bash
pip uninstall -y setuptools
pip install setuptools==59.5.0
```

## Usage

### 1. Data Preparation

#### 1.1. QaTa-COV19, MosMedData+, and MoNuSeg Datasets (demo dataset)

The original data can be downloaded from the following links:

- QaTa-COV19 Dataset and paired text reports - [Link](https://github.com/HUANGLIZI/LViT)
- MosMedData+ Dataset and paired text reports - [Link](https://github.com/HUANGLIZI/LViT)
- BUSI Dataset and paired text reports - [Link](https://tahakoleilat.github.io/MedCLIPSeg/)
- Kvasir-seg Dataset and paired text reports - [Link](https://tahakoleilat.github.io/MedCLIPSeg/)

The paired text reports for the **MosMedData+** and **QaTa-COV19** datasets were provided by **LViT** - [Original Link](https://github.com/HUANGLIZI/LViT).

The paired text reports for the **BUSI** and **Kvasir-seg** datasets were provided by **MedCLIPSeg** - [Original Link](https://tahakoleilat.github.io/MedCLIPSeg/).


#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── QaTa-Covid19
    │   ├── Test_Folder
    |   |   ├── Test_text.xlsx
    │   │   ├── images
    │   │   └── labels
    │   ├── Train_Folder
    |   |   ├── Train_text.xlsx
    │   │   ├── images
    │   │   └── labels
    │   └── Val_Folder
    |	    ├── Val_text.xlsx
    │       ├── images
    │       └── labels
    └── MosMedDataPlus
        ├── Test_Folder
        |   ├── Test_text.xlsx
        │   ├── images
        │   └── labels
        ├── Train_Folder
        |   ├── Train_text.xlsx
        │   ├── images
        │   └── labels
        └── Val_Folder
            ├── Val_text.xlsx
            ├── images
            └── labels
```


### 2. Training

### 2.1. Training

For the QaTa-COV19 and MosMedData+ datasets, run:
python Train_convid19/train_covid19.py

For the BUSI and Kvasir-SEG datasets, run:
python Train_BUSI_KVASIR/train_model_BUSI.py


### 3. Evaluation

#### 3.1. Test the Model and Visualize the Segmentation Results
For the QaTa-COV19 and MosMedData+ datasets, run:
python Train_convid19/test_covid19.py

For the BUSI and Kvasir-SEG datasets, run:
python Train_BUSI_KVASIR/test_model_BUSI.py

### 4. Pretrained Weights

The pretrained weights for **AmGANet** can be downloaded from Google Drive:

- [Download pretrained weights](https://drive.google.com/drive/folders/1r3iGSDQlJO31yGNGg7a60zrTYN_RfsRu?usp=sharing)

After downloading, place the checkpoint file in the appropriate directory before evaluation or inference.
