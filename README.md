# AmGANet: Attribute-Modulated Geometric Alignment Attention for Language-Guided Medical Image Segmentation

Official implementation of **AmGANet**, proposed in our paper:  
**Attribute-Modulated Geometric Alignment Attention for Language-Guided Medical Image Segmentation**

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
Python == 3.10.18 and install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
Questions about NumPy version conflict. The NumPy version we use is 1.17.5. We can install bert-embedding first, and install NumPy then.


## Usage

### 1. Data Preparation
#### 1.1. QaTa-COV19, MosMedData+ and MoNuSeg Datasets (demo dataset)
The original data can be downloaded in following links:
* QaTa-COV19 Dataset - [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

* MosMedData+ Dataset - [Link (Original)](http://medicalsegmentation.com/covid19/) or [Kaggle](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)

* BUSI Dataset - [Link (Original)](http://medicalsegmentation.com/covid19/) or [Kaggle](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)

* Kvasir-seg Dataset - [Link (Original)](http://medicalsegmentation.com/covid19/) or [Kaggle](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)


The text annotation of QaTa-COV19 has been released!

  *(Note: The text annotation of QaTa-COV19 train and val datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDkm5jsTw5dX_RpuRr?e=uaZq6W).
  The partition of train set and val set of QaTa-COV19 dataset [download link](https://1drv.ms/f/c/c3143e7c85766728/QihndoV8PhQggMO2rwAAAAAADo5kj33mUee33g).
  The text annotation of QaTa-COV19 test dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDkj1vvvLt2jDCHqiM?e=954uDF).)*

  ***(Note: The contrastive label is available in the repo.)***
  
***(Note: The text annotation of MosMedData+ train dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDguIIKCRfYB9Z0NL8Dw?e=8rj6rY).
The text annotation of MosMedData+ val dataset [download link](https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMGsQAAAAAAtAgZiRQFYfsAjw).
The text annotation of MosMedData+ test dataset [download link](https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMHsQAAAAAAdHkwXMxGlgU9Tg).)***
  
  *If you use the datasets provided by us, please cite the LViT.*

#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── QaTa-Covid19
    │   ├── Test_Folder
    |   |   ├── Test_text.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    |   |   ├── Train_text.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    |	    ├── Val_text.xlsx
    │       ├── img
    │       └── labelcol
    └── MosMedDataPlus
        ├── Test_Folder
        |   ├── Test_text.xlsx
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        |   ├── Train_text.xlsx
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── Val_text.xlsx
            ├── img
            └── labelcol
```


### 2. Training

### 2.1. Training

```angular2html
python train_model.py
```


### 3. Evaluation

#### 3.1. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase. Then run:
```angular2html
python test_model.py
``` 
