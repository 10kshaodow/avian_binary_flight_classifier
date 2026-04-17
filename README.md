# Bird Flight Image Classifier

## Overview
This project builds a binary image classifier that predicts whether a bird is **in flight** or **not in flight** from still images. The goal is to classify the bird’s state based on its pose and appearance rather than relying on scene context alone.

A key challenge in this task is background bias. Full-scene bird images often contain shortcuts such as open sky, branches, water, or ground textures that can influence predictions. To address this, the pipeline includes a **YOLO-based bird detection and cropping stage** so the classifier can focus more directly on the bird itself.

---

## Project Goal
The main objective is to train and evaluate deep learning models that can distinguish between:
- **in_flight**
- **not_in_flight**

from bird images using transfer learning.

---

## Pipeline
The project follows this workflow:

1. **Dataset preparation**
   - Organize images into binary class folders: `in_flight` and `not_in_flight`
   - Split the dataset into training, validation, and test sets

2. **Bird localization with YOLO**
   - Use a pretrained YOLO detector to locate the bird in each image
   - Crop the image around the bird’s bounding box with padding
   - Save cropped images into the same binary class structure

3. **Image classification**
   - Train pretrained deep learning classifiers on the cropped bird images
   - Evaluate performance using accuracy, F1 score, ROC-AUC, and confusion matrices

4. **Interpretability analysis**
   - Visualize saliency maps / attention behavior
   - Check whether models focus on the bird or background context

---

## Models Used

### 1. ResNet50
**Type:** Convolutional Neural Network (CNN)  
**Pretrained:** Yes, pretrained on **ImageNet**

ResNet50 is a deep CNN that uses **residual connections**, which help the network train effectively without suffering as much from vanishing gradients. It is a strong and reliable baseline for image classification tasks.

**Why it was used:**
- Stable and widely trusted baseline
- Strong transfer learning performance
- Good balance of performance and simplicity

---

### 2. EfficientNet-B3
**Type:** Convolutional Neural Network (CNN)  
**Pretrained:** Yes, pretrained on **ImageNet**

EfficientNet-B3 is a more optimized CNN architecture that scales network depth, width, and image resolution in a balanced way. Compared with older CNNs, it often achieves better accuracy with fewer parameters.

**Why it was used:**
- More parameter-efficient than many standard CNNs
- Strong performance on image classification
- Good candidate for comparing against ResNet50

---

### 3. Vision Transformer (ViT)
**Type:** Transformer-based vision model  
**Pretrained:** Yes, pretrained on **ImageNet** or equivalent pretrained weights depending on the implementation

Unlike CNNs, ViT processes an image as a sequence of patches and uses **self-attention** to model relationships between them. This gives it a more global view of the image compared with convolution-based models.

**Why it was used:**
- Provides a transformer-based comparison against CNNs
- Can capture broader global structure in the image
- Useful for comparing architectural differences on the same task

---

## Model Differences

### CNNs: ResNet50 and EfficientNet-B3
Both ResNet50 and EfficientNet-B3 are **convolutional neural networks**, meaning they learn local spatial features through convolution filters.

- **ResNet50** focuses on depth and residual learning
- **EfficientNet-B3** focuses on scaling the model more efficiently

These models are generally strong for tasks where local visual patterns matter, such as feathers, wing shape, and body posture.

### Transformer: ViT
ViT differs because it does not rely on convolutions in the same way. Instead, it treats the image as a set of patches and learns relationships between them using attention.

This can help the model reason more globally, but it may also require stronger pretraining and careful tuning.

---

## Pretraining
All classification models in this project use **pretrained weights** rather than training from scratch.

### Why pretraining matters
Pretraining helps because:
- the models start with useful visual features already learned
- training converges faster
- performance is better on smaller custom datasets
- fewer labeled examples are needed than training from scratch

In this project, pretraining is especially important because the dataset is specialized and not large enough to justify training a deep model from zero.

---

## Why YOLO Was Added
Initial saliency analysis suggested that the classifier sometimes relied on **background cues** such as sky, branches, and water instead of focusing only on the bird. To reduce this shortcut behavior, the pipeline was extended with **YOLO-based bird detection**.

YOLO is used here as a preprocessing step:
- detect the bird
- generate a bounding box
- crop around the bird with padding
- pass the crop to the classifier

This makes the classification task more aligned with the true goal: recognizing whether the **bird itself** is in flight or not.

---

## Dataset Structure
Example structure after preprocessing:

```text
avian_binary_dataset/
    in_flight/
    not_in_flight/
