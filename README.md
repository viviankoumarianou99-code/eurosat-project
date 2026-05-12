# EuroSAT RGB Image Classification Project

## 1. Problem Statement
This project addresses a multi-class image classification task in the field of remote sensing.  
Given a satellite image patch, the objective is to automatically classify the corresponding **land use / land cover (LULC)** category.

The task is formulated as a supervised image classification problem using convolutional neural networks (CNNs).



## 2. Dataset – EuroSAT RGB
The experiments are conducted on the **EuroSAT RGB dataset**, a widely used benchmark for satellite image classification.

The dataset consists of **13,500 RGB satellite images** derived from **Sentinel-2** observations.  
Each image corresponds to a **64×64 pixel** patch and belongs to one of **10 land use / land cover classes**, including:

- AnnualCrop  
- Forest  
- HerbaceousVegetation  
- Highway  
- Industrial  
- Pasture  
- PermanentCrop  
- Residential  
- River  
- SeaLake  

The dataset follows a **folder-per-class** structure, which makes it suitable for standard deep learning pipelines.



## 3. Data Preparation
Before training, the following preprocessing steps are applied:

- Images are resized to **224×224 pixels** to match the input requirements of ImageNet pre-trained models.
- Pixel values are normalized using ImageNet mean and standard deviation.
- The dataset is split into **training**, **validation**, and **test** subsets using a fixed random seed to ensure reproducibility.



## 4. Methodology
Two convolutional neural network architectures are evaluated using **transfer learning**:

- **ResNet50** – A deep residual network serving as a strong baseline.
- **EfficientNet-B0** – A more computationally efficient architecture with competitive performance.

Both models are initialized with **ImageNet pre-trained weights**.  
During training, a **head-only fine-tuning strategy** is adopted:
- The backbone (feature extractor) is frozen.
- Only the final classification layer is trained on the EuroSAT dataset.



## 5. Training Setup
The following experimental setup is used:

- Framework: PyTorch  
- Hardware: CPU  
- Loss function: Cross-Entropy Loss  
- Optimizer: Adam  
- Training mode: Head-only transfer learning  
- Number of epochs: 5  
- Batch size: 8  

This configuration allows stable training while remaining computationally feasible on a CPU-only environment.



## 6. Results

### 6.1 ResNet50
- Best validation accuracy: **0.9222**
- Test accuracy: **0.9175**

The model demonstrates strong and stable convergence.  
Loss curves, accuracy curves, and the confusion matrix on the test set are provided in the `outputs/resnet50/` directory.



### 6.2 EfficientNet-B0
- Best validation accuracy: **0.9069**
- Test accuracy: **0.9091**

EfficientNet-B0 achieves competitive performance while requiring fewer parameters and reduced computational complexity.  
Corresponding plots and the confusion matrix are available in the `outputs/efficientnet_b0/` directory.



## 7. Model Comparison

| Model | Best Validation Accuracy | Test Accuracy | Remarks |
|------|--------------------------|---------------|---------|
| ResNet50 | 0.9222 | 0.9175 | Strong baseline, stable and accurate |
| EfficientNet-B0 | 0.9069 | 0.9091 | More computationally efficient |



## 8. Discussion
Both models achieve high classification accuracy on the EuroSAT RGB dataset.  
ResNet50 consistently outperforms EfficientNet-B0 in this setup, achieving higher validation and test accuracy.

However, EfficientNet-B0 offers a favorable trade-off between performance and computational efficiency, making it attractive for resource-constrained environments.

Inspection of the confusion matrices indicates that most misclassifications occur between visually similar land cover categories, which is a common challenge in satellite image analysis.



## 9. Reproducibility and How to Run

### 9.1 Environment Setup
```bash
pip install -r requirements.txt

### 9.2 Training
Model training is performed using the provided training script.  
Both models are trained using a head-only transfer learning strategy.

```bash
python -m src.train --model resnet50 --train_mode head
python -m src.train --model efficientnet_b0 --train_mode head
``

### 9.3 Evaluation
Evaluation is carried out on a held-out test set using the evaluation script.
The process computes test accuracy and generates confusion matrices.

```bash
python -m src.eval --model resnet50
python -m src.eval --model efficientnet_b0
