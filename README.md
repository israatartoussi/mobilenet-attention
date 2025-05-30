# Lightweight CNN with Spatial Attention for Plant Disease Classification

This repository contains code and results for my doctoral research on deep learning-based image classification for plant diseases using small agricultural datasets. The goal is to develop lightweight models enhanced with spatial attention mechanisms to improve robustness against occlusion and limited data scenarios.

## 🌿 Project Objective

- Develop lightweight deep learning architectures (<3M parameters) for image-based plant disease classification.
- Address common challenges in agricultural imagery, especially **occlusion** caused by overlapping leaves and plants.
- Compare various **spatial attention mechanisms** integrated with MobileNetV2 to improve classification accuracy.

## 🧠 Methods

### Baseline
- **Architecture:** MobileNetV2 (without attention)
- **Dataset:** Small-scale plant disease dataset from [AgML](https://github.com/Project-AgML/AgML)
- **Training:** 100 epochs
- **Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix, t-SNE, Grad-CAM

### Attention Modules Implemented
| Attention Module | Description | Parameters Added | Notes |
|------------------|-------------|------------------|-------|
| CBAM             | Convolutional Block Attention Module (spatial + channel) | Moderate | Lightweight adaptation used |
| SAM              | Spatial Attention Module only | Very Low | Focuses on spatial saliency |
| BAM              | Bottleneck Attention Module | Low | Efficient for deeper layers |
| SE (Planned)     | Squeeze-and-Excitation | Low | Channel-focused |

Each model was trained and evaluated independently to allow fair comparison.

## 🧪 Results (Summary)

| Model                 | Accuracy | F1-score | Notes |
|-----------------------|----------|----------|-------|
| MobileNetV2 (baseline)| XX%      | XX%      | No attention |
| MobileNetV2 + CBAM    | XX%      | XX%      | Best performance in occluded regions |
| MobileNetV2 + SAM     | XX%      | XX%      | Competitive, low computation |
| MobileNetV2 + BAM     | XX%      | XX%      | Slight improvement over baseline |

> Full classification reports, confusion matrices, Grad-CAM, and t-SNE plots are available in the `results/` directory.

## 📁 Folder Structure
mobilenet_v2_cbam/
│
├── mobilenet_v2_base.ipynb # Baseline without attention
├── mobilenet_v2_cbam.ipynb # CBAM model
├── mobilenet_v2_sam.ipynb # SAM model
├── mobilenet_v2_bam.ipynb # BAM model
├── utils/
│ └── attention_modules.py # CBAM, SAM, BAM implementations
│
├── data_loader.py # AgML dataset loader
├── results/
│ ├── reports/ # Classification reports
│ ├── plots/ # Grad-CAM & t-SNE visualizations
│ └── logs/ # Training logs


## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score (using `sklearn`)
- Confusion Matrix (Seaborn)
- Grad-CAM heatmaps (saliency visualization)
- t-SNE feature mapping (representation space inspection)

## 🔧 Tools & Libraries

- PyTorch
- torchvision
- scikit-learn
- matplotlib, seaborn
- AgML (data loader)
- Grad-CAM library

## ✍️ Thesis Contribution (in progress)

This work will contribute to:
- A comprehensive **survey of spatial attention mechanisms** in agricultural deep learning
- Evaluation of **attention-integrated lightweight models** for occlusion handling
- A reproducible **pipeline** for training, evaluation, and visualization of such models

## 📌 Future Work

- Integrate and evaluate ** BAM, and SE attention**
- Add more lightweight architectures (e.g., EfficientNet-lite)
- Complete the literature **survey document**
- Extend experiments to **multi-class datasets**

## 📬 Contact

If you have any questions, feel free to contact me:  
📧 israa.tartoussi@gmail.comAZ



