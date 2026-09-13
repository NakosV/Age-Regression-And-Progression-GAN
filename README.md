<h1 align="center">Facial Age Regression & Progression GAN</h1>

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Architecture](https://img.shields.io/badge/Architecture-Conditional_GAN-2EA44F?style=for-the-badge)
![Generative AI](https://img.shields.io/badge/Domain-Generative_AI-9B59B6?style=for-the-badge)

</div>

A custom-built Conditional Generative Adversarial Network (cGAN) designed to perform realistic facial age regression and progression. Developed entirely from scratch in PyTorch as part of a university machine learning assignment.

---

## Visual Demonstration & Results

Unlike standard GANs that generate random, non-existent faces from static noise, this model extracts the identity of a specific person and transforms their age across five distinct demographic groups while preserving their core facial features.

<div align="center">
  <img width="900" alt="Results" src="https://github.com/user-attachments/assets/33938f38-892f-405e-91c8-c030c49540d5" />
</div>

> [!TIP]
> **How to read the grid:**  
> **Original** (Far Left) ➔ **Reconstruction** (Base Identity) ➔ **0-20** ➔ **21-35** ➔ **36-55** ➔ **56-65** ➔ **65+** (Far Right)

---

## Under The Hood: Architecture & Features

The model logic is encapsulated within a single executable pipeline: [`Conditional-GAN.py`](Conditional-GAN.py). To ensure high-quality transformations and stable training, the architecture utilizes several advanced techniques:

*   **Encoder-Decoder Generator:** An Encoder network strips the original image of its age characteristics to create a latent "identity" vector. The Generator then fuses this identity vector with a target age label to decode and produce the new face.
*   **Spectral Normalization:** Applied to the Discriminator to enforce Lipschitz continuity, preventing exploding gradients and stabilizing the adversarial training process.
*   **Weighted Random Sampling:** Dynamically balances the heavily skewed UTKFace dataset, ensuring the model sees an equal distribution of all age groups during training.

> [!NOTE]
> If you want to dive deeper into the mathematical implementation, the loss functions (Adversarial + L1 Reconstruction), and view the visual progression of the model across different epochs, read the **[Comprehensive Explanation Guide](EXPLANATION.md)**.

---

## Installation & Setup

To run this project locally, ensure you have Python installed. GPU acceleration is highly recommended for training and rapid inference.

**1. Install PyTorch with CUDA support**  
*(If you are running on an NVIDIA GPU, use the index provided below. Otherwise, install standard PyTorch).*
```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url [https://download.pytorch.org/whl/cu126](https://download.pytorch.org/whl/cu126)
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

Ensure your dataset paths and output directories are correctly configured at the top of the script. Run the main pipeline using:
```bash
python Conditional-GAN.py
```
The script will automatically check for existing checkpoints in the saving directory. You can choose to resume training from the latest epoch, start from scratch, or generate an inference grid.

---

## Useful Resources & Data

- **Training Dataset:** [UTKFace-New Dataset (Kaggle)](https://www.kaggle.com/datasets/jangedoo/utkface-new) - Contains over 20,000 face images with annotations for age, gender, and ethnicity.

---

## Contributions

While the core academic requirements for this project are complete, contributions are highly welcome. Feel free to suggest performance improvements, tune the hyperparameters, address active issues, or fork the repository to experiment with higher resolution images (e.g., 128x128).

---

> [!IMPORTANT]
> **Explore More of My Work**  
> This project is part of a broader collection of my academic machine learning implementations. To explore Computer Vision pipelines, Custom Large Language Models, and Deep Reinforcement Learning agents, check out my **[University Machine Learning Projects Catalog](https://github.com/NakosV/University-Machine-Learning-Projects-Catalog)**.
