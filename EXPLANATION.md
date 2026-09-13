<h1 align="center">Comprehensive Architecture & Logic Guide</h1>

This document provides an in-depth breakdown of the facial age transformation pipeline, detailing dataset balancing protocols, adversarial training mechanics, inference architecture, and visual evolution across epochs.

---

## 1. Preprocessing & Dataset Balancing

The **UTKFace dataset** serves as the foundation for this generative model. Because real-world demographic data is heavily skewed toward younger individuals, standard training would introduce severe generational bias.

> [!TIP]
> **Class Balancing via Sampling**  
> To counteract skewness, a **WeightedRandomSampler** was implemented. This ensures the network encounters an equal distribution across all 5 discrete age classes (0-20, 21-35, 36-55, 56-65, and 65+) during optimization.

> [!WARNING]
> **Hardware Constraints**  
> Due to the substantial VRAM and computational constraints of training a GAN from scratch locally, images were scaled down to **64x64 pixels**.

---

## 2. Adversarial Training Dynamics

The core system relies on a custom **Conditional GAN (cGAN)** trained over 100 epochs in a zero-sum adversarial loop.

<div align="center">

![Generator LR](https://img.shields.io/badge/Generator_LR-0.0006-2EA44F?style=for-the-badge)
![Discriminator LR](https://img.shields.io/badge/Discriminator_LR-0.00002-E74C3C?style=for-the-badge)
![Loss](https://img.shields.io/badge/Reconstruction_Loss-L1_(λ=30)-F39C12?style=for-the-badge)

</div>

*   **Differential Learning Rates:** The asymmetry in learning rates prevents the Discriminator from prematurely overpowering the Generator, a common failure mode in cGAN training.
*   **Identity Preservation (L1 Loss):** To prevent the model from generating arbitrary faces, the L1 Reconstruction Loss is enforced alongside adversarial feedback. This anchors the core bone structure and facial identity of the subject, restricting alterations primarily to age-related features.

---

## 3. Inference & Transformation Pipeline

Transforming a face relies on a decoupled Encoder-Decoder architecture:

1.  **Latent Space Compression:** The input image passes through an **Encoder** to compress facial features into a latent vector while stripping away age markers.
2.  **Conditional Generation:** The latent vector fuses with a **Target Age Label** via learned embeddings and passes into the Generator.
3.  **Decoder Architecture:** The Generator features an initial projection sequence followed by a deeper **Decoder** block utilizing custom double-convolution upsampling layers (`upsample_block`). This synthesizes 5 distinct age variations of the exact same subject from a single source image.

---

## 4. Stability Measures & Fault Tolerance

To mitigate the instability inherent to GAN architectures, two main strategies were integrated:

> [!IMPORTANT]
> **Spectral Normalization:** Applied to the Discriminator layers to enforce Lipschitz continuity, effectively suppressing exploding gradients and preventing mode collapse.

> [!NOTE]
> **Automated Checkpointing:** Model weights, optimizer states, and epoch metrics are saved every **5 epochs**, allowing training to resume seamlessly from interruptions.

---

## Epoch-by-Epoch Visual Progression

The following captures how the model's structural coherence, texture mapping, and age-transition accuracy evolved throughout the 100-epoch training cycle:

<br>

### ![Epoch 1](https://img.shields.io/badge/Phase-Epoch_1-95A5A6?style=for-the-badge)
<img width="977" height="665" alt="Epoch 1" src="https://github.com/user-attachments/assets/952ceb89-a9d7-4ad8-86ae-6568febd8972" />

### ![Epoch 20](https://img.shields.io/badge/Phase-Epoch_20-3498DB?style=for-the-badge)
<img width="981" height="664" alt="Epoch 20" src="https://github.com/user-attachments/assets/e93cad26-8299-479c-b225-e0d1ccf83bc8" />

### ![Epoch 50](https://img.shields.io/badge/Phase-Epoch_50-9B59B6?style=for-the-badge)
<img width="979" height="658" alt="Epoch 50" src="https://github.com/user-attachments/assets/166219f1-f061-4e41-b513-ad501232aa54" />

### ![Epoch 80](https://img.shields.io/badge/Phase-Epoch_80-E67E22?style=for-the-badge)
<img width="979" height="659" alt="Epoch 80" src="https://github.com/user-attachments/assets/2b5e7ee1-6ced-49e0-9920-e920481d072b" />

### ![Epoch 100](https://img.shields.io/badge/Phase-Epoch_100-2ECC71?style=for-the-badge)
<img width="979" height="657" alt="Epoch 100" src="https://github.com/user-attachments/assets/7ee157b5-d89f-4ed1-bc70-4bc1532c4f7f" />
