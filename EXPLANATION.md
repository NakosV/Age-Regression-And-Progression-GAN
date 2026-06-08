# Detailed Explanation

## 1. Preprocessing
As listed before, the **UTKFace dataset** was used in this project, but due to its uneven age distribution, some balancing was required before the training. The dataset is heavily skewed towards younger individuals, with significantly more 20 year olds than 80 year olds. To prevent the model from becoming biased, I utilized a **WeightedRandomSampler**. This ensured that the network saw an equal distribution of all **5 age classes** (0-20, 21-35, 36-55, 56-65, 65+) during training, effectively solving the imbalance issue.

## 2. Training
As mentioned before, a custom **Conditional GAN (cGAN)** was built from scratch for the image generation part of this program. The training itself is an intense process where two models compete against each other for **100 epochs**. The **Generator** tries to create realistic faces, while the **Discriminator** tries to spot the fake ones. I trained the Generator with a learning rate of **0.0006** and the Discriminator with a learning rate of **0.00002**. The difference between may seem huge, but for me, when I tried to put them closer to each other, after a few epochs the Discriminator would be completely win over the Generator. Furthermore, to ensure the network doesn't generate a completely random face, an **L1 Reconstruction Loss** is applied. This forces the model to preserve the original identity and bone structure of the person, altering only the age-related features.

## 3. Inference
In this part of the program, the trained models are utilized to actually transform faces. The inference pipeline follows a very specific technique. First, the original image is passed through an **Encoder**, which compresses the face into a basic latent vector, stripping away its current age features. Then, this vector is combined with a specific **Target Age Label** and passed to the **Generator**. To make the Generator more powerful I split it into two sub-blocks: an initial projection block and a deeper **Decoder** block. The Decoder utilizes custom upsampling layers equipped with **double Convolutional layers** (`upsample_block`), allowing the network to meticulously reconstruct the face to match the targeted age group. This way, the program can take a single input image and output 5 different high-quality age variations of the exact same person.

## 4. Extras
Because of how complex and unstable GAN training can be, I implemented two technics. First, I used **Spectral Normalization** on the Discriminator to stabilize the training and prevent mode collapse. Secondly, because of how long the training takes, I used **checkpoints (saving every 5 epochs)** to ensure that if something went wrong, the progress always gets saved. It isn't completely necessary, but highly advised that you do the same when running this program.

# Progression
Here you can see how the program evolved over time.
- **Epoch 1**
  <img width="977" height="665" alt="Epoch 1" src="https://github.com/user-attachments/assets/952ceb89-a9d7-4ad8-86ae-6568febd8972" />

- **Epoch 20**
  <img width="981" height="664" alt="Epoch 20" src="https://github.com/user-attachments/assets/e93cad26-8299-479c-b225-e0d1ccf83bc8" />

- **Epoch 50**
  <img width="979" height="658" alt="Epoch 50" src="https://github.com/user-attachments/assets/166219f1-f061-4e41-b513-ad501232aa54" />

- **Epoch 80**
  <img width="979" height="659" alt="Epoch 80" src="https://github.com/user-attachments/assets/2b5e7ee1-6ced-49e0-9920-e920481d072b" />

- **Epoch 100**
  <img width="979" height="657" alt="Epoch 100" src="https://github.com/user-attachments/assets/7ee157b5-d89f-4ed1-bc70-4bc1532c4f7f" />

# Final Words
On my machine, the training of this GAN took a significant amount of time to run, mainly because generating high-quality images from scratch is a very heavy process. The run time and the success of the program are mainly based on the machine of the user and the hyperparameters fed to the program. I highly recommend everyone to play around with the variables and the datasets given to the program to see how the generated faces change.
