# Age-Regression-And-Progression-GAN
A Facial Age Trasformation Conditional GAN featuring an Encoder-Decoder architecture to perform age Regression and Progression, built from scratch in Python 

# Contents of this Project  
This repository contains the code of the project which is the following file:
- **[`Conditional-GAN.py`](Conditional-GAN.py)**  

# How the program works
The code itself even though it is one file, it can be split in two parts. The first part being the training of the **ResNet-50 model** on the **CompCars dataset** in order for the model to be able to learn characteristics of specific car brands. The second part is the usage of the **YOLOv8n model** in conjuction with the now **trained ResNet-50 model** on a video with a few distinct and unique techniques. If you want to dive deeper and learn about the inner workings of the code in more detail **[you can click here.](EXPLANATION.md)**

# Useful Links
 - ***Dataset:*** https://www.kaggle.com/datasets/renancostaalencar/compcars
