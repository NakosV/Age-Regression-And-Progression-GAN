# Age Regression And Progression GAN
A Facial Age Trasformation Conditional-GAN featuring an Encoder-Decoder architecture to perform age Regression and Progression, built from scratch in Python as part of a university course assignment.

# Contents of this Project  
This repository contains the code of the project which is the following file:
- **[`Conditional-GAN.py`](Conditional-GAN.py)**  

# Some words about the program
Unlike other GANs, this model doesn't just generate random faces. It has been specifically built to take the face of someone and show it in different ages. In order to insure that the program worked to some extend, I had to utilize an Encoder-Decoder architecture along with Spectral Normalization and custom data sampling. If you want to take a look under the hood and learn about the inner workings of the code in more detail as well as the progression of the program at different epochs **[you can click here.](DETAILS-ABOUT-THE-CODE.md)**

# Results
Here you can see the results of the program. At the far right are the **original pictures**, next to them are the **reconstructions* and after that are the **reconstructions at the age groups 0-20, 21-35, 36-55, 56-65 and lastly 65+.**  
<img width="3060" height="2212" alt="Results" src="https://github.com/user-attachments/assets/33938f38-892f-405e-91c8-c030c49540d5" />

# Useful Links
 - ***Dataset:*** https://www.kaggle.com/datasets/jangedoo/utkface-new

# Contributions
Even though I am largely finished with this project feel free to suggest improvements, address issues or fork the repository to experiment with your own datasets.

# Other Projects
I have built many more projects that revolve around machine learning. If you are interested to see them **[you can click here to check them out](https://github.com/NakosV/University-Machine-Learning-Projects-Catalog)**
