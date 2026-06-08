import os
import re
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn.utils.parametrizations import spectral_norm
from collections import Counter

batch_size = 32
epoch_number = 100
latent_dimension = 512
number_of_classes = 5
img_size = 64
learning_rate_G = 0.0006
learning_rate_D = 0.00002
lambda_recon = 30 
number_of_workers = 4

data_dir = "where/ypur/dataset/is"
saving_dir = "where/the/chechpoints/get/saved"
output_dir = "where/the/output/gets/saved"
os.makedirs(saving_dir, exist_ok = True)
os.makedirs(output_dir, exist_ok = True)
device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")

class UTKDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.dataset_path = data_dir
        self.transform = transform

        # How the program gets the age of the person
        pattern = re.compile(r"^(\d+)_\d+_\d+_\d+\.jpg\.chip\.jpg$", re.IGNORECASE)
        self.samples = []
        for i in os.listdir(data_dir):
            m = pattern.match(i)
            if m:
                age = int(m.group(1))
                if 0 <= age <= 116:
                    self.samples.append((i, self.get_age_class((age))))
                    
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        classes = ['0-20', '21-35', '36-55', '56-65', '65+']
        print(f"\nΥπάρχουν {len(self.samples)} εικόνες:")
        for i, name in enumerate(classes):
            print(f" [{i}] {name}: {counts[i]} εικόνες")
            
    def get_age_class(self, age):
        if age <= 20: return 0
        elif age <= 35: return 1
        elif age <= 55: return 2
        elif age <= 65: return 3
        else: return 4
        
    def __getitem__(self, index):
        fname, label = self.samples[index]
        image = Image.open(os.path.join(self.dataset_path, fname)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Some image augmentation to make the model more robust
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    
dataset = UTKDataset(data_dir = data_dir, transform = transform)

# I calculate the weights of ages in order to give the program even amounts of the faces
label_counts = Counter(label for _, label in dataset.samples)
class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
sample_weights = torch.tensor([class_weights[label] for _, label in dataset.samples], dtype = torch.float)
sampler = WeightedRandomSampler(weights = sample_weights, num_samples = len(sample_weights), replacement = True)
dataloader = DataLoader(dataset, batch_size = batch_size, sampler = sampler, num_workers = number_of_workers, pin_memory = True, drop_last = True)

# The Blocks
# Two Conv Layers to make the Generator stronger
def upsample_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.InstanceNorm2d(out_channels, affine = True),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.InstanceNorm2d(out_channels, affine = True),
        nn.ReLU(inplace = True),)

def downsample_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.InstanceNorm2d(out_channels, affine = True),
        nn.LeakyReLU(0.2, inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.InstanceNorm2d(out_channels, affine = True),
        nn.LeakyReLU(0.2, inplace = True),)

class Encoder(nn.Module):
    # It takes the identity from the original face without the age
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(64, affine = True),
            nn.LeakyReLU(0.2, inplace = True),
            downsample_block(in_channels = 64, out_channels = 128),
            downsample_block(in_channels = 128, out_channels = 256),
            downsample_block(in_channels = 256, out_channels = 512),
            downsample_block(in_channels = 512, out_channels = 512),
            nn.Conv2d(in_channels = 512, out_channels = latent_dimension, kernel_size = 2, stride = 1,padding = 0, bias = False),
            nn.Tanh())
    
    def forward(self, img):
        return self.model(img)
    
class Generator(nn.Module):
    # It fuses the identity with the target age to generate a new face
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(number_of_classes, embedding_dim = 128)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels = latent_dimension + 128, out_channels = 512, kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True))
        
        self.decoder = nn.Sequential(
            upsample_block(in_channels = 512, out_channels = 256),
            upsample_block(in_channels = 256, out_channels = 128),
            upsample_block(in_channels = 128, out_channels = 64),
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm2d(64, affine = True),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.Tanh())
    
    def forward(self, z, labels):
        c = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([z, c], dim = 1)
        x = self.gen(x)
        return self.decoder(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(number_of_classes, 1 * img_size * img_size)
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.LeakyReLU(0.2, inplace = True),
            spectral_norm(nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            spectral_norm(nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            spectral_norm(nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 4, stride = 2, padding = 1, bias = False)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace = True),
            spectral_norm(nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size = 4, stride = 1, padding = 0, bias = False)),)
        
    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, img_size, img_size)
        x = torch.cat([img, c], dim = 1)
        return self.model(x)
        
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname and not hasattr(m, 'weight_orig'):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname or 'InstanceNorm' in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    
encoder = Encoder().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

encoder.apply(weights_init)
generator.apply(weights_init)
discriminator.apply(weights_init)
    
total_E = sum(p.numel() for p in encoder.parameters())
total_G = sum(p.numel() for p in generator.parameters())    
total_D = sum(p.numel() for p in discriminator.parameters())    
    
print(f"\nEncoder Parameters: {total_E:,}")
print(f"Discriminator Parameters: {total_D:,}")    
print(f"Generator Parameters: {total_G:,}\n")    
    
criterion_loss = nn.MSELoss()
criterion_recon = nn.L1Loss()

optimizer_G = optim.Adam(list(encoder.parameters()) + list(generator.parameters()), lr = learning_rate_G, betas = (0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr = learning_rate_D, betas = (0.5, 0.999))
    
def checkpoints(epoch, d_loss, g_loss):
    path = os.path.join(saving_dir, f"ckpt_epoch_{epoch:03}.pt")
    torch.save({"epoch": epoch, "E_state": encoder.state_dict(), "G_state": generator.state_dict(), 
                "D_state": discriminator.state_dict(), "opt_G": optimizer_G.state_dict(), "opt_D": optimizer_D.state_dict(),
                "d_loss": d_loss, "g_loss":  g_loss, "latent_dimension": latent_dimension, "img_size": img_size,}, path)
    print(f"Checkpoint saved: {path}")
    all_checkpoints = sorted(glob.glob(os.path.join(saving_dir, "ckpt_epoch_*.pt")))
    for old in all_checkpoints[:-5]:
        os.remove(old)

def find_the_checkpoints():
    return sorted(glob.glob(os.path.join(saving_dir, "ckpt_epoch_*.pt")))
    
def load_checkpoint(path):
    checkpoint = torch.load(path, map_location = device)
    encoder.load_state_dict(checkpoint["E_state"])
    generator.load_state_dict(checkpoint["G_state"])
    discriminator.load_state_dict(checkpoint["D_state"])
    optimizer_G.load_state_dict(checkpoint["opt_G"])
    optimizer_D.load_state_dict(checkpoint["opt_D"])
    print(f"Το checkpoint: epoch {checkpoint['epoch']} φορτώθηκε")
    return checkpoint["epoch"]

# Display
classes = ['0-20', '21-35', '36-55', '56-65', '65+']

examples = {i: [] for i in range(5)}
for imgs, labels in dataloader:
    for img, label in zip(imgs, labels):
        lbl = label.item()
        if len(examples[lbl]) < 3:
            examples[lbl].append(img)
    if all(len(v) == 3 for v in examples.values()):
        break

def to_img(tensor):
    t = tensor.squeeze().cpu().clamp(-1, 1)
    t = (t * 0.5 + 0.5)
    t = (t * 255).byte()
    t = t.permute(1, 2, 0)
    return t

def show_results(epoch):
    encoder.eval()
    generator.eval()
    
    fig, axes = plt.subplots(5, 7, figsize = (20, 14))
    fig.suptitle(f"Results of Epoch {epoch}\n" "[Real | Reconstruction | 0-20 | 21-35 | 36-55 | 56-65 | 65+ ]", fontsize = 13)
    
    with torch.no_grad():
        for i in range(5):
            real_img = examples[i][0].unsqueeze(0).to(device)
            z = encoder(real_img)
            recon_img = generator(z, torch.tensor([i], device = device))
            transformed = [generator(z, torch.tensor([t], device = device)) for t in range(number_of_classes)]
            
            axes[i, 0].imshow(to_img(real_img))
            axes[i, 0].axis('off')
            axes[i, 0].set_ylabel(classes[i], fontsize = 11, fontweight = 'bold', rotation = 0, labelpad = 55, va = 'center')
            axes[i, 1].imshow(to_img(recon_img))
            axes[i, 1].axis('off')
            
            for t, t_img in enumerate(transformed):
                axes[i, t + 2].imshow(to_img(t_img))
                axes[i, t + 2].axis('off')
            if i == 0:
                axes[0, 0].set_title("Πραγματική", fontsize = 9, fontweight = 'bold')
                axes[0, 1].set_title("Recon", fontsize = 9, fontweight = 'bold')
                for t in range(5):
                    axes[i, t + 2].set_title(classes[t], fontsize = 9, fontweight = 'bold')
                    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(save_path, dpi = 100, bbox_inches = 'tight')
    plt.close(fig)
    
    encoder.train()
    generator.train()
    
starting_epoch = 0
existing_checkpoints = find_the_checkpoints()

if existing_checkpoints:
    latest_checkpoint = existing_checkpoints[-1]
    print(f"\n Checkpoint: {os.path.basename(latest_checkpoint)}")
    answer = input("Do you want to continue from that checkpoint? (yes/no)").strip().lower()
    if answer == 'yes':
        starting_epoch = load_checkpoint(latest_checkpoint)
    else:
        print("Training starting from scratch")
else:
    print("No checkpoints, starting from scratch")
    
# Training
print(f"The training starts from epoch {starting_epoch + 1}\n")
for epoch in range(starting_epoch, epoch_number):
    epoch_d, epoch_g, epoch_recon = 0.0, 0.0, 0.0
    for i, (real_imgs, labels) in enumerate(dataloader):
        Real = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        labels = labels.to(device)
        target_labels = torch.randint(0, number_of_classes, (Real,), device = device)
        valid = torch.ones(Real, 1, 1, 1, device = device)
        fake = torch.zeros(Real, 1, 1, 1, device = device)
        noise_factor = max(0.0, 0.1 * (1.0 - epoch / epoch_number))
        noise_real = real_imgs + noise_factor * torch.randn_like(real_imgs)
        
        optimizer_D.zero_grad()
        real_loss = criterion_loss(discriminator(noise_real, labels), valid)
        z = encoder(real_imgs)
        generated_imgs = generator(z, target_labels)
        fake_loss = criterion_loss(discriminator(generated_imgs.detach(), target_labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        z = encoder(real_imgs)
        generated_imgs = generator(z, target_labels)
        adv_loss = criterion_loss(discriminator(generated_imgs, target_labels), valid)
        recon_imgs = generator(z, labels)
        recon_loss = criterion_recon(recon_imgs, real_imgs)
        g_loss = adv_loss + lambda_recon * recon_loss
        g_loss.backward()
        optimizer_G.step()
        
        epoch_d = epoch_d + d_loss.item()
        epoch_g = epoch_g + g_loss.item()
        epoch_recon = epoch_recon + recon_loss.item()
        
    n = len(dataloader)
    avg_d = epoch_d / n
    avg_g = epoch_g / n
    avg_recon = epoch_recon / n
        
    print(f"[Epoch {epoch + 1: 2d} / {epoch_number}] "
          f"D: {avg_d:.4f} | G: {avg_g:.4f} | Recon: {avg_recon:.4f}")
    if (epoch + 1) % 5 == 0 or epoch == 0:
        checkpoints(epoch + 1, avg_d, avg_g)
        show_results(epoch +1)

# Results
def inference(num_samples = 5):
    encoder.eval()
    generator.eval()
    dataiter = iter(dataloader)
    imgs, labels = next(dataiter)
    imgs = imgs[:num_samples].to(device)
    labels = labels[:num_samples].to(device)
    fig, axes = plt.subplots(num_samples, number_of_classes + 2, figsize = (3 * (number_of_classes + 2), 3 * num_samples))
    fig.suptitle("Ages", fontsize = 16, fontweight = 'bold')
    
    with torch.no_grad():
        for i in range(num_samples):
            img = imgs[i].unsqueeze(0)
            original_lbl = labels[i].item()
            z = encoder(img)
            recon = generator(z, torch.tensor([original_lbl], device = device))
            axes[i, 0].imshow(to_img(img))
            axes[i, 0].set_title(f"Original\n({classes[original_lbl]})", fontweight = 'bold')
            axes[i, 0].axis('off')
            axes[i, 1].imshow(to_img(recon))
            axes[i, 1].set_title("Reconstruction", fontweight = 'bold')
            axes[i, 1].axis('off')
            
            for t in range(number_of_classes):
                transformed_img = generator(z, torch.tensor([t], device = device))
                axes[i, t+2].imshow(to_img(transformed_img))
                title_color = 'green' if t != original_lbl else 'black'
                axes[i, t+2].set_title(f"Σε {classes[t]}", fontweight = 'bold', color = title_color)
                axes[i, t+2].axis('off')

    plt.tight_layout()
    final_path = os.path.join(output_dir, "inference.png")
    plt.savefig(final_path, dpi = 150, bbox_inches = 'tight')
    plt.close(fig)

inference()
