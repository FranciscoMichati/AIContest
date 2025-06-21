# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Model Definition 
# 
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(torch.prod(torch.tensor((1, 28, 28))))),
            nn.Tanh()
            #nn.Sigmoid()
        )

    def forward(self, noise, labels):
        x = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(x)
        return img.view(img.size(0), *(1, 28, 28))


# Load Generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator_mnist3.pth", map_location=device))
generator.eval()

# Streamlit UI
st.title("MNIST Digit Generator (Conditional GAN)")
digit = st.selectbox("Select a digit (0-9):", list(range(10)))

if st.button("Generate"):
    with torch.no_grad():
        z = torch.randn(5, 100).to(device)
        labels = torch.full((5,), digit, dtype=torch.long).to(device)
        generated_imgs = generator(z, labels).cpu()

        grid = vutils.make_grid(generated_imgs, nrow=5, normalize=True, pad_value=1)
        plt.figure(figsize=(10, 2))
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis("off")
        st.pyplot(plt)


