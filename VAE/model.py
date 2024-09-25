import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Define the encoder
class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        # Convolutional layers for encoding (w - k + 2p)/s + 1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)  # (32, 109, 89)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # (64, 55, 45)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # (128, 28, 23)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(4, 3), stride=2, padding=1)  # (256, 14, 12)

        # Fully connected layers to get the mean and variance (for reparameterization trick)
        self.fc_mu = nn.Linear(256 * 14 * 12, latent_dim)
        self.fc_logvar = nn.Linear(256 * 14 * 12, latent_dim)

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten and compute mu and logvar
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, output_channels=3):
        super(Decoder, self).__init__()
        # Fully connected layer to reshape the latent space to match the size before convolution
        self.fc = nn.Linear(latent_dim, 256 * 14 * 12)

        # Transposed convolutional layers for decoding  (w-1)s + k - 2p
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 3), stride=2, padding=1)  # (128, 28, 23)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)  # (64, 55, 45)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)  # (32, 109, 89)
        self.deconv4 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)  # (32, 218, 178)
        self.deconv5 = nn.ConvTranspose2d(35, output_channels, kernel_size=3, stride=1, padding=1)  # (3, 218, 178)


    def forward(self, z, original):  # original -> (batch_size, 3, 218, 178)
        # Fully connected layer to reshape z
        z = self.fc(z)
        z = z.view(z.size(0), 256, 14, 12)  # Reshape to the size before encoding

        # Apply transposed convolutional layers

        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv4(z))                   # (batch_size, 32, 218, 178)
        z = torch.concat([z, original], dim = 1)      # (batch_size, 35, 218, 178)
        z = torch.sigmoid(self.deconv5(z))  # Use sigmoid for output to be between [0, 1]

        return z

# Define the full VAE
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode the input image to get mu and logvar
        mu, logvar = self.encoder(x)

        # Reparameterization trick
        z = self.reparameterize(mu, logvar)

        # Decode the latent variable
        x_reconstructed = self.decoder(z, x)

        return x_reconstructed, mu, logvar