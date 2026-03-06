import torch
import torch.nn as nn

class AnomalyAutoencoder(nn.Module):
    """
    A Deep Learning Convolutional Autoencoder architecture.
    In a real-world scenario, this is trained on healthy esophageal images.
    Anomalies (tumors/lesions) are detected by high reconstruction error.
    """
    def __init__(self):
        super(AnomalyAutoencoder, self).__init__()
        
        # Encoder: Extracts spatial patterns and compresses the image
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # Decoder: Attempts to reconstruct the image from compressed features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Output pixels in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
