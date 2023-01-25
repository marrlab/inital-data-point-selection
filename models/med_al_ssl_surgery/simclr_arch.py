import torch.nn as nn
from models.med_al_ssl_surgery.resnet import resnet18
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimCLRArch(nn.Module):
    def __init__(self, num_channels, num_classes, drop_rate,
                 normalize, latent_dim=64, projection_dim=64,
                 arch='lenet', input_size=32):
        super(SimCLRArch, self).__init__()

        self.normalize = normalize

        if arch == 'lenet':
            self.encoder = nn.Sequential(
                nn.Conv2d(num_channels, 6, 3),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16 * 7 * 7, latent_dim),
                nn.ReLU(),
            )
        else:
            self.encoder = load_pretrained(resnet18(num_classes, input_size, drop_rate))
            latent_dim = self.encoder.linear[1].in_features
            self.encoder.linear = Identity()

        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(latent_dim, 84),
            nn.ReLU(),
            nn.Dropout(p=drop_rate, inplace=False),
            nn.Linear(84, num_classes)
        )

    # def forward(self, x):
    #     x = self.encoder(x)
    #     return x

    def forward(self, x):
        out, _ = self.encoder.forward_embeddings(x)
        return out


def load_pretrained(model):
    model_dict = model.state_dict()
    resnet18_pretrained_dict = models.resnet18(pretrained=True).state_dict()

    for key in list(model_dict.keys()):
        if 'linear' in key or 'conv1.weight' == key:
            continue
        model_dict[key] = resnet18_pretrained_dict[key.replace('shortcut', 'downsample')]

    model.load_state_dict(model_dict)

    return model
