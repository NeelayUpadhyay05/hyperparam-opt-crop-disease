import torch
import torch.nn as nn
from torchvision import models


def get_efficientnet(num_classes=38, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)

    # Replace final classifier for 38 crop disease classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == '__main__':
    model = get_efficientnet(num_classes=38)

    # Test with dummy input
    dummy = torch.randn(4, 3, 256, 256)
    output = model(dummy)

    print(f'Model: EfficientNet-B0')
    print(f'Input:  {dummy.shape}')
    print(f'Output: {output.shape}')
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print('Model ready!')