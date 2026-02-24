import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, Rotate, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2


def get_transforms(split):
    if split == 'train':
        return Compose([
            HorizontalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(p=0.3),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    return Compose([
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class PlantVillageDataset(Dataset):
    def __init__(self, data_root, split='train'):
        self.root = os.path.join(data_root, split)
        self.transform = get_transforms(split)
        self.samples = []

        classes = sorted(os.listdir(self.root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for class_name in classes:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, img_file), self.class_to_idx[class_name]))

        print(f'[{split}] {len(self.samples)} images, {len(classes)} classes')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        image = self.transform(image=image)['image']
        return image, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_ds = PlantVillageDataset('data', 'train')
    val_ds   = PlantVillageDataset('data', 'val')
    test_ds  = PlantVillageDataset('data', 'test')

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False)

    imgs, labels = next(iter(train_loader))
    print(f'Batch shape: {imgs.shape}')
    print(f'Labels: {labels[:5]}')
    print('Dataset ready!')