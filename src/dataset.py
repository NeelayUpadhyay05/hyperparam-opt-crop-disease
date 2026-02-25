import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
from albumentations import Compose, Normalize, HorizontalFlip, Rotate, RandomBrightnessContrast, Resize
from albumentations.pytorch import ToTensorV2


def get_transforms(split, img_size=224):
    if split == 'train':
        return Compose([
            Resize(img_size, img_size),
            HorizontalFlip(p=0.5),
            Rotate(limit=15, p=0.5),
            RandomBrightnessContrast(p=0.3),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return Compose([
            Resize(img_size, img_size),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


class PlantDocDataset(Dataset):
    def __init__(self, samples, class_to_idx, split='train'):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        image = self.transform(image=image)['image']
        return image, label


def build_dataloaders(data_root,
                      batch_size=32,
                      val_split=0.2,
                      seed=42):

    train_root = os.path.join(data_root, 'train')
    test_root = os.path.join(data_root, 'test')

    classes = sorted(os.listdir(train_root))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Collect train samples
    train_samples = []
    for class_name in classes:
        class_dir = os.path.join(train_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                train_samples.append(
                    (os.path.join(class_dir, img_file),
                     class_to_idx[class_name])
                )

    print(f"[Train Total] {len(train_samples)} images")
    print(f"[Classes] {len(classes)} classes")

    # Split train -> train + val
    generator = torch.Generator().manual_seed(seed)

    val_size = int(val_split * len(train_samples))
    train_size = len(train_samples) - val_size

    train_subset, val_subset = random_split(
        train_samples,
        [train_size, val_size],
        generator=generator
    )

    train_dataset = PlantDocDataset(train_subset,
                                     class_to_idx,
                                     split='train')

    val_dataset = PlantDocDataset(val_subset,
                                   class_to_idx,
                                   split='val')

    # Test samples
    test_samples = []
    for class_name in classes:
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_samples.append(
                    (os.path.join(class_dir, img_file),
                     class_to_idx[class_name])
                )

    test_dataset = PlantDocDataset(test_samples,
                                    class_to_idx,
                                    split='test')

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, len(classes)


if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes = build_dataloaders("data")

    imgs, labels = next(iter(train_loader))
    print("Batch shape:", imgs.shape)
    print("Sample labels:", labels[:5])
    print("Num classes:", num_classes)
    print("Dataset ready!")