import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
import json

RAW_ROOT = 'data/raw/plantvillage_dataset/color'  
if not os.path.exists(RAW_ROOT):
    print(f"❌ {RAW_ROOT} not found. Check: ls data/raw/")
    print("Expected: data/raw/color/, data/raw/grayscale/, data/raw/segmented/")
    exit(1)

SPLITS = ['train', 'val', 'test']
SIZES = [0.7, 0.15, 0.15]

# Create splits
for split in SPLITS:
    os.makedirs(f'data/{split}', exist_ok=True)

class_stats = Counter()
for class_name in os.listdir(RAW_ROOT):
    class_path = os.path.join(RAW_ROOT, class_name)
    if not os.path.isdir(class_path): 
        continue
    
    imgs = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.jpeg'))]
    n_imgs = len(imgs)
    class_stats[class_name] = n_imgs
    
    if n_imgs < 20:
        print(f"⚠️ Skipping tiny {class_name}: {n_imgs} imgs")
        continue
    
    print(f"Splitting {class_name}: {n_imgs} imgs")
    
    # Split
    train_imgs, temp = train_test_split(imgs, train_size=SIZES[0], random_state=42)
    val_imgs, test_imgs = train_test_split(temp, test_size=SIZES[1]/(SIZES[1]+SIZES[2]), random_state=42)
    
    splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}
    for split_name, split_imgs in splits.items():
        dest = f'data/{split_name}/{class_name}'
        os.makedirs(dest, exist_ok=True)
        for img in split_imgs:
            shutil.copy(os.path.join(class_path, img), dest)

# Stats
stats = {'classes': dict(class_stats), 'total': sum(class_stats.values())}
with open('data_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ SPLIT COMPLETE! {len(class_stats)} classes, {stats['total']} images")
print("data/train/ now has 38 class folders!")