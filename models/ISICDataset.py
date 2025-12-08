
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

IMG_EXTS = [".jpg", ".jpeg", ".png"]

# helper to resolve actual file path (handles names with/without extension)
def _find_image_path(img_name, images_dir):
    # if image_name already contains an extension and exists, return it
    base = os.path.join(images_dir, img_name)
    if os.path.exists(base):
        return base
    # try with common extensions
    name_root, ext = os.path.splitext(img_name)
    if ext:
        # had extension but file not found
        return None
    for e in IMG_EXTS:
        p = os.path.join(images_dir, name_root + e)
        if os.path.exists(p):
            return p
    return None

class ISICDataset(Dataset):
    def __init__(self, df_subset, transform=None):
        self.df = df_subset.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(row["target"]), dtype=torch.long)
        return img, label