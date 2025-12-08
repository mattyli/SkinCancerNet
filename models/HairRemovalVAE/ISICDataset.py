
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# IMG_EXTS = [".jpg", ".jpeg", ".png"]

# # helper to resolve actual file path (handles names with/without extension)
# def _find_image_path(img_name, images_dir):
#     # if image_name already contains an extension and exists, return it
#     base = os.path.join(images_dir, img_name)
#     if os.path.exists(base):
#         return base
#     # try with common extensions
#     name_root, ext = os.path.splitext(img_name)
#     if ext:
#         # had extension but file not found
#         return None
#     for e in IMG_EXTS:
#         p = os.path.join(images_dir, name_root + e)
#         if os.path.exists(p):
#             return p
#     return None

class ISICDataset(Dataset):
    def __init__(self, split, transform=None):
        self.split = split                          # one of [train, val, finetune] TYPE IS LIST
        self.transform = transform                  # resize, reshape etc

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        record = self.split[idx]
        img_path = record['image_paths']
        img_path = os.path.join("../..", img_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(int(record["target"]), dtype=torch.long)
        return img, label                           # label not used for the VAE