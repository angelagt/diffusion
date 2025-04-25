import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    Dataset for images with optional multi-label conditioning.

    If class_cond=True, expects filenames to encode labels as
    'label1_label2_..._id.ext' (with optional numeric suffix). Splits
    by '_' and matches tokens against provided class_list.
    """
    def __init__(
        self,
        folder: str,
        image_size: int,
        class_cond: bool = False,
        class_list: list[str] | None = None,
        transforms=None,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.transforms = transforms
        self.class_cond = class_cond

        if class_cond:
            assert class_list is not None, "Need class_list for multi-label"
            # Mapping from class name to index
            self.class_to_idx = {c: i for i, c in enumerate(class_list)}
            self.num_classes = len(class_list)
            # Scan folder for image paths and their label-lists
            self.filenames, self.class_label_lists = self._scan_folder(folder)
        else:
            # Just scan all images without labels
            self.filenames = self._scan_folder(folder)

    def _scan_folder(self, folder: str):
        """
        Walks `folder` for image files.
        If class_cond, returns (filenames, class_label_lists).
        Otherwise, returns just filenames list.
        """
        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        filenames = []
        labels_list = []

        for root, dirs, files in os.walk(folder):
            for fname in files:
                name, ext = os.path.splitext(fname)
                if ext.lower() not in exts:
                    continue

                path = os.path.join(root, fname)
                filenames.append(path)

                if self.class_cond:
                    parts = name.split('_')
                    # Drop trailing numeric ID if present
                    if parts and parts[-1].isdigit():
                        parts = parts[:-1]
                    # Keep only known classes
                    label_names = [p for p in parts if p in self.class_to_idx]
                    labels_list.append(label_names)

        if self.class_cond:
            return filenames, labels_list
        return filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int):
        # Load and preprocess image
        path = self.filenames[idx]
        img = Image.open(path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        else:
            img = np.array(img.resize((self.image_size, self.image_size)))
        out = {"image": np.array(img)}            
        
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))

        arr = np.array(img, dtype=np.float32)           # shape (H,W,3), values 0–255
        tensor = torch.from_numpy(arr)                  # uint8→float32
        tensor = tensor.permute(2, 0, 1)                # → (3,H,W)
        tensor = tensor.mul_(1/127.5).sub_(1.0)         # scale 0–255 → –1–1
        out = {"image": tensor}

        if self.class_cond:
            labels = self.class_label_lists[idx]
            multi_hot = np.zeros(self.num_classes, dtype=np.float32)
            for lab in labels:
                multi_hot[self.class_to_idx[lab]] = 1.0
            out["y"] = multi_hot

        return out

# Example load_data modification

def load_data(
    *,
    data_dir: str,
    batch_size: int,
    image_size: int,
    class_cond: bool = False,
    class_list: list[str] | None = None,
    deterministic: bool = False
):
    """
    Returns a DataLoader for ImageDataset with optional multi-label conditioning.
    """
    from torch.utils.data import DataLoader
    # Define any torchvision or custom transforms here
    transforms = None  # e.g., your resize/normalize pipeline

    dataset = ImageDataset(
        folder=data_dir,
        image_size=image_size,
        class_cond=class_cond,
        class_list=class_list,
        transforms=transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        drop_last=True,
        num_workers=4,
    )
    return loader
