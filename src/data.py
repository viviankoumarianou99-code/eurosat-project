from pathlib import Path
from typing import Tuple, List

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_transforms(image_size: int = 224, augment: bool = False):
    """
    Returns torchvision transforms for training/evaluation.
    If augment=True, applies light augmentations (safe for EuroSAT).
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates train/val/test DataLoaders from a folder-structured dataset (ImageFolder).
    Expected structure:
        data_dir/
          class1/
          class2/
          ...

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_path}\n"
            f"Check your path. Example: data/raw/EuroSAT_RGB"
        )

    # Base dataset for size & classes
    base_dataset = datasets.ImageFolder(str(data_path), transform=get_transforms(image_size, augment=False))
    class_names = base_dataset.classes

    total_size = len(base_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Reproducible split
    generator = __import__("torch").Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        base_dataset,
        lengths=[train_size, val_size, test_size],
        generator=generator
    )

    # IMPORTANT: Train set should use augmentations, val/test not.
    train_ds.dataset.transform = get_transforms(image_size, augment=True)
    val_ds.dataset.transform = get_transforms(image_size, augment=False)
    test_ds.dataset.transform = get_transforms(image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_names