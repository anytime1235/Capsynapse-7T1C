"""ResNet-18 for CIFAR-10 (3x3 stem, no max-pool) and data loaders."""
import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def resnet18_cifar(num_classes: int = 10) -> nn.Module:
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


def conv_layers(m: nn.Module):
    """Ordered list of (name, module) for the 17 conv layers of ResNet18 (stem + 16 in blocks),
    excluding the 1x1 downsample shortcuts; fc is the 18th layer."""
    out = []
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d) and "downsample" not in name:
            out.append((name, mod))
    return out


def loaders(root="data/cifar", batch_size=128, workers=4, train_aug=True):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]) if train_aug else \
        transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tf_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr = datasets.CIFAR10(root, train=True, download=True, transform=tf_train)
    te = datasets.CIFAR10(root, train=False, download=True, transform=tf_test)
    return (DataLoader(tr, batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True),
            DataLoader(te, 256, shuffle=False, num_workers=workers, pin_memory=True))


@torch.no_grad()
def evaluate(m: nn.Module, loader, device="cuda") -> float:
    m.eval(); correct = n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (m(x).argmax(1) == y).sum().item(); n += y.numel()
    return 100.0 * correct / n
