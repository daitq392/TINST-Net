import torch
from PIL import Image
from torchvision import transforms


def load_image(img_path, img_height=None, img_width=None):
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def img_denormalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
    image = image * std + mean
    return image


def img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1)
    image = (image - mean) / std
    return image
