import torch
import matplotlib.pyplot as plt
from config import Config
from models.unet import UNet
from utils.data_utils import load_image, img_normalize, img_denormalize
from utils.loss_utils import cal_content_loss, cal_style_loss
from torchvision.transforms.functional import adjust_contrast


def main():
    device = Config.DEVICE
    content_image = load_image(Config.CONTENT_PATH, img_height=Config.IMG_HEIGHT, img_width=Config.IMG_WIDTH).to(device)
    style_net = UNet().to(device)
    optimizer = torch.optim.Adam(style_net.parameters(), lr=5e-4)

    for epoch in range(Config.MAX_STEP + 1):
        target_img = style_net(content_image, use_sigmoid=True).to(device)
        content_loss = cal_content_loss(content_image, target_img, style_net, img_normalize)
        style_loss = 0  # Placeholder if style image is not used

        total_loss = Config.CONTENT_WEIGHT * content_loss + Config.STYLE_WEIGHT * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item()}")
            output_image = torch.clamp(target_img.clone(), 0, 1)
            output_image = adjust_contrast(output_image, 1.5)
            plt.imshow(img_denormalize(output_image, device).cpu().squeeze(0).permute(1, 2, 0))
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main()