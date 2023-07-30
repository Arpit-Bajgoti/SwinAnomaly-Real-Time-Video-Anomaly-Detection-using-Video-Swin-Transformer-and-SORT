from config import *
from einops import rearrange
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def save_some_examples(gen, val_loader, epoch, folder):
    x, y, _ = next(iter(val_loader))
    # x = rearrange(x, 'b c t h w -> b t c h w')
    x, y = x.to(DEVICE), y.to(DEVICE)
    x, y = x.to(torch.float), y.to(torch.float)
    gen.eval()
    with torch.no_grad():
        y_fake, _ = gen(x)
        x = x.cpu().numpy()
        x = rearrange(x, 'b c t h w -> b t c h w')
        # y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        plt.rcParams["figure.figsize"] = [12, 12]
        plt.rcParams["figure.autolayout"] = True
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.axis('off')
            plt.title(f"frame_{i+1}")
            k = x[0, i, :, :]
            k = rearrange(k, 'c h w -> h w c')
            # plt.imshow(k, cmap="gray")
            plt.savefig('example', dpi=200)
        if epoch == 1:
            save_image(y, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr