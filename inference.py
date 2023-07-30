import os
import cv2
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from generator.encoder_model import SwinTransformer3D
import fnmatch
from patchify import patchify, unpatchify
from torchvision.utils import save_image
from dataloader.dataloader import AnomalyDataset
from trainer.train_utils import load_checkpoint
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import log10, sqrt
from einops import rearrange
from config import *
import shutil


def delete_files(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def create_dir(folder):
    if not os.path.exists(folder):
        print("creating folder ==============>")
        os.makedirs(folder)

def get_prediction_folder(root_dir, test_folder_no, mode='Test', number_of_frames=4, frame_jump=1):
    train_dataset = pd.DataFrame({'frames': [], 'label': []})
    lst = [
        f"{mode}{str(test_folder_no).zfill(3)}/{str(i+1).zfill(3)}.jpg" for i in range(0, len(fnmatch.filter(os.listdir(os.path.join(root_dir, f"{mode}{str(test_folder_no).zfill(3)}")), '*.jpg')), frame_jump)]
    items = [(lst[i:i+number_of_frames], lst[i+number_of_frames]) for i in range(len(lst)-(number_of_frames+1))]
    x = pd.DataFrame(items, columns=["frames", "label"])
    train_dataset = train_dataset.append(x, ignore_index=True)
    return train_dataset

def save_inference(gen, test_loader, folder, threshold=0.6):
    x_plot = []
    y_plot = []
    psnr_list = []
    for idx, (x, y, _) in enumerate(tqdm(test_loader)):
        x = torch.concat((x, torch.randn(1, 3, 4, 224, 224)), 0)
        y = torch.concat((y, torch.randn(1, 3, 224, 224)), 0)
        # x = rearrange(x, 'b c t h w -> b t c h w')
        x, y = x.to(DEVICE), y.to(DEVICE)
        x, y = x.to(torch.float), y.to(torch.float)
        gen.eval()
        with torch.no_grad():
            y_fake, _ = gen(x)
            predicted_frame = torch.unsqueeze(y_fake[0], dim=0)
            original_frame = torch.unsqueeze(y[0], dim=0)
            fig, ax = plt.subplots( nrows=1, ncols=1)  # create figure & 1 
            plt.xlim(0, len(test_loader))
            # plt.ylim(0, 100)
            x_plot.append(idx+1)
            p = PSNR_sliding_window(original_frame.numpy(), predicted_frame.numpy())
            y_plot.append(p)
            psnr_list.append(p)
            anomaly = anomaly_patch(original_frame, predicted_frame)
            ax.plot(x_plot, y_plot)
            fig.savefig('img.jpg')   # save the figure to file
            plt.close(fig)
            img = Image.open("./img.jpg")
            img = img.resize((224, 224))
            img = np.asarray(img)
            img = torch.tensor(img).unsqueeze(dim=0).to(DEVICE)/255
            img = rearrange(img, 'b h w c -> b c h w')
            save_image(original_frame, "original.jpg")
            save_image(predicted_frame, "pred.jpg")
            save_image(torch.concat((original_frame, predicted_frame, anomaly, img), 0), folder + f"/frame_{str(idx).zfill(3)}.png")
            # save_image(y, folder + f"/label_{epoch}.png")
    plt.clf()
    x = [i for i in range(len(psnr_list))]
    y = regularity_score(psnr_list)
    # threshold = 0.5
    detection = [0 if i > threshold else 1 for i in y]
    # print(detection)
    fig, ax = plt.subplots( nrows=1, ncols=1)  # create figure & 1 
    ax.plot(x, y)
    fig.savefig('regularity.jpg')   # save the figure to file
    plt.close(fig)
    plt.clf()
    fig, ax = plt.subplots( nrows=1, ncols=1)  # create figure & 1 
    ax.plot(x, detection)
    fig.savefig('classification.jpg')   # save the figure to file
    plt.close(fig)


def regularity_score(psnr_list):
    min_psnr_patch = min(psnr_list)
    psnr_diff = max(psnr_list) - min_psnr_patch
    regularity_list = []
    for psnr in psnr_list:
        s = (psnr - min_psnr_patch)/psnr_diff
        regularity_list.append(s)
    return regularity_list

def inference(test_folder, save_folder):
    gen = SwinTransformer3D().to(DEVICE)
    test_dataframe = get_prediction_folder(TEST_DIR, test_folder, frame_jump=2)
    test_dataset = AnomalyDataset(test_dataframe, root_dir=TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
    save_inference(gen, test_loader, folder=save_folder.split("\\")[-1], threshold=0.6)

def PSNR_sliding_window(original, predicted):
    partition_orig = window_partition(torch.tensor(original, requires_grad=False))
    partition_orig = rearrange(partition_orig, 'p h w c -> p (h w c)')
    partition_pred = window_partition(torch.tensor(predicted, requires_grad=False))
    partition_pred = rearrange(partition_pred, 'p h w c -> p (h w c)')
    mse = np.mean((partition_orig.numpy() - partition_pred.numpy()) ** 2)
    max_pixel = predicted.max()**2
    psnr = 10*log10(max_pixel / sqrt(mse))
    return psnr

def unpatch(output_patches):
    out = torch.Tensor(output_patches).view(16, 16, 1, 14, 14, 3)
    output_image = unpatchify(out.numpy(), (224, 224, 3))
    return output_image
    # cv2.imwrite("k.png", output_image*255)

def MSE(img1, img2):
        squared_diff = (img1 -img2) ** 2
        summed = np.sum(squared_diff)
        num_pix = img1.shape[0] * img1.shape[1] #img1 and 2 should have same shape
        err = summed / num_pix
        return err

def anomaly_patch(original, prediction):
    orig = window_partition(original)
    pred = window_partition(prediction)
    # print(pred.shape)
    patch_list = []
    for i in range(orig.shape[0]):
        error = MSE(orig[i, :, :, :].numpy(), pred[i, :, :, :].numpy())
        patch_list.append(error)
    max_patches = list(np.argpartition(np.array(patch_list), -4)[-4:])
    patch_list = list(np.diff(np.absolute(np.diff(patch_list))))
    # min_patch = patch_list.index(max(patch_list))
    h = orig.shape[1]
    for patch_idx in max_patches:
        ordered_patch = cv2.rectangle(orig[patch_idx, :, :, :].numpy()*255, (0, 0), (h-1, h-1), (255, 0, 0), 1)
        orig[patch_idx, :, :, :] = torch.Tensor(ordered_patch/255)
    orig = orig.numpy()
    # pred[min_patch, :, :, :] = image_bordered/255
    anomaly = unpatch(orig)
    # image_bordered = cv2.resize(pred[min_patch, :, :, :]*255, (224, 224))
    # cv2.imwrite("img_patch.png", pred[5, :, :, :]*255)
    anomaly = torch.Tensor(anomaly).unsqueeze(dim=0)
    anomaly = rearrange(anomaly, 'b h w c -> b c h w')
    return anomaly

def PSNR(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    max_pixel = predicted.max()**2
    psnr = 10*log10(max_pixel / sqrt(mse))
    return psnr

def get_prediction_graph_video(folder, k=None):
    if k:
        initial = k
        final = k+1
    else:
        initial = 1
        final = 13
    for i in range (initial, final):
        delete_files(folder)
        create_dir(folder)

        inference(i, folder)

        x = os.listdir(folder)
        x.sort()
        image_files = [os.path.join(folder, i)  for i in x]
        image_files.sort()
        frameSize = (906, 228)
        out = cv2.VideoWriter(os.path.join('./new_vids_2', f'output_train_batch_{str(i).zfill(3)}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, frameSize)

        for filename in image_files:
            img = cv2.imread(filename)
            out.write(img)

        out.release()

def window_partition(x, window_size=14):
    x = rearrange(x, 'b c h w -> b h w c')
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

if __name__ == "__main__":
    folder = "D:\AnomalyDetection\\infer"
    get_prediction_graph_video(folder, 91)
    # folder = "patches"
    # image = Image.open("img.jpg")  # for example (3456, 5184, 3)
    # image = image.resize((224, 224))
    # image = np.asarray(image)
    # image = torch.tensor(image).unsqueeze(dim=0)/255
    # x = window_partition(image, 28)
    # save_image(x, "par.jpg")