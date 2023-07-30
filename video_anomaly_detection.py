import os
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from generator.encoder_model import SwinTransformer3D
from dataloader.dataloader import AnomalyDataset
from trainer.train_utils import load_checkpoint
import numpy as np
from config import *
from inference import create_dir, delete_files, get_prediction_folder,  window_partition, MSE
from nori_2 import get_model, extract_frame_from_yolo
from tracking_utils.sort import *

def anomaly_patch(original, prediction, patch_size):
    orig = window_partition(original)
    pred = window_partition(prediction)
    # print(pred.shape)
    patch_list = []
    for i in range(orig.shape[0]):
        error = MSE(orig[i, :, :, :].numpy(), pred[i, :, :, :].numpy())
        patch_list.append(error)
    max_patches = list(np.argpartition(np.array(patch_list), -4)[-4:])   
    ###################################################################
    # h = orig.shape[1]    
    # for patch_idx in max_patches:
    #     ordered_patch = cv2.rectangle(orig[patch_idx, :, :, :].numpy()*255, (0, 0), (h-1, h-1), (255, 0, 0), 1)
    #     orig[patch_idx, :, :, :] = torch.Tensor(ordered_patch/255)

    out = orig.detach().clone().view(16, 16, 1, 14, 14, 3)
    output_image, hwhw_list = u_patch(out.numpy(), max_patches, (224, 224, 3))
    # cv2.imwrite('patches.jpg', output_image*255)
    ###################################################################
    return hwhw_list, output_image*255

def u_patch(patches, anomaly_patches, imsize):

    assert len(patches.shape) / 2 == len(
        imsize
    ), "The patches dimension is not equal to the original image size"

    assert len(patches.shape) == 6
    anomaly_patch_list = []
    i_h, i_w, i_c = imsize
    image = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, n_c, p_h, p_w, p_c = patches.shape

    s_w = 0 if n_w <= 1 else (i_w - p_w) / (n_w - 1)
    s_h = 0 if n_h <= 1 else (i_h - p_h) / (n_h - 1)
    s_c = 0 if n_c <= 1 else (i_c - p_c) / (n_c - 1)

    # The step size should be same for all patches, otherwise the patches are unable
    # to reconstruct into a image
    if int(s_w) != s_w:
        print("patch width not compatible with img size")
        return
    if int(s_h) != s_h:
        print("patch width not compatible with img size")
        return
    if int(s_c) != s_c:
        print("patch channels not compatible with img size")
        return

    s_w = int(s_w)
    s_h = int(s_h)
    s_c = int(s_c)

    i, j, k = 0, 0, 0
    count = 0
    while True:

        i_o, j_o, k_o = i * s_h, j * s_w, k * s_c
        image[i_o : i_o + p_h, j_o : j_o + p_w, k_o : k_o + p_c] = patches[i, j, k]
        if count in anomaly_patches:
            anomaly_patch_list.append([j_o, i_o, j_o + p_w, i_o+p_h])
        count += 1
        if k < n_c - 1:
            k = min((k_o + p_c) // s_c, n_c - 1)
        elif j < n_w - 1 and k >= n_c - 1:
            j = min((j_o + p_w) // s_w, n_w - 1)
            k = 0
        elif i < n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            i = min((i_o + p_h) // s_h, n_h - 1)
            j = 0
            k = 0
        elif i >= n_h - 1 and j >= n_w - 1 and k >= n_c - 1:
            # Finished
            break
        else:
            raise RuntimeError("Unreachable")

    return image, anomaly_patch_list




def save_inference(gen, test_loader, file, yolo_model, stride, device, save_folder, threshold=0.6,):
    sort_max_age = 5
    sort_min_hits = 1
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    
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
            anomaly_patches, x_yolo = anomaly_patch(original_frame, predicted_frame, patch_size=14)
        extract_frame_from_yolo(x_yolo, yolo_model, stride, device, anomaly_patches, file, idx, save_folder, sort_tracker)


def inference(test_folder, save_file, save_folder):
    delete_files(save_folder)
    create_dir(save_folder)
    gen = SwinTransformer3D().to(DEVICE)
    yolo_model, stride, device = get_model()
    test_dataframe = get_prediction_folder(TEST_DIR, test_folder, frame_jump=1)
    test_dataset = AnomalyDataset(test_dataframe, root_dir=TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LEARNING_RATE)
    save_inference(gen, test_loader, save_file, yolo_model, stride, device, save_folder, threshold=0.6)

def create_vid(img_folder, video_dir, test_folder_idx):
    x = os.listdir(img_folder)
    x.sort()
    image_files = [os.path.join(img_folder, i)  for i in x]
    image_files.sort()
    print(image_files)
    frameSize = (224, 224)
    out = cv2.VideoWriter(os.path.join(video_dir, f'output_{str(test_folder_idx).zfill(3)}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 12, frameSize)
    for filename in image_files:
            img = cv2.imread(filename)
            out.write(img)

    out.release()

if __name__ == '__main__':
    img_folder= 'infer'
    video_dir = "new_vids_2"
    test_folder = 91
    inference(test_folder=test_folder, save_file='yolo_img.jpg', save_folder=img_folder)
    create_vid(img_folder, video_dir, test_folder)