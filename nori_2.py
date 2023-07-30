import os
import cv2

from random import randint
from tracking_utils.sort import *
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import TracedModel

# bbox_xyxy = tracked_dets[:,:4]
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / boxAArea
	# return the intersection over union value
	return iou

def patch_detect(tracks, patches, threshold=0.7):

    iou_scores = []  # calculate iou score for every patch 
    indexs_max_tracker = []  # for clearing the update count for all the tracks except items in the list
                             # and for avoiding count update for multiple patches
    overlapping_max_patch_idx = []
    for patch in patches:
        iou_scores = []
        for track in tracks:
            iou_scores.append(bb_intersection_over_union(patch, track.bbox_history[-1][:4]))
        if(max(iou_scores) > threshold):
            overlapping_max_patch_idx.append(np.argpartition(np.array(iou_scores), -1)[-1]) # get max overlap bbox for each patch
    
    overlapping_max_patch_idx = [*set(overlapping_max_patch_idx)]
    
    for idx in overlapping_max_patch_idx:
        tracks[idx].update_patch_overlap_count()

    for idx, track in enumerate(tracks):
        if idx not in overlapping_max_patch_idx:
            track.patch_relaxation_count += 1
            if track.patch_relaxation_count > 5:
                track.patch_overlap_count = 0

    return indexs_max_tracker, iou_scores
 
def get_model(weights="yolov7.pt", imgsz=256):
    device = "cpu"
    set_logging()
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model = TracedModel(model, device, imgsz)
    return model, stride, device


def get_prediction(
    img0,
    model,
    stride,
    image_size=256,
    conf_threshold=0.25,
    iou_threshold=0.45,
    augment=False,
    classes=None,
    agnostic_nms=False,
    device="cpu",
):

    img = letterbox(img0, image_size, stride=stride)[0]
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, conf_threshold, iou_threshold, classes=classes, agnostic=agnostic_nms
        )
    return img, pred


def extract_frame_from_yolo(img0, model, stride, device, patches, file, idx, save_folder, sort_tracker):
    colored_trk = True
    img, pred = get_prediction(img0, model, stride, image_size=224, device=device)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                # img0 = plot_one_box(xyxy, img0)
                #..................USE TRACK FUNCTION....................
                #pass an empty array to sort
                dets_to_sort = np.empty((0,6))
                
                # NOTE: We send in detected object class too
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks = sort_tracker.getTrackers()
                #loop over tracks
                indexes = patch_detect(tracks, patches)
                tracker_list = []
                for track in tracks:
                    tracker_list.append(track.patch_overlap_count)

                cv2.imwrite(file, img0)
                image = cv2.imread(file)

                for track in tracks:
                    # print(type(image), image.shape)
                    # print(image)
                    # color = compute_color_for_labels(id)cls
                    #draw colored tracks
                    if colored_trk and track.patch_overlap_count > 8:
                        # [cv2.line(image, (int(track.centroidarr[i][0]),
                        #             int(track.centroidarr[i][1])), 
                        #             (int(track.centroidarr[i+1][0]),
                        #             int(track.centroidarr[i+1][1])),
                        #             (5, 255, 5), thickness=1) 
                        #             for i, _ in  enumerate(track.centroidarr) 
                        #             if i < len(track.centroidarr)-1 ]
                        x1, y1, x2, y2 = [int(i) for i in track.bbox_history[-1][:4]]

                        cv2.rectangle(image, (x1, y1), (x2, y2), (5, 5, 255), 1)
            cv2.imwrite('anomaly.jpg', image)
            cv2.imwrite(os.path.join(save_folder, f'frame_{str(idx+1).zfill(3)}.jpg'), image)


# model, stride, device = get_model()
# extract_frame_from_yolo('test.avi', model, stride, device)