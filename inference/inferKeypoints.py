from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
import math
from matplotlib import pyplot as plt

import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_max_preds, get_final_preds
from utils.transforms import get_affine_transform

def parse_args():
    parser = argparse.ArgumentParser(description='Inference with trained model')
    # general
    parser.add_argument('--cfg',
                        help='Experiment configure file name.',
                        required=True,
                        type=str)

    parser.add_argument('--model',
                        help="Inference model .pth file (from output dir).",
                        required=True,
                        type=str)
                        
    parser.add_argument('--center',
                        help="Center of bounding box for inference image. Two floating point numbers.",
                        required=True,
                        type=float,  # Convert each input to a float
                        nargs='+')   # Accept multiple values (e.g., "10 20 30")
                        
    parser.add_argument('--scale',
                        help="Scale for bounding box.",
                        required=True,
                        type=float)
                        
    parser.add_argument('--imagePath',
                        help="Path to the image from origin.",
                        required=True,
                        type=str)
                        
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
                        
    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args

def save_heatmap_overlay(input_image, heatmaps, file_name, center, scale, normalize=True):
    '''
    input_image: [channel, height, width] - Single image tensor.
    heatmaps: [num_joints, height, width] - Heatmaps for the single image.
    file_name: Path to save the output overlayed heatmap image.
    normalize: Whether to normalize the input image.
    '''
    if isinstance(heatmaps, torch.Tensor):  # If heatmaps is a PyTorch tensor
        heatmaps = heatmaps.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

    assert isinstance(heatmaps, np.ndarray), "Heatmaps must be a numpy array"
    print(f"Heatmap shape: {heatmaps.shape}") # [1,16,256,256]
    input_image = np.array(input_image)
    #if normalize:
    #    input_image = input_image.copy()
    #    min_val = float(input_image.min())
    #    max_val = float(input_image.max())
    #    input_image.add_(-min_val).div_(max_val - min_val + 1e-5)

    num_joints = heatmaps.shape[1]  # Access second dimension (num keypints)
    heatmap_height, heatmap_width = heatmaps.shape[2], heatmaps.shape[3]

    # Prepare the image
    if input_image.ndim == 4:  # If input_image is a batch
        input_image = input_image[0]  # Use the first image in the batch
    #image = input_image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    resized_image = cv2.resize(input_image, (heatmap_width, heatmap_height))

    # Get predictions and maximum values
    preds, maxvals = get_final_preds(cfg, heatmaps, np.asarray([center]), np.asarray([scale]))
    #preds_image = resized_image.copy()
    heatmaps = heatmaps[0]
    #print(heatmaps.shape) #[16,256,256]
    composite_heatmap = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)

    for j in range(num_joints):
        
        print("Maxvals ",j,": ",maxvals[0][j][:])
        if maxvals[0][j][0] > 0.5: 
            col = [255,255,0]
            print("Prediction ",j,": ",preds[0][j][:])
        else:
            col = [0,0,255]
            print("###Prediction ",j,": ",preds[0][j][:])
        cv2.circle(input_image,
                       (int(preds[0][j][0]), int(preds[0][j][1])),
                       2, col, 2)
        heatmap = heatmaps[j, :, :] # heatmaps has NUM_KEYPOINTS images
        #composite_heatmap += heatmap  # Accumulate heatmaps

        # Normalize the composite heatmap to [0, 255]
        #composite_heatmap = (composite_heatmap / composite_heatmap.max() * 255).astype(np.uint8)
        heatmap = ((heatmap / heatmap.max()) * 255).astype(np.uint8)
        # Colorize the composite heatmap
        #colored_composite = cv2.applyColorMap(composite_heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # Overlay the composite heatmap on the original image
        overlay_image = (heatmap * 0.7 + resized_image * 0.3).astype(np.uint8)

    # Save the image
    #cv2.imwrite(file_name, grid_image)
        #cv2.imwrite(f"inference/{cfg.DATASET.DATASET}/{j}{file_name}", overlay_image)
    cv2.imwrite(f"inference/{cfg.DATASET.DATASET}/preds_{file_name}", input_image)
    #grid_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    #plt.imshow(grid_image_rgb)
    #plt.show()
    #print(f"Overlay heatmap saved to {file_name}")



def infer(config, center, scale, inferenceImagePath, modelFileName):
    print(inferenceImagePath)
    image = cv2.imread(inferenceImagePath)  # Open the image and convert to RGB
    
    center = np.array(center)
    # Define the preprocessing pipeline (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    rotation = 0
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    # Crop smaller image of people
    warpedImage = cv2.warpAffine(
        np.array(image),
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    input_image = transform(warpedImage).unsqueeze(0) 

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    #model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    checkpoint = torch.load(modelFileName, map_location=torch.device('cpu'))
    # Remove 'module.' prefix if present, MAYBE THIS IS THE ISSUE???
    if "checkpoint" in modelFileName:
        state_dict = checkpoint['state_dict']  # Adjust as per your checkpoint structure
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = checkpoint
    model.load_state_dict(new_state_dict)

    # switch to evaluate mode
    model.eval()
    #model = model.cuda() # move to gpu

    with torch.no_grad():
        end = time.time()
        #input_image = input_image.cuda() # move image to GPU
        # compute output
        print("Image shape: ",input_image.shape)
        outputs = model(input_image)
        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        prefix = os.path.basename(inferenceImagePath)
        print("Output shape before save_heatmap_overlay:", output.shape)

        save_heatmap_overlay(image, output.clone().cpu().numpy(), f'{prefix}_heatmap_pred.jpg',
                            center, scale)
                

if __name__ == '__main__':
    args = parse_args()
    print(args)
    update_config(cfg, args)
    infer(cfg, args.center, args.scale,args.imagePath,args.model)