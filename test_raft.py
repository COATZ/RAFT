import sys
sys.path.append('core')

import argparse
import os
# import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT as RAFT
from raft_sphe import RAFT as RAFT_SPHE
from utils import flow_viz
from utils.utils import InputPadder
from utils.frame_utils import readFlow


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo, save_file, save_dir):
    print(save_file)
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    plt.imsave(os.path.join(save_dir, save_file),img_flo / 255.0)

    # cv2.imshow(img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    save_dir = './OUTPUT/'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            save_file = imfile1.split("/")[-1]
            viz(image1, flow_up, save_file, save_dir)

def demo_sphe(args):
    model = torch.nn.DataParallel(RAFT_SPHE(args))
    model.load_state_dict(torch.load(args.model))

    print(model)

    model = model.module
    model.to(DEVICE)
    model.eval()

    save_dir = './OUTPUT/'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            save_file = imfile1.split("/")[-1]
            viz(image1, flow_up, save_file, save_dir)

def compare_gt(args):
    model = torch.nn.DataParallel(RAFT_SPHE(args))
    model.load_state_dict(torch.load(args.model))

    print(model)

    model = model.module
    model.to(DEVICE)
    model.eval()

    save_dir = './OUTPUT/'
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path+'/images_1000_500/', '*.png')) + \
                 glob.glob(os.path.join(args.path+'/images_1000_500/', '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            save_file = imfile1.split("/")[-1]
            # print(args.path+'/ground_truth/'+save_file[:-4]+'.flo')
            # flow_gt = readFlow(args.path+'/ground_truth/'+save_file[:-4]+'.flo')
            # print(flow_gt)
            # print(flow_gt.shape)
            
            # map flow to rgb image
            # img_flo_flow_gt = flow_viz.flow_to_image(flow_gt)
        
            # import matplotlib.pyplot as plt
            # plt.imsave(args.path+'/ground_truth/'+save_file, img_flo_flow_gt / 255.0)
            

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            viz(image1, flow_up, save_file, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo_sphe(args)
    # compare_gt(args)
