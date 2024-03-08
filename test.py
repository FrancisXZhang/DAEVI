# -*- coding: utf-8 -*-
import sys
import cv2
from PIL import Image
import numpy as np
import time
import importlib
import os
import argparse
import json
import pathlib

import torch
import torch.nn.functional as F
from torchvision import transforms
from core.utils import ZipReader

# My libs
from core.utils import Stack, ToTorchFormatTensor

# depth libs
from depth_layers import disp_to_depth
from depth_model.depth_decoder import DepthDecoder
from depth_model.resnet_encoder import ResnetEncoder

import logging

# Configure logging with time format
logging.basicConfig(
    filename='test_DAS_faster.log',
    filemode='w',
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
# set the logging level
logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="STTN")
parser.add_argument("-f", "--frame", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-c", "--ckptpath", type=str, required=True)
parser.add_argument("-cn", "--ckptnumber", type=str, required=True)
parser.add_argument("--model", type=str, default='sttn')
parser.add_argument("--shifted", action='store_true')
parser.add_argument("--overlaid", action='store_true')
parser.add_argument("--famelimit", type=int, default=927)
parser.add_argument("--zip", action='store_true')
parser.add_argument("-g", "--gpu", type=str, default="7", required=True)
parser.add_argument("-d", "--Dil", type=int, default=8)
parser.add_argument("-r", "--readfiles", action='store_true')
parser.add_argument("--ref_num", type=int, default=-1)

args = parser.parse_args()


ref_length = 10
neighbor_stride = 5
default_fps = 24
window_size = 128


epsilon=sys.float_info.epsilon

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# sample reference frames from the whole video 
def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index



# read frame-wise masks 
def read_mask(mpath):
    masks = []
    mnames = os.listdir(mpath)
    mnames.sort()
    for m in mnames: 
        m = Image.open(os.path.join(mpath, m))
        sz=m.size
        m = np.array(m.convert('L'))
        m = np.array(m > 199).astype(np.uint8) 
        if args.Dil !=0:
            m = cv2.dilate(m, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.Dil, args.Dil)), iterations=1) 
        if args.shifted:
            M = np.float32([[1,0,50],[0,1,0]])
            m_T = cv2.warpAffine(m,M,sz)  
            m_T[m!=0]=0
            m = np.copy(m_T)
        masks.append(Image.fromarray(m*255))
    return masks


#  read frames from video 
def read_frames(fpath):
    frames = []
    fnames = os.listdir(fpath)
    fnames.sort()
    for f in fnames: 
        f = Image.open(os.path.join(fpath, f))
        frames.append(f)
    return frames, fnames

def read_frames_mask_zip(fpath, mpath):
    frames = {}
    masks = {}
    fnames = {}
    with open(os.path.join(os.path.abspath(os.path.join(fpath, os.pardir)), 'test.json'), 'r') as f:
        video_dict = json.load(f)
    video_names = list(video_dict.keys())
    for video_name in video_names: #[:1]:
        frames_v = []
        masks_v = []       
        zfilelist = ZipReader.filelist("{}/{}.zip".format(
            fpath, video_name)) 
        fnames[video_name]=zfilelist
        for zfile in zfilelist: #[:100]:
            img = ZipReader.imread('{}/{}.zip'.format(
                fpath, video_name), zfile).convert('RGB')
            frames_v.append(img)
            m = ZipReader.imread('{}/{}.zip'.format(
                mpath, video_name), zfile).convert('RGB')
            sz=m.size
            m = np.array(m.convert('L'))
            m = np.array(m > 199).astype(np.uint8) 
            if args.Dil !=0:
                m = cv2.dilate(m, cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (args.Dil, args.Dil)), iterations=1) 
            if args.shifted:
                M = np.float32([[1,0,50],[0,1,0]])
                m_T = cv2.warpAffine(m,M,sz)  
                m_T[m!=0]=0
                m = np.copy(m_T)
            all_mask=Image.fromarray(m*255)
            masks_v.append(all_mask)
        frames[video_name]=frames_v
        masks[video_name]=masks_v
        print(video_name)
        
    return frames, fnames, masks, video_names, sz

def evaluate(w, h, frames, fnames, masks, video_name, model, device, overlaid, shifted, Dil):
    #added for memory issue
    if len(frames)>args.famelimit or len(masks)>args.famelimit:
        masks=masks[:args.famelimit]
        frames=frames[:args.famelimit]
    
    logging.info('start evaluating')
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0)*2-1
    frames = [np.array(f).astype(np.uint8) for f in frames]

    logging.info('video_length: {}'.format(video_length))
    binary_masks = [np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks]
    masks = _to_tensors(masks).unsqueeze(0)
    feats, masks = feats.to(device), masks.to(device)
    comp_frames = [None]*video_length
    pred_frames = [None]*video_length
    
    with torch.no_grad():
        # Initialize an empty tensor for storing encoded features
        all_feats = []
        all_depths = []

        # count window number
        window_number = video_length // window_size + 1

        logging.info('window_number: {}'.format(window_number))
        # Process frames in windows
        for start in range(0, window_number):
            logging.info('start: {}'.format(start))
            end = min(start * window_size + window_size, video_length)
            window_feats = feats[:, start * window_size: end, :, :, :]  # Select frames for the current window

            # Encode the current window of frames
            input = (window_feats * (1 - masks[:, start * window_size:end, :, :, :]).float())
            input = input.view(end - start * window_size, 3, h, w)
            encoded_window = model.encoder(input)
            depth_encoded_window = model.decoder_depth(encoded_window)
            depth_encoded_window = F.interpolate(depth_encoded_window, scale_factor=1.0/4)

            all_feats.append(encoded_window.detach().cpu())
            all_depths.append(depth_encoded_window.detach().cpu())

        logging.info('all_feats: {}'.format(len(all_feats)))
        # Concatenate all encoded features
        feats = torch.cat(all_feats, dim=0)
        depths = torch.cat(all_depths, dim=0)

        logging.info('feats: {}'.format(feats.size()))
        # Reshape feats to match the expected format
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)
        depths = depths.view(1, video_length, 1, feat_h, feat_w)
        logging.info('reshaped feats: {}'.format(feats.size()))
    
    print('loading frames and masks from: {}'.format(args.frame))

    # completing holes by spatial-temporal transformers
    logging.info('Completing holes by spatial-temporal transformers')
    for f in range(0, video_length, neighbor_stride):
        logging.info('f: {}'.format(f))
        neighbor_ids = [i for i in range(max(0, f-neighbor_stride), min(video_length, f+neighbor_stride+1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_num=args.ref_num)
        print('neighbor_ids',neighbor_ids)
        print('ref_ids',ref_ids)
        with torch.no_grad():
            feats_infer = feats[0, neighbor_ids+ref_ids, :, :, :].to(device)
            mask_infer = masks[0, neighbor_ids+ref_ids, :, :, :].to(device)
            depths_infer = depths[0, neighbor_ids+ref_ids, :, :, :].to(device)
           
            
            feats_infer = model.fusion(feats_infer, depths_infer)

            pred_feat = model.infer(feats_infer, mask_infer)


            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy()*255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                if args.overlaid:
                    overlay_mult=binary_masks[idx]
                    overlay_add=frames[idx] * (1-binary_masks[idx])
                else:
                    overlay_mult=1
                    overlay_add=0

                output = np.array(pred_img[i]).astype(np.uint8)
                img = np.array(pred_img[i]).astype(np.uint8)*overlay_mult + overlay_add

                if pred_frames[idx] is None:
                    pred_frames[idx] = output
                else:
                    pred_frames[idx] = pred_frames[idx].astype(
                        np.float32)*0.5 + output.astype(np.float32)*0.5
                
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32)*0.5 + img.astype(np.float32)*0.5

    logging.info('Saving results')
    # save results
    logging.info('save results')
    savebasepath=os.path.join(args.output,"gen_"+args.ckptnumber.zfill(5),"full_video",video_name, overlaid, shifted, Dil)
    frameresultpath=os.path.join(savebasepath,"frameresult")
    pathlib.Path(frameresultpath).mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(savebasepath+"/result.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    for f in range(video_length):
        if args.overlaid:
            overlay_mult=binary_masks[f]
            overlay_add=frames[f] * (1-binary_masks[f])
        else:
            overlay_mult=1
            overlay_add=0
        comp = np.array(comp_frames[f]).astype(
            np.uint8)*overlay_mult+overlay_add
        fnameNew=os.path.basename(fnames[f])
        cv2.imwrite(frameresultpath+f"/{fnameNew}",cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
        writer.write(cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB))
    writer.release()
    print('Finish in {}'.format(savebasepath+"/result.mp4"))

    # save output
    logging.info('save output')
    savebasepath=os.path.join(args.output,"gen_"+args.ckptnumber.zfill(5),"full_video",video_name, overlaid, shifted, Dil)
    predresultpath=os.path.join(savebasepath,"predresult")
    pathlib.Path(predresultpath).mkdir(parents=True, exist_ok=True)
    pred_writer = cv2.VideoWriter(savebasepath+"/predresult.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    for f in range(video_length):
        output = np.array(pred_frames[f]).astype(np.uint8)
        fnameNew=os.path.basename(fnames[f])
        cv2.imwrite(predresultpath+f"/{fnameNew}",cv2.cvtColor(np.array(output).astype(np.uint8), cv2.COLOR_BGR2RGB))
        pred_writer.write(cv2.cvtColor(np.array(output).astype(np.uint8), cv2.COLOR_BGR2RGB))
    pred_writer.release()
    print('Finish in {}'.format(savebasepath+"/predresult.mp4"))

    # save depth results
    logging.info('save depth results')
    savebasepath=os.path.join(args.output,"gen_"+args.ckptnumber.zfill(5),"full_video",video_name, overlaid, shifted, Dil)
    depthresultpath=os.path.join(savebasepath,"depthresult")
    pathlib.Path(depthresultpath).mkdir(parents=True, exist_ok=True)
    dep_writer = cv2.VideoWriter(savebasepath+"/depthresult.mp4", cv2.VideoWriter_fourcc(*"mp4v"), default_fps, (w, h))
    depths = F.interpolate(depths, scale_factor=4)
    depths = (depths + 1) / 2 * max(torch.max(depths), epsilon)
    depths, _ = disp_to_depth(depths, 0.1, 150)
    depths, _ = disp_to_depth(depths, 0.1, 150)
    for f in range(video_length):
        depth = np.array(depths[0, f, 0, :, :].cpu()).astype(np.uint8)
        depth = np.repeat(depth[:, :, np.newaxis], 3, axis=2)
        fnameNew=os.path.basename(fnames[f])
        cv2.imwrite(depthresultpath+f"/{fnameNew}",depth)
        dep_writer.write(depth)
    dep_writer.release()
    print('Finish in {}'.format(savebasepath+"/depthresult.mp4"))

def main_worker():
    overlaid="overlaid" if args.overlaid else "notoverlaid"
    shifted="shifted" if args.shifted else "notshifted"
    Dil = "noDil" if args.Dil == 0 else ""
    # set up models 
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    

    model = net.InpaintGenerator(in_channels=3).to(device)
    model_path = os.path.join(args.ckptpath,"gen_"+args.ckptnumber.zfill(5)+".pth")
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data['netG'])
    print('loading from: {}'.format(args.ckptpath))
    model.eval()

    if args.zip:
        file1 = os.path.join(os.path.abspath(os.path.join(args.frame, os.pardir)), 'files/testframes_v.npy') # 'files/frames_v.npy')
        file2 = os.path.join(os.path.abspath(os.path.join(args.frame, os.pardir)), 'files/testfnames_v.npy') # 'files/fnames_v.npy')
        file3 = os.path.join(os.path.abspath(os.path.join(args.frame, os.pardir)), 'files/testmasks_v.npy') # 'files/masks_v.npy')
        file4 = os.path.join(os.path.abspath(os.path.join(args.frame, os.pardir)), 'files/testvideo_names.npy') # 'files/video_names.npy')
        file5 = os.path.join(os.path.abspath(os.path.join(args.frame, os.pardir)), 'files/testsz.npy') # 'files/sz.npy')
        file1Ex = os.path.isfile(file1)
        file2Ex = os.path.isfile(file2)
        file3Ex = os.path.isfile(file3)
        file4Ex = os.path.isfile(file4)
        file5Ex = os.path.isfile(file5)

        if file1Ex and file2Ex and file3Ex and file4Ex and file5Ex and args.readfiles:
            # start timer
            start = time.time()
            frames_v = np.load(file1, allow_pickle='TRUE').item()
            # end timer
            end = time.time()
            print("frames_v loaded")
            print(f"Time taken to load frames_v: {end - start} seconds") 
            fnames_v = np.load(file2, allow_pickle='TRUE').item()
            print("fnames_v loaded")
            masks_v = np.load(file3, allow_pickle='TRUE').item()
            print("masks_v loaded")
            video_names = np.load(file4, allow_pickle='TRUE')
            print("video_names loaded")
            sz = np.load(file5, allow_pickle='TRUE')
            print("sz loaded")
            print("files loaded...")
        else:
            os.makedirs(os.path.join(os.path.abspath(os.path.join(args.frame, os.pardir)), 'files'), exist_ok=True)
            frames_v, fnames_v, masks_v, video_names, sz = read_frames_mask_zip(args.frame, args.mask)
            np.save(file1, frames_v) 
            np.save(file2, fnames_v) 
            np.save(file3, masks_v) 
            np.save(file4, video_names) 
            np.save(file5, sz) 

        w, h = sz
        for video_name in video_names:
            frames = frames_v[video_name]
            fnames = fnames_v[video_name]
            masks = masks_v[video_name]
            logging.info('video_name: {}'.format(video_name))
            evaluate(w, h, frames, fnames, masks, video_name, model, device, overlaid, shifted, Dil)
    else:
        # prepare datset, encode all frames into deep space 
        video_name=os.path.basename(args.frame.rstrip("/"))
        frames, fnames = read_frames(args.frame)
        w, h=frames[0].size
        masks = read_mask(args.mask)
        logging.info('video_name: {}'.format(video_name))
        evaluate(w, h, frames, fnames, masks, video_name, model, depth_encoder, depth_decoder, device, overlaid, shifted, Dil)
     
  
if __name__ == '__main__':
    main_worker()
