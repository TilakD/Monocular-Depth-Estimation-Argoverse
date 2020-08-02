import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image
import torch
import torch.nn as nn
from torchvision import transforms


lib_path = os.path.abspath(os.path.join(os.path.realpath(__file__), '../..'))
if lib_path not in sys.path:
    sys.path.append(lib_path)
    
from bts import BtsModel

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--save_name', type=str, help='model name', \
                    default='./argo_val_eval/')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts or resnext101_bts',\
                     default='resnext101_bts')
parser.add_argument('--media_path', type=str, help='path to the data', required=True)
parser.add_argument('--dataset', type=str, help='dataset to teest on', default='argo')
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=200)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', \
                    default='./models/bts_resnext101_argo/model-291500')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)
parser.add_argument('--min_depth_eval',type=float, help='minimum depth for evaluation', \
    default=1e-3)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def test_images(params):
    """Test function."""
    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
 
    # apply transformations    
    loader_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    pred_depths = []
    
    save_name = args.save_name
    if not os.path.exists(save_name + 'raw'):
        os.mkdir(save_name + 'raw')

    with torch.no_grad():
        gt_file = open(args.media_path, "r")
        num_test_samples = len(gt_file.readlines())
        print(num_test_samples)
        max_distance_list = [10, 25, 50, 100, 150, 200]
        for max_dist in max_distance_list:
            globals()['silog_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['log10_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['rms_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['log_rms_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['abs_rel_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['sq_rel_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['d1_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['d2_%s' % max_dist] = np.zeros(num_test_samples, np.float32)
            globals()['d3_%s' % max_dist] = np.zeros(num_test_samples, np.float32)

        i = 0
        gt_file = open(args.media_path, "r")
        for x in gt_file:
            output = x.split()
            file_sub_path = output[0]
            focal = output[2]
            print(file_sub_path, focal)
            media_path = os.path.join("../train_val_split/val/rgb/", file_sub_path)

            save_dir = os.path.join(args.save_name, "raw", os.path.dirname(file_sub_path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            image = cv2.imread(media_path) 
            height, width, _ = image.shape
            #check divisibility be 32
            adjusted_height = lambda height: 32*(math.ceil(height/32)) if height%32 !=0 else height
            adjusted_width = lambda height: 32*(math.ceil(width/32)) if width%32 !=0 else width
            image_original = cv2.resize(image, (adjusted_width(width), adjusted_height(height)), interpolation = cv2.INTER_LANCZOS4)

            image = loader_transforms(image_original).float().cuda()
            image = image.unsqueeze(0)
            _, _, _, _, depth_est = model(image, focal)

            depth_est = depth_est.cpu().numpy().squeeze()       
            depth_est_scaled = cv2.resize(depth_est, (1920, 1200), interpolation = cv2.INTER_LANCZOS4)    

            filename_pred_png = os.path.join(save_dir, os.path.splitext(os.path.basename(media_path))[0] + ".png")
            pred_depth_scaled = depth_est_scaled * 256.0
            
            pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
            cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])


            gt_depth_path = os.path.join("../train_val_split/val/depth/", file_sub_path)
            depth = cv2.imread(gt_depth_path, -1)

            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(i)
                continue
            gt_depth = depth.astype(np.float32) / 256.0
            gt_depth_copy = np.copy(gt_depth)
            gt_depth[gt_depth > 0] = 1 #mask

            pred_depth_maskd = np.where(gt_depth, depth_est_scaled, 0)

            for max_dist in max_distance_list:
                pred_depth_maskd[pred_depth_maskd < args.min_depth_eval] = args.min_depth_eval
                pred_depth_maskd[pred_depth_maskd > max_dist] = max_dist
                pred_depth_maskd[np.isinf(pred_depth_maskd)] = max_dist

                gt_depth_copy[np.isinf(gt_depth_copy)] = 0
                gt_depth_copy[np.isnan(gt_depth_copy)] = 0

                valid_mask = np.logical_and(gt_depth_copy > args.min_depth_eval, gt_depth_copy < max_dist)
                if max_dist == 10:
                    silog_10[i], log10_10[i], abs_rel_10[i], sq_rel_10[i], rms_10[i], log_rms_10[i], d1_10[i], d2_10[i], d3_10[i] = compute_errors(gt_depth_copy[valid_mask], pred_depth_maskd[valid_mask])
                elif max_dist == 25:
                    silog_25[i], log10_25[i], abs_rel_25[i], sq_rel_25[i], rms_25[i], log_rms_25[i], d1_25[i], d2_25[i], d3_25[i] = compute_errors(gt_depth_copy[valid_mask], pred_depth_maskd[valid_mask])
                elif max_dist == 50:
                    silog_50[i], log10_50[i], abs_rel_50[i], sq_rel_50[i], rms_50[i], log_rms_50[i], d1_50[i], d2_50[i], d3_50[i] = compute_errors(gt_depth_copy[valid_mask], pred_depth_maskd[valid_mask])
                elif max_dist == 100:
                    silog_100[i], log10_100[i], abs_rel_100[i], sq_rel_100[i], rms_100[i], log_rms_100[i], d1_100[i], d2_100[i], d3_100[i] = compute_errors(gt_depth_copy[valid_mask], pred_depth_maskd[valid_mask])
                elif max_dist == 150:
                    silog_150[i], log10_150[i], abs_rel_150[i], sq_rel_150[i], rms_150[i], log_rms_150[i], d1_150[i], d2_150[i], d3_150[i] = compute_errors(gt_depth_copy[valid_mask], pred_depth_maskd[valid_mask])
                elif max_dist == 200:
                    silog_200[i], log10_200[i], abs_rel_200[i], sq_rel_200[i], rms_200[i], log_rms_200[i], d1_200[i], d2_200[i], d3_200[i] = compute_errors(gt_depth_copy[valid_mask], pred_depth_maskd[valid_mask])
                    print(silog_200[i], log10_200[i], abs_rel_200[i], sq_rel_200[i], rms_200[i], log_rms_200[i], d1_200[i], d2_200[i], d3_200[i])
            i+=1
        
        eval_dump_dist = {}
        eval_dump_dist[max_distance_list[0]] = {'d1' : float(d1_10.mean()), 
                                            'd2' : float(d2_10.mean()), 
                                            'd3' : float(d3_10.mean()),
                                            'abs_rel' : float(abs_rel_10.mean()), 
                                            'sq_rel' : float(sq_rel_10.mean()), 
                                            'rms' : float(rms_10.mean()), 
                                            'log_rms' : float(log_rms_10.mean()), 
                                            'silog' : float(silog_10.mean()), 
                                            'log10' : float(log10_10.mean())}
        eval_dump_dist[max_distance_list[1]] = {'d1' : float(d1_25.mean()), 
                                            'd2' : float(d2_25.mean()), 
                                            'd3' : float(d3_25.mean()),
                                            'abs_rel' : float(abs_rel_25.mean()), 
                                            'sq_rel' : float(sq_rel_25.mean()), 
                                            'rms' : float(rms_25.mean()), 
                                            'log_rms' : float(log_rms_25.mean()), 
                                            'silog' : float(silog_25.mean()), 
                                            'log10' : float(log10_25.mean())}
        eval_dump_dist[max_distance_list[2]] = {'d1' : float(d1_50.mean()), 
                                            'd2' : float(d2_50.mean()), 
                                            'd3' : float(d3_50.mean()),
                                            'abs_rel' : float(abs_rel_50.mean()), 
                                            'sq_rel' : float(sq_rel_50.mean()), 
                                            'rms' : float(rms_50.mean()), 
                                            'log_rms' : float(log_rms_50.mean()), 
                                            'silog' : float(silog_50.mean()), 
                                            'log10' : float(log10_50.mean())}
        eval_dump_dist[max_distance_list[3]] = {'d1' : float(d1_100.mean()), 
                                            'd2' : float(d2_100.mean()), 
                                            'd3' : float(d3_100.mean()),
                                            'abs_rel' : float(abs_rel_100.mean()), 
                                            'sq_rel' : float(sq_rel_100.mean()), 
                                            'rms' : float(rms_100.mean()), 
                                            'log_rms' : float(log_rms_100.mean()), 
                                            'silog' : float(silog_100.mean()), 
                                            'log10' : float(log10_100.mean())}
        eval_dump_dist[max_distance_list[4]] = {'d1' : float(d1_150.mean()), 
                                            'd2' : float(d2_150.mean()), 
                                            'd3' : float(d3_150.mean()),
                                            'abs_rel' : float(abs_rel_150.mean()), 
                                            'sq_rel' : float(sq_rel_150.mean()), 
                                            'rms' : float(rms_150.mean()), 
                                            'log_rms' : float(log_rms_150.mean()), 
                                            'silog' : float(silog_150.mean()), 
                                            'log10' : float(log10_150.mean())}
        eval_dump_dist[max_distance_list[5]] = {'d1' : float(d1_200.mean()), 
                                            'd2' : float(d2_200.mean()), 
                                            'd3' : float(d3_200.mean()),
                                            'abs_rel' : float(abs_rel_200.mean()), 
                                            'sq_rel' : float(sq_rel_200.mean()), 
                                            'rms' : float(rms_200.mean()), 
                                            'log_rms' : float(log_rms_200.mean()), 
                                            'silog' : float(silog_200.mean()), 
                                            'log10' : float(log10_200.mean())}

    return eval_dump_dist

if __name__ == '__main__':
    eval_dump_dist = test_images(args)
    dump_dir = os.path.join(args.save_name , "raw", "eval_dump")
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    json_output_file = os.path.join(dump_dir, os.path.splitext(os.path.basename(args.media_path))[0] + ".json")
    with open(json_output_file, 'w') as fd:
        json.dump(eval_dump_dist, fd, indent=4, sort_keys=True)
