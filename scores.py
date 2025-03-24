import argparse
import numpy as np
from cv2 import imread, imwrite, resize as imresize
import cv2

import os
import sys
import glob
import time

def fast_hist(im, gt, n=9):
    """
    n is num_of_classes
    """
    k = (gt >= 0) & (gt < n)
    return np.bincount(n * gt[k].astype(int) + im[k], minlength=n**2).reshape(n, n)

from utils.rgb_ind_convertor import *

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='R3D',
					help='define the benchmark')

parser.add_argument('--result_dir', type=str, default='./out',
					help='define the storage folder of network prediction')

def evaluate_semantic(benchmark_path, result_dir, num_of_classes=11, need_merge_result=False, out_downsample=True, gt_downsample=True):
    # benchmark file is a TSV file with the following format:
    # [orig]\t[wall]\t[door(close)]\t[room]\t[close_wall]
    # Each line is a test sample.
    
    # get the benchmark file path
	gt_paths = open(benchmark_path, 'r').read().splitlines()
	gt_d_paths = [p.split('\t')[2] for p in gt_paths] # 1 denote wall, 2 denote door, 3 denote room
	gt_r_paths = [p.split('\t')[3] for p in gt_paths] # 1 denote wall, 2 denote door, 3 denote room
	gt_cw_paths = [p.split('\t')[-1] for p in gt_paths] # 1 denote wall, 2 denote door, 3 denote room, last one denote close wall
 
	# get the result file path
	out_r_paths = [os.path.join(result_dir, p.split('/')[-1]) for p in gt_r_paths]
	if need_merge_result:
		out_r_paths = [os.path.join(result_dir+'/room', p.split('/')[-1]) for p in gt_r_paths]
		out_d_paths = [os.path.join(result_dir+'/door', p.split('/')[-1]) for p in gt_d_paths]
		out_cw_paths = [os.path.join(result_dir+'/close_wall', p.split('/')[-1]) for p in gt_cw_paths]

	n = len(out_r_paths)
	hist = np.zeros((num_of_classes, num_of_classes))
 
	for i in range(n):
		# read the result and ground truth
		out_r = imread(out_r_paths[i], cv2.IMREAD_COLOR)[:,:,::-1] # BGR to RGB
		if need_merge_result:
			out_d = imread(out_d_paths[i], cv2.IMREAD_GRAYSCALE)
			out_cw = imread(out_cw_paths[i], cv2.IMREAD_GRAYSCALE)
   
		# create fuse semantic label
		gt_r = imread(gt_r_paths[i], cv2.IMREAD_COLOR)[:,:,::-1] # BGR to RGB
		gt_cw = imread(gt_cw_paths[i], cv2.IMREAD_GRAYSCALE)
		gt_d = imread(gt_d_paths[i], cv2.IMREAD_GRAYSCALE)

		# The output image is downsampled to 512 × 512
		if out_downsample:
			out_r = imresize(out_r, (512, 512))
			if need_merge_result:
				out_d = imresize(out_d, (512, 512))
				out_cw = imresize(out_cw, (512, 512))
				out_d  //= 255
				out_cw //= 255

		# The ground truth image is downsampled to 512 × 512
		if gt_downsample:
			gt_cw = imresize(gt_cw, (512, 512))
			gt_d = imresize(gt_d, (512, 512))
			gt_r = imresize(gt_r, (512, 512))

		# normalize (The cw and d are binary images)
		gt_cw //= 255
		gt_d //= 255

		# convert the RGB image to index image
		out_r_ind = rgb2ind(out_r, color_map=floorplan_fuse_map)
		if out_r_ind.sum() == 0:
			out_r_ind = rgb2ind(out_r + 1)
		gt_r_ind = rgb2ind(gt_r, color_map=floorplan_fuse_map)
		if gt_r_ind.sum() == 0:
			gt_r_ind = rgb2ind(gt_r + 1)

		# merge the label and produce on results
		if need_merge_result:
			out_d  = (out_d > 0.5).astype(np.uint8)
			out_cw = (out_cw > 0.5).astype(np.uint8)
			out_r_ind[out_cw == 1] = 10
			out_r_ind[out_d == 1] = 9

		# merge the label and produce on ground truth
		gt_cw = (gt_cw > 0.5).astype(np.uint8)
		gt_d = (gt_d > 0.5).astype(np.uint8)
		gt_r_ind[gt_cw == 1] = 10
		gt_r_ind[gt_d == 1] = 9

		out_r_name = out_r_paths[i].split('/')[-1]
		gt_r_name = gt_r_paths[i].split('/')[-1]
		
		print('Evaluating {}(im) <=> {}(gt)...'.format(out_r_name, gt_r_name))

		hist += fast_hist(out_r_ind.flatten(), gt_r_ind.flatten(), num_of_classes)

	print('*' * 60)
	# overall accuracy
	acc = np.diag(hist).sum() / hist.sum()
	print('overall accuracy {:.4}'.format(acc))
	# per-class accuracy, avoid div zero
	acc = np.diag(hist) / (hist.sum(1) + 1e-6)
	print('room-type: mean accuracy {:.4}, room-type+bd: mean accuracy {:.4}'.format(np.nanmean(acc[:7]), (np.nansum(acc[:7]) + np.nansum(acc[-2:])) / 9.))
	for t in range(0, acc.shape[0]):
		if t not in [7, 8]:
			print('room type {}th, accuracy = {:.4}'.format(t, acc[t]))

	print('*' * 60)
	# per-class IU, avoid div zero
	iu = np.diag(hist) / (hist.sum(1) + 1e-6 + hist.sum(0) - np.diag(hist))
	print('room-type: mean IoU {:.4}, room-type+bd: mean IoU {:.4}'.format(np.nanmean(iu[:7]), (np.nansum(iu[:7]) + np.nansum(iu[-2:])) / 9.))
	for t in range(iu.shape[0]):
		if t not in [7, 8]: # ignore class 7 & 8
			print('room type {}th, IoU = {:.4}'.format(t, iu[t]))

if __name__ == '__main__':
	FLAGS, unparsed = parser.parse_known_args()

	if FLAGS.dataset.lower() == 'r2v':
		benchmark_path = './dataset/r2v_test.txt'
	else:
		benchmark_path = './dataset/r3d_test.txt'

	result_dir = FLAGS.result_dir

	tic = time.time()
	evaluate_semantic(benchmark_path, result_dir, need_merge_result=True, out_downsample=True, gt_downsample=True) # same as previous line but 11 classes by combining the opening and wall line

	print("*" * 60)
	print("Evaluate time: {} sec".format(time.time() - tic))