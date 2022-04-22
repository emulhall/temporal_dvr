import cv2
import numpy as np
import sys
import os
import pickle
import torch
import glob
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
import point_cloud_utils as pcu
from utils import get_dp_correspondences


def convert(size):
	num_matches = 1000
	sys.path.append('/home/mulha024/Documents/detectron2/projects/DensePose')
	ROOT = '/media/mulha024/i/HUMBI_%i'%(size)
	#Set up directories to save results to
	
	subjects = natsorted(glob.glob(os.path.join(ROOT, '*')))

	for s in subjects:
		_, subject = os.path.split(s)
		time_stamps = natsorted(glob.glob(os.path.join(s, '0*')))

		for t in time_stamps:
			_, time = os.path.split(t)

			valid_cameras = glob.glob(os.path.join(t,'camera','*_K.txt'))
			overall_correspondence_dict = {}

			iuv_path = os.path.join(t,'correspondences')

			if not os.path.isdir(iuv_path):
				os.mkdir(iuv_path)

			for c_1 in valid_cameras:
				_,cam_num = os.path.split(c_1)
				cam_num=int(cam_num[:cam_num.index('_')])

				print("Processing camera %i"%(cam_num))
				
				dp_path = os.path.join(t,'densepose','%07d.png'%(cam_num))
				mask_path = os.path.join(t,'mask','%07d.png'%(cam_num))

				dp_1 = np.asarray(Image.open(dp_path))
				mask_1 = np.where(np.asarray(Image.open(mask_path))>0,1.0,0.0)

				if cam_num not in overall_correspondence_dict:
					overall_correspondence_dict[cam_num] = {}

				for c_2 in set(valid_cameras)-set([c_1]):
					_, tail = os.path.split(c_2)
					cam_num_2=int(tail[:tail.index('_')])

					if cam_num_2 in overall_correspondence_dict[cam_num]:
						continue

					print("\t Comparing with camera %i"%(cam_num_2))

					if cam_num_2 not in overall_correspondence_dict:
						overall_correspondence_dict[cam_num_2] = {}

					dp_path_2 = os.path.join(t,'densepose','%07d.png'%(cam_num_2))
					mask_path_2 = os.path.join(t,'mask','%07d.png'%(cam_num_2))

					dp_2 = np.asarray(Image.open(dp_path_2))
					mask_2 = np.where(np.asarray(Image.open(mask_path_2))>0,1.0,0.0)

					x1, x2,_ = get_dp_correspondences(dp_1, dp_2, mask_1[...,np.newaxis], mask_2[...,np.newaxis], num_matches=num_matches, threshold=1e-4, visualize=False)
					
					overall_correspondence_dict[cam_num][cam_num_2] = x1
					overall_correspondence_dict[cam_num_2][cam_num] = x2


			for corr in overall_correspondence_dict:
				save_path = os.path.join(iuv_path, '%07d.pkl'%(corr))
				with open(save_path, 'wb') as f:
					print(len(overall_correspondence_dict[corr]))
					pickle.dump(overall_correspondence_dict[corr], f)




if __name__ == '__main__':
	convert(512)