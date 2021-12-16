import torch
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
import scipy
import cv2
import os
import matplotlib.pyplot as plt

class RenderPeopleDataset(Dataset):
	def __init__(self, usage='train', pickle_file='./data/RenderPeople_dvr.pkl',skip_every_n_image=1):

		self.to_tensor=transforms.ToTensor()

		with open(pickle_file, 'rb') as f:
			self.data_info = pickle.load(f)[usage]
			self.idx = [i for i in range(0, len(self.data_info[0][0]), skip_every_n_image)]
			self.data_len = len(self.idx)


	def __getitem__(self, index):
		output = {}

		color = self.to_tensor(Image.open(self.data_info[0][0][index]).resize((256,256)))

		mask = cv2.resize(cv2.imread(self.data_info[0][2][index]),(256,256),interpolation=cv2.INTER_NEAREST)
		K = np.loadtxt(self.data_info[0][4][index],delimiter=',')
		R = np.loadtxt(self.data_info[0][5][index],delimiter=',')
		C = np.loadtxt(self.data_info[0][6][index],delimiter=',')

		gt_depth = np.genfromtxt(self.data_info[0][7][index],delimiter=',')
		gt_depth_3d=np.concatenate((gt_depth[...,np.newaxis],gt_depth[...,np.newaxis],gt_depth[...,np.newaxis]),axis=2)
		gt_depth_3d = cv2.resize(gt_depth_3d, (256,256), interpolation=cv2.INTER_NEAREST)
		gt_depth=gt_depth_3d[...,0]

		#Refine the mask with valid depth
		valid_depth=np.where(gt_depth>=1.0, np.ones_like(gt_depth),np.zeros_like(gt_depth))*np.where(gt_depth<=8.0, np.ones_like(gt_depth),np.zeros_like(gt_depth))
		mask_refined = self.to_tensor(np.where(mask>100,np.ones_like(mask),np.zeros_like(mask))*valid_depth[...,np.newaxis])
		mask_refined_1D = mask_refined[0,...]

		gt_depth = self.to_tensor(np.asarray(gt_depth,dtype='f'))

		origin = self.data_info[0][13][index]

		scaling = 2*np.asarray([self.data_info[0][14][index]],dtype='f')

		#Refine color image, depth image and normal image using the refined mask
		color=torch.where(mask_refined>0,color,torch.ones_like(color))
		gt_depth=gt_depth*mask_refined_1D[np.newaxis,...]


		output['color'] = color

		output['mask'] = mask_refined

		output['K'] = self.to_tensor(np.asarray(K,dtype='f'))

		output['R'] = self.to_tensor(np.asarray(R,dtype='f'))

		output['C'] = self.to_tensor(np.asarray(C[...,np.newaxis],dtype='f'))

		output['gt_depth'] = gt_depth

		output['origin'] = self.to_tensor(np.asarray(origin[np.newaxis,...],dtype='f'))

		output['scaling'] = self.to_tensor(scaling[np.newaxis,...])

		return output


	def __len__(self):
		return self.data_len



class HUMBIDataset(Dataset):
	def __init__(self, usage='train', pickle_file='./data/HUMBI_dvr.pkl',skip_every_n_image=1):

		self.to_tensor=transforms.ToTensor()

		with open(pickle_file, 'rb') as f:
			self.data_info = pickle.load(f)[usage]
			self.idx = [i for i in range(0, len(self.data_info[0][0]), skip_every_n_image)]
			self.data_len = len(self.idx)


	def __getitem__(self, index):
		output = {}

		color_1 = self.to_tensor(Image.open(self.data_info[0][0][index]))
		color_2 = self.to_tensor(Image.open(self.data_info[1][0][index]))

		dp_1 = self.to_tensor(Image.open(self.data_info[0][1][index]))
		dp_2 = self.to_tensor(Image.open(self.data_info[1][1][index]))	

		mask_1 =self.to_tensor(Image.open(self.data_info[0][2][index]))
		mask_2 = self.to_tensor(Image.open(self.data_info[1][2][index]))

		mask_1=torch.where(mask_1[-1,...]>0, torch.ones_like(mask_1[-1,...]),torch.zeros_like(mask_1[-1,...]))
		mask_2=torch.where(mask_2[-1,...]>0, torch.ones_like(mask_2[-1,...]),torch.zeros_like(mask_2[-1,...]))
		mask_1=mask_1[np.newaxis,...]
		mask_2=mask_2[np.newaxis,...]

		color_1 = color_1*mask_1
		color_2 = color_2*mask_2

		dp_1 = dp_1*mask_1
		dp_2 = dp_2*mask_2

		K = np.loadtxt(self.data_info[0][4][index],delimiter=',')
		R = np.loadtxt(self.data_info[0][5][index],delimiter=',')
		C = np.loadtxt(self.data_info[0][6][index],delimiter=',')

		iuv_1 = self.data_info[1][11][index]
		iuv_2 = self.data_info[1][12][index]

		origin_1 = self.data_info[0][13][index]
		origin_2 = self.data_info[1][13][index]		

		scaling_1 = 2*np.asarray([self.data_info[0][14][index]],dtype='f')
		scaling_2 = 2*np.asarray([self.data_info[1][14][index]],dtype='f')		

		output['color_1'] = color_1
		output['color_2'] = color_2

		output['dp_1'] = dp_1
		output['dp_2'] = dp_2

		output['mask_1'] = mask_1
		output['mask_2'] = mask_2		

		output['K'] = self.to_tensor(np.asarray(K,dtype='f'))	

		output['R'] = self.to_tensor(np.asarray(R,dtype='f'))	

		output['C'] = self.to_tensor(np.asarray(C[...,np.newaxis],dtype='f'))		

		output['origin_1'] = self.to_tensor(np.asarray(origin_1[np.newaxis,...],dtype='f'))
		output['origin_2'] = self.to_tensor(np.asarray(origin_2[np.newaxis,...],dtype='f'))		

		output['scaling_1'] = self.to_tensor(scaling_1[np.newaxis,...])
		output['scaling_2'] = self.to_tensor(scaling_2[np.newaxis,...])

		output['iuv_1'] = self.to_tensor(np.asarray(iuv_1,dtype='f'))
		output['iuv_2'] = self.to_tensor(np.asarray(iuv_2,dtype='f'))

		return output


	def __len__(self):
		return self.data_len