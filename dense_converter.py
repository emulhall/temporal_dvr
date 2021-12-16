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

from utils import rescale_square

#To run densepose:
#python apply_net.py dump configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml model_final_844d15.pkl "/media/mulha024/h/train_512_RenderPeople_all_sparse/9/color/*.png" --output /media/mulha024/h/train_512_RenderPeople_all_sparse/9/dump.pkl -v

def mask_matrix_resize(mask, matrix, w, h):
	if (w != mask.shape[1]) or (h != mask.shape[0]):
		mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST)
	if (w != matrix.shape[2]) or (h != matrix.shape[1]):
		matrix_v = cv2.resize(matrix[0,:,:], (w, h), cv2.INTER_LINEAR)
		matrix_u = cv2.resize(matrix[1,:,:], (w, h), cv2.INTER_LINEAR)
		matrix_i = cv2.resize(matrix[2,:,:], (w, h), cv2.INTER_LINEAR)

		matrix=np.concatenate((matrix_v[...,np.newaxis], matrix_u[...,np.newaxis], matrix_i[...,np.newaxis]), axis=2)

	elif (w==matrix.shape[2]) and (h==matrix.shape[1]):
		#Guarantee that we permute to be h,w,c instead of c,h,w
		matrix=np.concatenate((np.reshape(matrix[0,:,:], (h, w,1)), np.reshape(matrix[1,:,:], (h,w,1)), np.reshape(matrix[2,:,:], (h, w,1))), axis=2)
	return mask, matrix

def visualize(image_bgr, mask, matrix, bbox_xyxy, inplace):
	image_target_bgr=image_bgr
	if inplace:
		image_target_bgr=image_target_bgr*0
	
	x1,y1,x2,y2=[int(v) for v in bbox_xyxy]
	w=x2-x1
	h=y2-y1
	if w<=0 or h<=0:
		return image_bgr
	mask, matrix=mask_matrix_resize(mask, matrix, w, h)

	mask_bg=np.tile((mask==0)[:,:,np.newaxis], [1,1,3])

	matrix_vis=matrix.clip(0,255).astype(np.uint8)
	matrix_vis[mask_bg]=image_target_bgr[y1 : y2, x1 : x2, :][mask_bg]
	image_target_bgr[y1:y2, x1:x2]=matrix_vis

	return image_target_bgr.astype(np.uint8)

def vui(results):
	uv=results.uv
	labels=results.labels
	vui_array=torch.cat((torch.reshape(uv[1,:,:], (1, uv.shape[1], uv.shape[2]))*255, 
		torch.reshape(uv[0,:,:], (1, uv.shape[1], uv.shape[2]))*255, 
		torch.reshape(labels.type(torch.float32), (1, labels.shape[0], labels.shape[1])))).type(torch.uint8).cpu()

	return vui_array.numpy()

def convert_all_rp():
	sys.path.append('../detectron2/projects/DensePose')
	cameras=natsorted(glob.glob(os.path.join('/media/mulha024/i/train_512_RenderPeople_all_sparse','*')))

	for c in cameras[67:]:
		print("Converting images in " + c)
		if c=='/media/mulha024/i/train_512_RenderPeople_all_sparse/56' or c=='/media/mulha024/i/train_512_RenderPeople_all_sparse/83':
			continue
		f=open(os.path.join(c, 'dump.pkl'), 'rb')
		data=pickle.load(f)

		for im in range(len(data)):
			orig_path=data[im]['file_name']
			image_bgr=cv2.imread(orig_path.replace('/h/','/i/'))
			dp=image_bgr

			removebg_mask=cv2.imread(orig_path.replace('color','complete_mask').replace('/h/','/i/'))
			removebg_mask=np.where(removebg_mask>100,np.ones_like(removebg_mask),np.zeros_like(removebg_mask))

			#Get subject number
			head, tail = os.path.split(orig_path)
			sub_num=tail[:-4]

			#Get results
			results=data[im]['pred_densepose']

			#Go through the results to get the largest bounding box - this is hopefully the correct one
			index=0
			biggest_area=-100
			indices={}
			for r in range(len(results)):
				#Get bounding box
				temp_box=data[im]['pred_boxes_XYXY'][r,:]
				x1,y1,x2,y2=[int(v) for v in temp_box]

				#Calculate box area
				area=(x2-x1)*(y2-y1)

				#Compare and save biggest area
				if area>biggest_area:
					biggest_area=area
					index=r
				indices[r]=area

			#Get vui image
			matrix=vui(results[index])

			#Get segmentation
			seg=matrix[-1,:,:]

			#Get box
			bbox_xyxy=data[im]['pred_boxes_XYXY'][index,:]

			#Create mask
			mask=np.zeros(seg.shape, dtype=np.uint8)
			mask[seg>0]=1


			#Visualize
			dp=visualize(dp, mask, matrix, bbox_xyxy, True)

			indices.pop(index)
			for ind in indices:
				matrix=vui(results[ind])

				#Get segmentation
				seg=matrix[-1,:,:]

				#Get box
				bbox_xyxy=data[im]['pred_boxes_XYXY'][ind,:]

				#Create mask
				mask=np.zeros(seg.shape, dtype=np.uint8)
				mask[seg>0]=1


				#Visualize
				dp=visualize(dp, mask, matrix, bbox_xyxy, False)


			#Multiply final visualization by the remove_bg mask to guarantee that extra subjects are removed
			dp=dp*removebg_mask
			#Save the visualization
			if not(os.path.isdir(os.path.join(c, 'densepose'))):
				os.mkdir(os.path.join(c, 'densepose'))

			#Has to be png, jpg changes values	
			cv2.imwrite(os.path.join(c, 'densepose', str(sub_num)+'.png'), cv2.cvtColor(dp, cv2.COLOR_BGR2RGB))


def convert_all_full_size(goal_size):
	sys.path.append('../detectron2/projects/DensePose')
	#Get all the different time stamps
	subjects=natsorted(glob.glob('/media/mulha024/i/Body_1_80_update/subject_*/body'))
	for s in subjects:
		f=open(os.path.join(s, 'dump.pkl'), 'rb')
		data=pickle.load(f)
		for im in range(len(data)):
			orig_path = data[im]['file_name']
			img = cv2.imread(orig_path)

			head, tail = os.path.split(orig_path)

			cam_num=tail[:-4]

			image_bgr=img
			dp=image_bgr

			#Get results
			results = data[im]['pred_densepose']

			#Go through the results to get the largest bounding box - this is hopefully the correct one
			index=0
			biggest_area=-100
			for r in range(len(results)):
				#Get bounding box
				temp_box=data[im]['pred_boxes_XYXY'][r,:]
				x1,y1,x2,y2=[int(v) for v in temp_box]

				#Calculate box area
				area=(x2-x1)*(y2-y1)

				#Compare and save biggest area
				if area>biggest_area:
					biggest_area=area
					index=r

			#Get vui image
			matrix=vui(results[index])

			#Get segmentation
			seg=matrix[-1,:,:]

			#Get box
			bbox_xyxy=data[im]['pred_boxes_XYXY'][index,:]

			#Create mask
			mask=np.zeros(seg.shape, dtype=np.uint8)
			mask[seg>0]=1

			#Visualize
			dp=visualize(dp, mask, matrix, bbox_xyxy, True)

			#Now, let's crop it to the desired size
			x1,y1,x2,y2=[int(v) for v in bbox_xyxy]
			w=x2-x1
			h=y2-y1

			#Crop to the bounding box
			dp=dp[int(y1):int(y2),int(x1):int(x2),:]
			img=img[int(y1):int(y2),int(x1):int(x2),:]

			output_dp=np.zeros((goal_size,goal_size,3),dtype=np.uint8)
			output_image=np.zeros((goal_size,goal_size,3),dtype=np.uint8)

			scale_factor=goal_size/max(h,w)

			width=int(w*scale_factor)
			height=int(h*scale_factor)

			resized_dp=cv2.resize(dp, (width,height))
			resized_img=cv2.resize(img, (width,height))

			x_offset = (goal_size-width) // 2
			y_offset = (goal_size-height) // 2

			output_dp[y_offset:y_offset+height, x_offset:x_offset+width,:]=resized_dp
			output_image[y_offset:y_offset+height, x_offset:x_offset+width,:]=resized_img

			scaling=(1/scale_factor)
			origin=np.asarray([x1,y1])-(1/scale_factor)*np.asarray([x_offset,y_offset])

			scaling_and_origin=np.zeros((2,2))
			scaling_and_origin[0,0]=scaling
			scaling_and_origin[1,:]=origin

			#Save the visualization
			temp = s.replace('Body_1_80_update', 'HUMBI_1_80_256')
			head, cam_num = os.path.split(orig_path)
			cam_num=cam_num[5:-4]
			head,_= os.path.split(head)
			_,t = os.path.split(head)
			head,_=os.path.split(temp)
			_,subject_num = os.path.split(head)

			if not(os.path.isdir('/media/mulha024/i/HUMBI_1_80_256')):
				os.mkdir(os.path.join('/media/mulha024/i','HUMBI_1_80_256'))

			output_path = os.path.join('/media/mulha024/i/HUMBI_1_80_256',t)

			if not(os.path.isdir(output_path)):
				os.mkdir(output_path)								

			if not(os.path.isdir(os.path.join(output_path, 'densepose_256'))):
				os.mkdir(os.path.join(output_path, 'densepose_256'))

			if not(os.path.isdir(os.path.join(output_path, 'image_256'))):
				os.mkdir(os.path.join(output_path, 'image_256'))

			if not(os.path.isdir(os.path.join(output_path, 'scaling_and_origin_256'))):
				os.mkdir(os.path.join(output_path, 'scaling_and_origin_256'))

			if not(os.path.isdir(os.path.join(output_path, 'mask_256'))):
				os.mkdir(os.path.join(output_path, 'mask_256'))

			cv2.imwrite(os.path.join(output_path, 'densepose_256', subject_num+'.png'), cv2.cvtColor(output_dp, cv2.COLOR_BGR2RGB))
			cv2.imwrite(os.path.join(output_path, 'image_256', subject_num+'.png'), output_image)
			np.savetxt(os.path.join(output_path, 'scaling_and_origin_256',subject_num+'.txt'),scaling_and_origin)			



if __name__ == '__main__':
	convert_all_full_size(256)