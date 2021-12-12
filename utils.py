import cv2
import os
import glob
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle
import time
from sklearn.neighbors import NearestNeighbors
from geometry import camera_to_world, world_to_camera
from view_3D import visualize_3D
import math


# **********************************************************************************************
def calculate_scaling_and_origin_rp(bbox, height):
	"""
	Calculates the origin and scaling of an image for the RenderPeople dataset

	Parameters
	----------
	bbox - the bounding box coords: ndarray of shape (4,2)
	height - the height of the final image: int
	"""
	bb1_t=bbox-1
	bbc1_t=bb1_t[2:4,0:3]
	origin = np.multiply([bb1_t[1,0]-bbc1_t[1,0],bb1_t[0,0]-bbc1_t[0,0]],2)
	squareSize = np.maximum(bb1_t[0,1]-bb1_t[0,0]+1,bb1_t[1,1]-bb1_t[1,0]+1)
	scaling = np.multiply(np.true_divide(squareSize,height),2)

	return origin, scaling


# Rescales an image to be a square
def rescale_square(img, size):
	#Create canvas
	output=np.zeros((size,size,img.shape[2]))

	#Get image shape
	h=img.shape[0]
	w=img.shape[1]

	scale_factor=size/max(h,w)

	width=int(w*scale_factor)
	height=int(h*scale_factor)

	if (width>size) or (height>size):
		print("Error: width or height is greater than the goal of the canvas size")

	img=cv2.resize(img, (width, height))

	x_offset = (size-width) // 2
	y_offset = (size-height) // 2

	output[y_offset:y_offset+height, x_offset:x_offset+width,:]=img



def sample(mask):
	batch_size =mask.shape[0]

	h=mask.shape[2]
	w=mask.shape[3]

	batch_size = mask.shape[0]

	valid_points = torch.where(mask[:,0,...]>0)

	pixel_locations = torch.stack([valid_points[2], valid_points[1]],dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)

	#Get a sample of n_points
	n = np.random.choice(pixel_locations.shape[1],size=pixel_locations.shape[1],replace=False)

	p = pixel_locations[:,n,:]

	return p


def sample_n(mask, n_points):
	batch_size =mask.shape[0]

	h=mask.shape[2]
	w=mask.shape[3]

	batch_size = mask.shape[0]

	valid_points = torch.where(mask[:,0,...]>0)

	pixel_locations = torch.stack([valid_points[2], valid_points[1]],dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)

	#Get a sample of n_points
	n = np.random.choice(pixel_locations.shape[1],size=n_points,replace=False)

	p = pixel_locations[:,n,:]

	return p



def get_freespace_points(p, K, R, C, origin, scaling, depth_range=[0.1,5.], depth_img=None, padding=1e-3):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	device=p.device

	batch_size,n_points,_ = p.shape

	d_freespace=None
	if depth_img is not None:
		d_freespace = get_tensor_values(depth_img,p[:,:int(3*n_points/4),:]) - padding
		depth_min = float(torch.min(depth_img[depth_img>0]))
		d_freespace = torch.cat([d_freespace, torch.from_numpy(np.random.choice(np.linspace(0, depth_min-padding, num=int(n_points/10)), size=math.ceil(n_points/4))).view(batch_size,-1,1).to(device)],dim=1)
	else:
		depth_min=depth_range[0]

		#Freespace points are outside of the cube of the object, which is defined by the min and max of the depth or depth range
		d_freespace = torch.from_numpy(np.random.choice(np.linspace(0, depth_min-padding, num=int(n_points/10)), size=n_points)).view(batch_size,-1,1).to(device)

	p_freespace = camera_to_world(p, d_freespace, K, R, C, origin, scaling)

	return p_freespace

def get_occupancy_points(pixels, K, R, C, origin, scaling, depth_img=None, depth_range=[0.1,5.], padding=1.):
	#modified from from https://github.com/autonomousvision/differentiable_volumetric_rendering
	device = pixels.device
	batch_size,n_points,_=pixels.shape

	d_occupancy=None
	if depth_img is not None:
		d_occupancy = get_tensor_values(depth_img,pixels[:,:int(3*n_points/4),:]) + 1e-3
		depth_max = float(torch.max(depth_img[depth_img>0]))
		d_occupancy = torch.cat([d_occupancy, torch.from_numpy(np.random.choice(np.linspace(depth_max, depth_max+padding, num=int(n_points/10)), size=math.ceil(n_points/4))).view(batch_size,-1,1).to(device)],dim=1)
	else:
		depth_max=depth_range[1]

		#Freespace points are outside of the cube of the object, which is defined by the min and max of the depth or depth range
		d_occupancy = torch.from_numpy(np.random.choice(np.linspace(depth_max, depth_max+padding, num=int(n_points/10)), size=n_points)).view(batch_size,-1,1).to(device)

	p_occupancy = camera_to_world(pixels, d_occupancy, K, R, C, origin, scaling)

	return p_occupancy

def intersect_camera_rays_with_unit_cube(pixels, K, R, C, origin, scaling, padding=0.1, eps=1e-6,use_ray_length_as_depth=True):
	batch_size, n_points, _ = pixels.shape

	pixel_world = image_points_to_world(pixels, K, R, C, origin, scaling)

	ray_vector = (pixel_world-C.squeeze(-1))

	p_cube, d_cube, mask_cube = check_ray_intersection_with_unit_cube(C.squeeze(-1).repeat(1,n_points,1), ray_vector, padding=padding, eps=eps)
	if not use_ray_length_as_depth:
		p_cam = world_to_camera(p_cube.view(batch_size,-1,3),K,R,C,origin,scaling).view(batch_size,n_points, -1,3)
		d_cube = p_cam[...,-1]

	return p_cube, d_cube, mask_cube


def points_to_world(p, K, R, C, origin, scaling):
	#modified from from https://github.com/autonomousvision/differentiable_volumetric_rendering
	d=torch.ones((p.shape[0],p.shape[1],1)).cuda(non_blocking=True)

	return camera_to_world(p,d,K,R,C,origin,scaling)

def image_points_to_world(p, K, R, C, origin,scaling):
	batch_size, n_points, dim = p.shape
	assert(dim==2)
	device = p.device

	d_image = torch.ones(batch_size,n_points,1).to(device)

	return camera_to_world(p, d_image, K, R, C, origin, scaling)


def get_logits_from_prob(probs, eps=1e-4):
	#modified from from https://github.com/autonomousvision/differentiable_volumetric_rendering
	probs = np.clip(probs, a_min=eps, a_max=1-eps)
	logits = np.log(probs/ (1-probs))
	return logits


def get_proposal_points_in_unit_cube(origin, ray_direction, padding=0.1, eps=1e-6, n_steps=40):
	#modified from from https://github.com/autonomousvision/differentiable_volumetric_rendering
	batch_size,n_points, _=origin.shape
	device = origin.device

	p_intervals, d_intervals, mask_inside_cube = check_ray_intersection_with_unit_cube(origin, ray_direction, padding, eps)

	d_proposal = d_intervals[...,0].unsqueeze(-1) + torch.linspace(0,1,steps=n_steps).cuda(non_blocking=True).view(1,1,-1)*(d_intervals[...,1]-d_intervals[...,0]).unsqueeze(-1)
	d_proposal = d_proposal.unsqueeze(-1)

	return d_proposal, mask_inside_cube


def check_ray_intersection_with_unit_cube(origin, ray_direction, padding=0.1, eps=1e-6):
	batch_size,n_points,_=ray_direction.shape

	device = origin.device

	#Calculate the intersections with the unit cube where <.,.> is the dot product
	# <n,x-p>=<n,origin+d*ray_direction -p_e>=0
	# d = - <n,origin -p_e> / <n,ray_direction>

	#Get points on plane p_e
	p_distance = 0.5+padding/2
	p_e = torch.ones(batch_size, n_points, 6).to(device)*p_distance
	p_e[...,3:]*=-1

	#Calculate the intersection points
	nominator = p_e - origin.repeat(1,1,2)
	denominator = ray_direction.repeat(1,1,2)
	d_intersect = nominator/denominator
	p_intersect = origin.unsqueeze(-2) + d_intersect.unsqueeze(-1)*ray_direction.unsqueeze(-2)

	#Calculate mask where points intersect unit cube
	p_mask_inside_cube=(
		(p_intersect[...,0] <= p_distance+eps) &
		(p_intersect[...,1] <= p_distance+eps) &
		(p_intersect[...,2] <= p_distance+eps) &
		(p_intersect[...,0] >= -(p_distance+eps)) &
		(p_intersect[...,1] >= -(p_distance+eps)) &
		(p_intersect[...,2] >= -(p_distance+eps))
		).cpu()

	#Correct rays intersect exactly 2 times
	mask_inside_cube = p_mask_inside_cube.sum(-1) == 2

	#Get interval values for valid ps
	p_intervals = p_intersect[mask_inside_cube][p_mask_inside_cube[mask_inside_cube]].view(-1,2,3)
	p_intervals_batch = torch.zeros(batch_size, n_points,2,3).to(device)
	p_intervals_batch[mask_inside_cube] = p_intervals

	#Calculate ray lengths for the interval points
	d_intervals_batch = torch.zeros(batch_size,n_points, 2).to(device)
	norm_ray = torch.norm(ray_direction[mask_inside_cube],dim=-1)
	d_intervals_batch[mask_inside_cube] = torch.stack([
		torch.norm(p_intervals[:,0]-origin[mask_inside_cube],dim=-1)/norm_ray,
		torch.norm(p_intervals[:,1]-origin[mask_inside_cube],dim=-1)/norm_ray],dim=-1)

	#Sort the ray lengths
	d_intervals_batch, indices_sort = d_intervals_batch.sort()
	p_intervals_batch = p_intervals_batch[
		torch.arange(batch_size).view(-1,1,1),
		torch.arange(n_points).view(1,-1,1),
		indices_sort]

	return p_intervals_batch, d_intervals_batch, mask_inside_cube


def normalize_tensor(tensor, min_norm=1e-5, feat_dim=-1):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	norm_tensor = torch.clamp(torch.norm(tensor,dim=feat_dim,keepdim=True),min=min_norm)
	normed_tensor = tensor/norm_tensor

	return normed_tensor

def to_pytorch(tensor, return_type=False):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	is_numpy=False
	if type(tensor) == np.ndarray:
		tensor = torch.from_numpy(tensor)
		is_numpy = True
	tensor = tensor.clone()
	if return_type:
		return tensor, is_numpy
	return tensor

def get_mask(tensor):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	tensor, is_numpy = to_pytorch(tensor, True)
	mask = ((abs(tensor)!=np.inf) & (torch.isnan(tensor)==False))
	mask = mask.bool()
	if is_numpy:
		mask = mask.numpy()
	return mask

def get_tensor_values(tensor,p,with_mask=False, squeeze_channel_dim=False):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	p=to_pytorch(p)
	tensor, is_numpy = to_pytorch(tensor,True)
	batch_size,_,_,_=tensor.shape

	p=p.long()
	values=tensor[torch.arange(batch_size).unsqueeze(-1),:,p[...,1],p[...,0]]

	if with_mask:
		mask = get_mask(values)
		if squeeze_channel_dim:
			mask = mask.squeeze(-1)
		if is_numpy:
			mask = mask.numpy()

	if squeeze_channel_dim:
		values = values.squeeze(-1)

	if is_numpy:
		values=values.numpy()

	if with_mask:
		return values, mask	

	return values



def normalize_imagenet(x):
	x = x.clone()
	x[:, 0] = (x[:, 0] - 0.485) / 0.229
	x[:, 1] = (x[:, 1] - 0.456) / 0.224
	x[:, 2] = (x[:, 2] - 0.406) / 0.225
	return x