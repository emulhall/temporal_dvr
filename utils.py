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
from geometry import camera_to_world


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



def sample(im,batch_size):
	#Slightly modified arange_pixels from https://github.com/autonomousvision/differentiable_volumetric_rendering
	h=im.shape[2]
	w=im.shape[3]

	image_range=(-1,1)

	n_points=h*w
	pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
	pixel_locations = torch.stack([pixel_locations[0], pixel_locations[1]],dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
	pixel_scaled = pixel_locations.clone().float()

	pixel_scaled[...,0] = 2*pixel_scaled[...,0]/(w-1)-1
	pixel_scaled[...,1] = 2*pixel_scaled[...,1]/(h-1)-1

	return pixel_locations, pixel_scaled


def patch_sample(im, batch_size, n_points, patch_size=1):
	#Slightly modified patch_sample from https://github.com/autonomousvision/differentiable_volumetric_rendering
	h_step=1/im.shape[2]
	w_step=1/im.shape[3]

	n_patches = int(n_points/patch_size**2)

	p_x = torch.randint(0, im.shape[3], size=(batch_size, n_patches,1)).float() /(im.shape[3]-1)
	p_y=torch.randint(0, im.shape[2], size=(batch_size, n_patches,1)).float() /(im.shape[2]-1)
	p = torch.cat((p_x, p_y),dim=-1)

	p[...,0] *= 1-(patch_size-1)*w_step
	p[...,1] *= 1-(patch_size-1)*h_step

	patch_arange=torch.arange(patch_size)
	x_offset, y_offset = torch.meshgrid(patch_arange, patch_arange)
	offsets = torch.stack((x_offset.reshape(-1), y_offset.reshape(-1)),dim=1).view(1,1,-1,2).repeat(batch_size,n_patches,1,1).float()

	offsets[...,0]*=w_step
	offsets[...,1]*=h_step

	p = p.view(batch_size, n_patches,1,2)+offsets

	p = p*2-1

	p = p.view(batch_size, -1,2)
	amax,amin = p.max(), p.min()
	assert(amax<=1 and amin>=-1)

	p_unscaled=p.clone().float()
	p_unscaled[...,0] = (p_unscaled[...,0]+1)*(im.shape[3])/2
	p_unscaled[...,1] = (p_unscaled[...,1]+1)*(im.shape[2])/2

	return p_unscaled, p



def get_freespace_points(p, K, R, C, origin, scaling, depth_range=[0,2.4]):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering

	d_freespace = torch.rand(p.shape[0], p.shape[1]).cuda(non_blocking=True)*depth_range[1]
	d_freespace=d_freespace.unsqueeze(-1) # (B, n_points, 1)
	

	p_freespace = camera_to_world(p, d_freespace, K, R, C, origin, scaling)

	return p_freespace

def get_occupancy_points(p, p_unscaled,K, R, C, origin, scaling, depth_input, depth_range=[0,2.4]):
	#modified from from https://github.com/autonomousvision/differentiable_volumetric_rendering
	d_occupancy = torch.rand(p.shape[0], p.shape[1]).cuda(non_blocking=True)*depth_range[1]
	d_occupancy=d_occupancy.unsqueeze(-1)

	if depth_input is not None:
		d_occupancy = depth_input[torch.arange(depth_input.shape[0]).unsqueeze(-1),:,p_unscaled[...,1].long(),p_unscaled[...,0].long()] 

	p_occupancy = camera_to_world(p, d_occupancy, K, R, C, origin, scaling)

	return p_occupancy


def points_to_world(p, K, R, C, origin, scaling):
	#modified from from https://github.com/autonomousvision/differentiable_volumetric_rendering
	d=torch.ones((p.shape[0],p.shape[1],1)).cuda(non_blocking=True)

	return camera_to_world(p,d,K,R,C,origin,scaling)


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
	batch_size,n_points,_=origin.shape
	device = origin.device

	#Calculate the intersections with the unit cube where <.,.> is the dot product
	# <n,x-p>=<n,origin+d*ray_direction -p_e>=0
	# d = - <n,origin -p_e> / <n,ray_direction>

	#Get poins on plane p_e
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

	#Corect rays intersect exactly 2 times
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
