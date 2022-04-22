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
#from view_3D import visualize_3D
import math
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from tqdm import tqdm
from torch import distributions as dist



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

	pixel_locations = torch.stack([valid_points[2], valid_points[1]],dim=-1).long().unsqueeze(0).repeat(batch_size, 1, 1)

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

	pixel_locations = torch.stack([valid_points[2], valid_points[1]],dim=-1).long().unsqueeze(0).repeat(batch_size, 1, 1)

	#Get a sample of n_points
	n = np.random.choice(pixel_locations.shape[1],size=min(n_points,pixel_locations.shape[1]),replace=False)

	p = pixel_locations[:,n,:]

	return p

def sample_correspondences(iuv_1, iuv_2, n_points):
	batch_size = iuv_1.shape[0]
	device = iuv_1.device

	iuv_valid=torch.where(iuv_1[...,0]!=-1)

	u_1=iuv_1[iuv_valid[0], iuv_valid[1],iuv_valid[2],:].repeat(batch_size,1,1) # (b,N,2)
	u_2=iuv_2[iuv_valid[0], iuv_valid[1],iuv_valid[2],:].repeat(batch_size,1,1) # (b,N,2)

	n = np.random.choice(u_1.shape[1],size=min(n_points,u_1.shape[1]),replace=False)

	u_1 = u_1[:,n,:]
	u_2 = u_2[:,n,:]

	return u_1, u_2


def get_freespace_points(p, K, R, C, origin, scaling, depth_range=[2.5,6.75], depth_img=None, scale=1):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	device=p.device

	batch_size,n_points,_ = p.shape
	padding = 1e-3

	#n_points=int(n_points/2)

	d_freespace=None
	if depth_img is not None:
		d_freespace =get_tensor_values(depth_img,p[:,:n_points,:])
		d_freespace -= torch.from_numpy(np.abs(np.random.normal(scale=scale, size=d_freespace.shape))).to(device)
	else:
		depth_min=depth_range[0]

		#Freespace points are outside of the cube of the object, which is defined by the min and max of the depth or depth range
		d_freespace = torch.from_numpy(np.random.choice(np.linspace(depth_min, max(depth_min-padding,0), num=math.ceil(n_points/10)), size=n_points)).view(batch_size,-1,1).to(device)

	p_freespace = camera_to_world(p, d_freespace, K, R, C, origin, scaling)


	return p_freespace

def get_occupancy_points(pixels, K, R, C, origin, scaling, depth_img=None, depth_range=[2.5,6.75], scale=1):
	#modified from from https://github.com/autonomousvision/differentiable_volumetric_rendering
	device = pixels.device
	batch_size,n_points,_=pixels.shape
	padding = 1e-3

	d_occupancy=None
	if depth_img is not None:
		d_occupancy = get_tensor_values(depth_img,pixels[:,:n_points,:])
		d_occupancy += torch.from_numpy(np.abs(np.random.normal(scale=scale, size=d_occupancy.shape))).to(device)
	else:
		depth_max=depth_range[1]

		#Occupancy points are inside of the cube of the object, which is defined by the min and max of the depth or depth range
		d_occupancy = torch.from_numpy(np.random.choice(np.linspace(depth_max, depth_max+padding, num=math.ceil(n_points/10)), size=n_points)).view(batch_size,-1,1).to(device)

	p_occupancy = camera_to_world(pixels, d_occupancy, K, R, C, origin, scaling) 

	return p_occupancy

def get_mask_points(mask, K, R, C, origin, scaling, n_points,bb_min,bb_max,depth_range=[2.5,6.75]):
	device=mask.device
	batch_size=mask.shape[0]

	mult=20

	d_mask = torch.from_numpy(np.random.choice(np.linspace(depth_range[0], depth_range[1], num=math.ceil(n_points/2)), size=n_points*mult)).view(batch_size,-1,1).to(device)

	h=mask.shape[2]
	w=mask.shape[3]

	batch_size = mask.shape[0]

	invalid_points = torch.where(mask[:,0,...]==0)

	pixel_locations = torch.stack([invalid_points[2], invalid_points[1]],dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)

	#Get a sample of n_points
	n = np.random.choice(pixel_locations.shape[1],size=n_points*mult)

	p = pixel_locations[:,n,:]

	p_mask = camera_to_world(p, d_mask, K, R, C, origin, scaling)

	X=bb_max[0]
	Y=bb_max[1]
	Z=bb_max[0]
	x=bb_min[0]
	y=bb_min[1]
	z=bb_min[2]

	points = np.asarray([[x,y,z],[X,y,z],[x,Y,z],[X,Y,z],[x,y,Z],
		[X,y,Z],[x,Y,Z],[X,Y,Z]])

	inside=np.argwhere(Delaunay(points).find_simplex(p_mask.cpu())>=0)

	p_mask = p_mask[:,inside[:,1],:]
	p_mask = p_mask[:,:n_points,:]

	return p_mask



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


def reshape_multiview_tensors(tensor):
	tensor = tensor.view(
		tensor.shape[0]*tensor.shape[1],
		tensor.shape[2],
		tensor.shape[3],
		tensor.shape[4])

	return tensor


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
	batch_size,_,h,w=tensor.shape
	values = torch.zeros_like(p)

	p=p.long()
	in_tensor = (p[...,0]>=0) & (p[...,0]<w) & (p[...,1]>=0) & (p[...,1]<h)
	#print(torch.unique(in_tensor))
	p=in_tensor[...,np.newaxis]*p

	values=tensor[torch.arange(batch_size).unsqueeze(-1),:,p[...,1],p[...,0]]
	values=in_tensor[...,np.newaxis]*values
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

def save_3D(X, mask,fname):
	x=X[0,...]
	y=X[1,...]
	z=X[2,...]

	np.savetxt(fname+'_x.txt',x)
	np.savetxt(fname+'_y.txt',y)
	np.savetxt(fname+'_z.txt',z)
	np.savetxt(fname+'_mask.txt',mask)


def get_dp_correspondences(dp1,dp2,mask1,mask2,num_matches=15,threshold=5,visualize=False):
	"""
	Gets correspondences between two DensePose images
	We choose the 15 correspondences with the lowest u,v distance
	For later computation, missing parts or correspondences are filled with -1s

	Parameters
	----------
	dp1 - IUV array 1: ndarray of shape (256,256,3)
	dp2 - IUV array 2: ndarray of shape (256,256,3)

	Returns
	-------
	x1 - Array of correspondences from dp1: ndarray of shape (360,2)
	x2 - Array of correspondences from dp2: ndarray of shape (360,2)
	"""


	final_corr_1=[]
	final_corr_2=[]

	if np.max(dp1)<=1.0:
		mask1=np.where(mask1>0,255.0,0.0)
		mask2=np.where(mask2>0,255.0,0.0)
	else:
		mask1=np.where(mask1>0,1.0,0.0)
		mask2=np.where(mask2>0,1.0,0.0)


	dp1=dp1*mask1
	dp2=dp2*mask2

	dp1=np.floor(dp1)
	dp2=np.floor(dp2)

	#Get the coordinates of valid points
	loc1=np.argwhere(dp1[:,:,2]>0) # N1 x 3
	loc2=np.argwhere(dp2[:,:,2]>0) # N2 x 3

	#Narrow down descriptors to only get valid descriptors
	des1=dp1[loc1[:,0], loc1[:,1], :] # N1 x 3
	des2=dp2[loc2[:,0], loc2[:,1], :] # N2 x 3

	#For each segment we need to find u,v correspondences
	#There are 25 possible segments
	if visualize:
		plt.imshow(dp1[:,:,2]),plt.show()
		plt.imshow(dp1[:,:,0]), plt.show()
		plt.imshow(dp1[:,:,1]), plt.show()
	warning=True

	#Let's check to see if these are viewed from the same side (front vs. back)
	#Get front:
	'''des1_f_ind=np.argwhere(des1[:,2]==2).flatten() # M1
	loc1_f=loc1[des1_f_ind,:] #M1 x 2

	des2_f_ind=np.argwhere(des2[:,2]==2).flatten() # M2
	loc2_f=loc2[des2_f_ind,:]


	#Get back:
	des1_b_ind=np.argwhere(des1[:,2]==1).flatten() # M1
	loc1_b=loc1[des1_b_ind,:] #M1 x 2

	des2_b_ind=np.argwhere(des2[:,2]==1).flatten() # M2
	loc2_b=loc2[des2_b_ind,:]

	front_dom_1=loc1_f.shape[0] > loc1_b.shape[0]
	front_dom_2=loc2_f.shape[0] > loc2_b.shape[0]

	if front_dom_1 != front_dom_2 and (np.abs(loc1_f.shape[0]-loc2_f.shape[0])>np.abs(loc1_f.shape[0]-loc1_b.shape[0])) and (np.abs(loc2_f.shape[0]-loc1_f.shape[0])>np.abs(loc2_f.shape[0]-loc2_b.shape[0])):
		final_corr_1=np.asarray(final_corr_1,dtype=np.int32)
		final_corr_2=np.asarray(final_corr_2,dtype=np.int32)		
		print("There have been no matches found between these two images.")
		return final_corr_1, final_corr_2, warning'''

	for i in range(1,26):
		#Filter kps and des by i
		des1_i_ind=np.argwhere(des1[:,2]==i).flatten() # M1
		des1_i=des1[des1_i_ind,:] # M1 x 3
		loc1_i=loc1[des1_i_ind,:] #M1 x 2

		des2_i_ind=np.argwhere(des2[:,2]==i).flatten() # M2
		des2_i=des2[des2_i_ind,:] # M2 x 2
		loc2_i=loc2[des2_i_ind,:]

		#If we are missing the body part from one of the frames then we can simply add a block of [-1,-1,-1] and move on
		if(len(des1_i_ind)==0 or len(des2_i_ind)==0):
			final_corr_1.extend(np.ones((num_matches,2))*-1)
			final_corr_2.extend(np.ones((num_matches,2))*-1)
			continue

		#Remove outliers
		#IQR Method
		loc1_q3, loc1_q1 = np.percentile(loc1_i, [75,25],axis=0)
		iqr=loc1_q3-loc1_q1

		not_too_small=loc1_i > loc1_q1[np.newaxis,:]-1.5*iqr[np.newaxis,:]
		loc1_i=loc1_i[not_too_small[:,0]*not_too_small[:,1],:]
		des1_i=des1_i[not_too_small[:,0]*not_too_small[:,1],:]

		not_too_big=loc1_i < loc1_q3+1.5*iqr
		loc1_i=loc1_i[not_too_big[:,0]*not_too_big[:,1],:]
		des1_i=des1_i[not_too_big[:,0]*not_too_big[:,1],:]

		loc2_q3, loc2_q1 = np.percentile(loc2_i, [75,25],axis=0)
		iqr=loc2_q3-loc2_q1

		not_too_small=loc2_i > loc2_q1[np.newaxis,:]-1.5*iqr[np.newaxis,:]
		loc2_i=loc2_i[not_too_small[:,0]*not_too_small[:,1],:]
		des2_i=des2_i[not_too_small[:,0]*not_too_small[:,1],:]

		not_too_big=loc2_i < loc2_q3[np.newaxis,:]+1.5*iqr[np.newaxis,:]
		loc2_i=loc2_i[not_too_big[:,0]*not_too_big[:,1],:]
		des2_i=des2_i[not_too_big[:,0]*not_too_big[:,1],:]

		#Let's check to make sure we have more than 0 samples
		if loc1_i.shape[0]<1 or loc2_i.shape[0]<1:			
			final_corr_1.extend(np.ones((num_matches,2))*-1)
			final_corr_2.extend(np.ones((num_matches,2))*-1)
			continue

		#Std Dev Method
		loc1_mean=np.mean(loc1_i,axis=0)
		loc1_stddev=np.std(loc1_i,axis=0)
		loc1_dist_from_mean = abs(loc1_i-loc1_mean[np.newaxis,:])
		max_dev=2
		loc1_not_outlier = loc1_dist_from_mean < max_dev*loc1_stddev[np.newaxis,:]
		loc1_i=loc1_i[loc1_not_outlier[:,0]*loc1_not_outlier[:,1],:]
		des1_i=des1_i[loc1_not_outlier[:,0]*loc1_not_outlier[:,1],:]

		loc2_mean=np.mean(loc2_i)
		loc2_stddev=np.std(loc2_i)
		loc2_dist_from_mean = abs(loc2_i-loc2_mean)
		max_dev=2
		loc2_not_outlier = loc2_dist_from_mean < max_dev*loc2_stddev
		loc2_i=loc2_i[loc2_not_outlier[:,0],:]
		des2_i=des2_i[loc2_not_outlier[:,0],:]


		#Let's check to make sure we have more than 0 samples
		if loc1_i.shape[0]<1 or loc2_i.shape[0]<1:
			final_corr_1.extend(np.ones((num_matches,2))*-1)
			final_corr_2.extend(np.ones((num_matches,2))*-1)
			continue

		des1_uv=des1_i
		des2_uv=des2_i
		matches=[]

		nbrs=NearestNeighbors(n_neighbors=1).fit(des2_uv)
		distances, indices=nbrs.kneighbors(des1_uv)

		for j in range(indices.shape[0]):
			if distances[j]<threshold:
				matches.append([distances[j][0],j,indices[j][0]])

		matches=np.asarray(matches)
		if len(matches)==0:
			final_corr_1.extend(np.ones((num_matches,2))*-1)
			final_corr_2.extend(np.ones((num_matches,2))*-1)
			continue
		else:
			ind=np.argsort(matches[:,0])
			matches=matches[ind]
			warning=False

		to_add=np.asarray(matches[:num_matches],dtype=np.int32)
		corr_1_to_add=loc1_i[to_add[:,1],:]
		corr_2_to_add=loc2_i[to_add[:,2],:]

		#Technically loc_1 is in v,u order, and so we need to flip that
		final_corr_1.extend(np.concatenate((np.reshape(corr_1_to_add[:,1],(-1,1)),np.reshape(corr_1_to_add[:,0], (-1,1))),axis=1))
		final_corr_2.extend(np.concatenate((np.reshape(corr_2_to_add[:,1],(-1,1)),np.reshape(corr_2_to_add[:,0], (-1,1))),axis=1))

		if len(to_add)<num_matches:
			final_corr_1.extend(np.ones((num_matches-len(to_add),2))*-1)
			final_corr_2.extend(np.ones((num_matches-len(to_add),2))*-1)


		if visualize:
			corr_1_to_add=np.asarray(np.concatenate((np.reshape(corr_1_to_add[:,1],(-1,1)),np.reshape(corr_1_to_add[:,0], (-1,1))),axis=1))
			corr_2_to_add=np.asarray(np.concatenate((np.reshape(corr_2_to_add[:,1],(-1,1)),np.reshape(corr_2_to_add[:,0], (-1,1))),axis=1))
			visualize_correspondences(corr_1_to_add[np.newaxis,...],corr_2_to_add[np.newaxis,...],dp1/255.0,dp2/255.0)

	final_corr_1=np.asarray(final_corr_1,dtype=np.int32)
	final_corr_2=np.asarray(final_corr_2,dtype=np.int32)


	if (warning):
		print("There have been no matches found between these two images.")


	return final_corr_1, final_corr_2,warning


def visualize_correspondences(iuv_1, iuv_2,dp_1,dp_2,show=True, save=False, fname='corr.png'):
	"""
		Visualize the iuv correspondences

		Parameters
		---------
		iuv1, iuv2 - iuv correspondences: ndarray of shape (1,360,2)
		dp1, dp2 - DensePose images: ndarray of shape (H,W,3)

	"""

	#Get only valid iuv parameters
	iuv_valid=np.where(iuv_1!=-1)
	iuv_1=iuv_1[iuv_valid[0],iuv_valid[1],:] # N,2
	iuv_2=iuv_2[iuv_valid[0],iuv_valid[1],:] # N,2

	canvas=np.zeros((dp_1.shape[0],dp_1.shape[1]*2,3))
	canvas[:,0:dp_1.shape[1],:]=dp_1
	canvas[:,dp_1.shape[1]:,:]=dp_2

	for i in range(len(iuv_1)):
		plt.plot([iuv_1[i,0],iuv_2[i,0]+dp_1.shape[1]],[iuv_1[i,1],iuv_2[i,1]])


	if show:
		plt.imshow(canvas),plt.show()

	else:
		plt.imsave(fname,canvas)

def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt

def calc_error(model, device, dataset, num_tests):
	if num_tests>len(dataset):
		num_tests=len(dataset)

	with torch.no_grad():
		IOU_arr, prec_arr, recall_arr = [], [], []

		for idx in tqdm(range(num_tests)):
			data = dataset[idx * len(dataset) // num_tests]

			#Retrieve data
			color = data['color'].to(device)
			mask = data['mask'].to(device)
			K = data['K'].to(device) 
			R = data['R'].to(device) 
			C = data['C'].to(device)
			origin = data['origin'].to(device)
			scaling = data['scaling'].to(device)
			samples = data['samples'].to(device).squeeze(1)
			labels = data['labels'].to(device).squeeze(1)

			c = model.encode_inputs(color)

			pred = model.decode(samples.float(),c=c).probs
			#pred = dist.Bernoulli(logits=torch.tensor(logits)).probs

			IOU, prec, recall = compute_acc(pred, labels.squeeze(2))
			IOU_arr.append(IOU.item())
			prec_arr.append(prec.item())
			recall_arr.append(recall.item())

	return (np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr))




