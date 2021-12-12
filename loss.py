import torch
from torch.nn import functional as F
from utils import get_tensor_values
from geometry import camera_to_world, world_to_camera

def calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, img,pixels, reduction_method, loss):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	if lambda_rgb!=0 and mask_rgb.sum() >0:
		batch_size,n_points,_=rgb_pred.shape
		loss_rgb_eval = torch.tensor(3)
		rgb_gt = get_tensor_values(img,pixels)

		#RGB loss
		loss_rgb = l1_loss(rgb_pred[mask_rgb],rgb_gt[mask_rgb],reduction_method)*lambda_rgb/batch_size

		loss['loss']+=loss_rgb
		loss['loss_rgb']+=loss_rgb


def l1_loss(val_gt, val_pred, reduction_method='sum',eps=0.,sigma_pow=1,feat_dim=True):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	assert(val_pred.shape==val_gt.shape)
	loss_out = (val_gt-val_pred).abs()
	loss_out = (loss_out +eps).pow(sigma_pow)
	if feat_dim:
		loss_out = loss_out.sum(-1)
	return apply_reduction(loss_out, reduction_method)

def apply_reduction(tensor, reduction_method='sum'):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	if reduction_method == 'sum':
		return tensor.sum()
	elif reduction_method == 'mean':
		return tensor.mean()

def image_gradient_loss(val_pred, val_gt, mask, patch_size, reduction_method='sum'):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	assert((val_gt.shape == val_pred.shape)&(patch_size>1)&(val_gt.shape[1]%(patch_size**2)==0))

	val_gt[mask==0] = 0.
	rgb_pred = torch.zeros_like(val_pred)
	rgb_pred[mask] = val_pred[mask]

	#Reshape tensors
	batch_size, n_points,_ = val_gt.shape
	val_gt = val_gt.view(batch_size,-1,patch_size,patch_size,3)
	rgb_pred = rgb_pred.view(batch_size,-1,patch_size,patch_size,3)

	#Get mask where all patch entries are valid
	mask_patch = mask.view(batch_size,-1,patch_size,patch_size).sum(-1).sum(-1) == patch_size*patch_size

	#Calculate gradients
	ddx = val_gt[...,0,0] - val_gt[...,0,1]
	ddy = val_gt[...,0,0] - val_gt[...,1,0]
	ddx_pred = rgb_pred[...,0,0] - rgb_pred[...,0,1]
	ddy_pred = rgb_pred[...,0,0] - rgb_pred[...,1,0]

	#Stack gradients
	ddx, ddy = ddx[mask_patch], ddy[mask_patch]
	grad_gt = torch.stack([ddx,ddy],dim=-1)
	ddx_pred, ddy_pred = ddx_pred[mask_patch], ddy_pred[mask_patch]
	grad_pred = torch.stack([ddx_pred, ddy_pred], dim=-1)

	#Calculate l2 norm on 3D vectors
	loss_out = torch.norm(grad_pred-grad_gt, dim=-1).sum(-1)
	return apply_reduction(loss_out, reduction_method)


def calculate_depth_loss(lambda_depth,mask_depth, depth_img, pixels, K, R, C, origin, scaling, p_world_hat, reduction_method, loss, depth_loss_on_world_points=False):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
	if lambda_depth!=0 and mask_depth.sum()>0:
		batch_size,n_points,_=p_world_hat.shape

		#Check if all values are valid
		depth_gt, mask_gt_depth = get_tensor_values(depth_img,pixels,squeeze_channel_dim=True, with_mask=True)
		mask_depth &=mask_gt_depth
		if depth_loss_on_world_points:
			p_world = camera_to_world(pixels, depth_gt.unsqueeze(-1), K,R,C,origin,scaling)
			loss_depth = l2_loss(p_world_hat[mask_depth], p_world[mask_depth],reduction_method)*lambda_depth/batch_size
		else:
			d_pred = world_to_camera(p_world_hat, K, R, C, origin, scaling)[...,-1]
			loss_depth = l1_loss(d_pred[mask_depth], depth_gt[mask_depth],reduction_method,feat_dim=False)*lambda_depth/batch_size

		loss['loss']+=loss_depth
		loss['loss_depth']+=loss_depth

def l2_loss(val_gt, val_pred, reduction_method='sum'):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering		
	assert(val_gt.shape==val_pred.shape)
	loss_out = torch.norm((val_gt-val_pred),dim=-1)
	return apply_reduction(loss_out,reduction_method)

def calculate_normal_loss(lambda_normal,normals, batch_size, loss):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering		
	if lambda_normal !=0:
		normal_loss = torch.norm(normals[0]-normals[1],dim=-1).sum()*lambda_normal /batch_size
		loss['loss']+=normal_loss
		loss['loss_normal']+=normal_loss


def calculate_freespace_loss(lambda_freespace, logits_hat, reduction_method, loss):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering		
	batch_size=logits_hat.shape[0]
	loss_freepace = freespace_loss(logits_hat, reduction_method=reduction_method)*lambda_freespace/batch_size
	loss['loss']+=loss_freepace
	loss['loss_freespace']+=loss_freepace

def freespace_loss(logits_pred, weights=None, reduction_method='sum'):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering		
	return cross_entropy_occupancy_loss(logits_pred, is_occupied=False, weights=weights,reduction_method=reduction_method)

def cross_entropy_occupancy_loss(logits_pred, is_occupied=True, weights=None, reduction_method='None'):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering

	if is_occupied:
		occ_gt = torch.ones_like(logits_pred)
	else:
		occ_gt = torch.zeros_like(logits_pred)

	loss_out = F.binary_cross_entropy_with_logits(logits_pred,occ_gt,reduction='none')

	if weights is not None:
		assert(loss_out.shape == weights.shape)
		loss_out = loss_out*weights

	return apply_reduction(loss_out, reduction_method)

def calculate_occupancy_loss(lambda_occupancy,logits_hat, reduction_method,loss):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering	
	batch_size=logits_hat.shape[0]

	loss_occupancy = occupancy_loss(logits_hat, reduction_method=reduction_method) * lambda_occupancy/batch_size
	loss['loss']+=loss_occupancy
	loss['loss_occupancy']+=loss_occupancy


def occupancy_loss(logits_pred, weights=None, reduction_method='sum'):
	#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering		
	return cross_entropy_occupancy_loss(logits_pred,weights=weights, reduction_method=reduction_method)





