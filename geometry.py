import torch
import numpy as np
def camera_to_world(p, d, K, R, C, origin, scaling):

	#Make homogenous
	u=scaling.squeeze(1)*p+origin.squeeze(1)
	u=torch.cat((u, torch.ones((p.shape[0],p.shape[1],1)).cuda(non_blocking=True)),dim=-1)

	#Convert to world coordinate system
	X=torch.transpose(R.squeeze(1), 1,2)@torch.inverse(K.squeeze(1))@torch.transpose(u, 1,2) # (1, 3, n_points)
	X=torch.transpose(d, 1,2)*X + C.squeeze(1)

	return X.permute(0,2,1)

def world_to_camera(p_world,K,R,C,origin,scaling):
	batch_size,n_points,_=p_world.shape
	device=p_world.device

	#Transform world points to homogeneous coordinates
	p_world = torch.cat((p_world, torch.ones(batch_size,n_points,1).to(device)),dim=-1).permute(0,2,1)

	P=R@torch.cat((torch.eye(3).repeat(batch_size,1,1,1).to(device),-C),axis=3)

	u=K.squeeze(1)@P.squeeze(1)@p_world # (B,3,n_points)
	u=(1/scaling.squeeze(1))*u
	u[...,:2]-=((1/scaling.squeeze(1))*origin.squeeze(1))

	return u.permute(0,2,1)


def depth_to_3D(depth, K, R, C, scale_factor,origin,scale_est=1):
	"""
		Project our depth images to a set of 3D points

		Parameters
		----------
		depth - depth image: torch Tensor of shape (B,1,H,W)
		K - intrinsic parameters: torch Tensor of shape (B,1,3,3)
		R - rotation matrix: torch Tensor of shape (B,1,3,3)
		C - camera center: torch Tensor of shape (B,1,3,1)
		scale_factor -resizing scaling factor: float

		Returns
		-------
		output - set of 3D points: torch Tensor of shape (B,3,256,256)
		valid_mask - mask of valid depth points: torch Tensor of shape (B,1,H,W) 
	"""
	#Valid depth estimations are greater than or equal to 0
	valid_mask=torch.where(depth>0,torch.ones_like(depth),torch.zeros_like(depth)) # (B,1,H,W)

	#Calculate the 3D points
	h=depth.shape[2]
	w=depth.shape[3]
	batch_size=depth.shape[0]

	image_range=(-1,1)

	n_points=h*w
	pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))

	u = torch.stack([pixel_locations[1], pixel_locations[0]],dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1) # (B,N,2)
	u[...,0]=2*u[...,0]/(w-1)-1
	u[...,1]=2*u[...,1]/(h-1)-1
	u = u.permute(0,2,1).cuda() # (B,2,N)

	#Get the indices
	'''vv,uu=torch.meshgrid(torch.arange(depth.shape[2]), torch.arange(depth.shape[3]))
	vv=vv.flatten()
	uu=uu.flatten()

	#Build coordinate matrices
	u=torch.cat((uu[np.newaxis,:], vv[np.newaxis,:]),axis=0).cuda(non_blocking=True)
	#Tile to number of batches
	u=torch.repeat_interleave(u[np.newaxis,...], depth.shape[0],dim=0) #(B,2,N)'''
	u=scale_factor[:,0,...]*u+torch.transpose(origin[:,0,...],1,2)
	u=torch.cat((u,torch.ones((depth.shape[0],1,u.shape[-1])).cuda(non_blocking=True)),axis=1) #(B,3,N)

	#Move to 3D by multiplying by inverse camera intrinsics
	X=torch.transpose(R, 2,3)@torch.inverse(K)@(u[:,np.newaxis,...])
	#Reshape
	X=X.view((X.shape[0],X.shape[2],X.shape[3])) # (B,3,N)
	C=C.view((C.shape[0],C.shape[2],C.shape[3])) # (B,3,1)
	X=X.view((X.shape[0],X.shape[1],depth.shape[2],depth.shape[3])) # (B,3,H,W)
	#Multiply  by the depth and add the camera center
	output=scale_est*depth*X + C[...,np.newaxis]# (B,3,H,W)

	return output, valid_mask
