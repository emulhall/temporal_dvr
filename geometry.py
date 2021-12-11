import torch
def camera_to_world(p, d, K, R, C, origin, scaling):
	'''
		Inputs
		-------------------------
		p: 2D points sampled from image plane (B,n_points, 2)
		d: depth (B, n_points, 1)
		K: camerea intrinsics (B, 1, 3, 3)
		R: camera rotation matrix (B,1,3,3)
		C: camera center of origin (B,1,3,1)
		origin: origin for rescaling purposes (B,1,1,2)
		scaling: scaling for rescaling purposes (B,1,1,1)

	'''

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
	print(u.shape)
