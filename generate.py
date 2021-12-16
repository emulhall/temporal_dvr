import torch
import numpy as np
import math
from skimage import measure
from torch import distributions as dist
#import open3d as o3d
#from view_3D import visualize_3D


def generate_mesh(model, device, color,points,mask, padding=0.1,fname='test_mesh'):
	batch_size=1000
	res=color.shape[2]
	coords=np.mgrid[:res,:res,:res]
	coords=coords.reshape(3,-1)
	coords=torch.from_numpy(coords)
	coords_matrix = torch.eye(4)

	valid=torch.where(mask[:,0,...]>0)
	points = points[valid[0],:,valid[1],valid[2]]

	x_min=torch.min(points[:,0])
	y_min=torch.min(points[:,1])
	z_min=torch.min(points[:,2])
	bb_min=torch.zeros((3))
	bb_min[0]=x_min - padding
	bb_min[1]=y_min - padding
	bb_min[2]=z_min - padding

	x_max=torch.max(points[:,0])
	y_max=torch.max(points[:,1])
	z_max=torch.max(points[:,2])
	bb_max=torch.zeros((3))
	bb_max[0]=x_max + padding
	bb_max[1]=y_max + padding
	bb_max[2]=z_max + padding

	length=bb_max-bb_min
	coords_matrix[0,0] = length[0]/res
	coords_matrix[1,1] = length[1]/res
	coords_matrix[2,2] = length[2]/res
	coords_matrix[0:3,3] = bb_min
	coords = coords_matrix[:3,:3]@coords.float() + coords_matrix[:3,3:4]

	coords=coords.T.to(device)

	sdf = torch.zeros((1,coords.shape[0]))
	logits = torch.zeros((1,coords.shape[0]))

	num_batches = int(math.ceil(coords.shape[0]/batch_size))

	c = model.encode_inputs(color)

	with torch.no_grad():
		for i in range(num_batches):
			logits[:,i*batch_size:i*batch_size+batch_size] = model.decode(coords[np.newaxis,i*batch_size:i*batch_size+batch_size,:].float(),c=c).logits
	
	sdf = dist.Bernoulli(logits=logits).probs
	sdf = sdf.view((res,res,res))

	# Finally we do marching cubes
	if torch.min(sdf)<=0.5 and torch.max(sdf)>=0.5:
		try:
			verts, faces, normals, values = measure.marching_cubes(sdf.cpu().numpy(), 0.50)
			#Convert verts into the world coordinate system
			verts = coords_matrix[:3,:3].cpu().numpy()@verts.T+coords_matrix[:3,3:4].cpu().numpy()
			verts=verts.T
			save_obj_mesh(fname+'.obj',verts, faces)
		except:
			print("Error: cannot perform marching cubes")
	else:
		print("Error: cannot perform marching cubes")
		print("Min:%2.2f"%(torch.min(sdf)))
		print("Max:%2.2f"%(torch.max(sdf)))

def generate_mesh_unsupervised(model, device, color,points,mask, padding=0.1,fname='test_mesh'):
	batch_size=1000
	res=color.shape[2]
	coords=np.mgrid[:res,:res,:res]
	coords=coords.reshape(3,-1)
	coords=torch.from_numpy(coords)
	coords_matrix = torch.eye(4)

	valid=torch.where(mask>0)
	points = points[valid[0],valid[1],:]

	x_min=torch.min(points[:,0])
	y_min=torch.min(points[:,1])
	z_min=torch.min(points[:,2])
	bb_min=torch.zeros((3))
	bb_min[0]=x_min - padding
	bb_min[1]=y_min - padding
	bb_min[2]=z_min - padding

	x_max=torch.max(points[:,0])
	y_max=torch.max(points[:,1])
	z_max=torch.max(points[:,2])
	bb_max=torch.zeros((3))
	bb_max[0]=x_max + padding
	bb_max[1]=y_max + padding
	bb_max[2]=z_max + padding

	length=bb_max-bb_min
	coords_matrix[0,0] = length[0]/res
	coords_matrix[1,1] = length[1]/res
	coords_matrix[2,2] = length[2]/res
	coords_matrix[0:3,3] = bb_min
	coords = coords_matrix[:3,:3]@coords.float() + coords_matrix[:3,3:4]

	coords=coords.T.to(device)

	sdf = torch.zeros((1,coords.shape[0]))
	logits = torch.zeros((1,coords.shape[0]))

	num_batches = int(math.ceil(coords.shape[0]/batch_size))

	c = model.encode_inputs(color)

	with torch.no_grad():
		for i in range(num_batches):
			logits[:,i*batch_size:i*batch_size+batch_size] = model.decode(coords[np.newaxis,i*batch_size:i*batch_size+batch_size,:].float(),c=c).logits
	
	sdf = dist.Bernoulli(logits=logits).probs
	sdf = sdf.view((res,res,res))

	# Finally we do marching cubes
	if torch.min(sdf)<=0.5 and torch.max(sdf)>=0.5:
		try:
			verts, faces, normals, values = measure.marching_cubes(sdf.cpu().numpy(), 0.50)
			#Convert verts into the world coordinate system
			verts = coords_matrix[:3,:3].cpu().numpy()@verts.T+coords_matrix[:3,3:4].cpu().numpy()
			verts=verts.T
			save_obj_mesh(fname+'.obj',verts, faces)
		except:
			print("Error: cannot perform marching cubes")
	else:
		print("Error: cannot perform marching cubes")
		print("Min:%2.2f"%(torch.min(sdf)))
		print("Max:%2.2f"%(torch.max(sdf)))

def save_obj_mesh(mesh_path, verts, faces):
	#Code from PIFU mesh_util.py
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()




