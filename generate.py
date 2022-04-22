import torch
import numpy as np
import math
from skimage import measure
from torch import distributions as dist
from geometry import world_to_camera
from utils import get_tensor_values
#import open3d as o3d
#from view_3D import visualize_3D

def batch_eval(points, model,c, num_samples=512 * 512 * 512):
	#Based on the function of the same name from https://github.com/shunsukesaito/PIFu
    num_pts = points.shape[1]
    sdf = np.zeros(num_pts)

    num_batches = num_pts // num_samples
    for i in range(num_batches):
        sdf[i * num_samples:i * num_samples + num_samples] = model.decode(
            points[:, i * num_samples:i * num_samples + num_samples].float(),c=c).logits
    if num_pts % num_samples:
        sdf[num_batches * num_samples:] = model.decode(points[:, num_batches * num_samples:].float(),c=c).logits

    return sdf


def eval_grid_octree(coords, model,c,
                     init_resolution=64, threshold=0.01,
                     num_samples=512 * 512 * 512):
    resolution = coords.shape

    sdf = np.zeros(resolution)
    print(sdf.shape)

    dirty = np.ones(resolution, dtype=np.bool)
    grid_mask = np.zeros(resolution, dtype=np.bool)

    reso = resolution[0] // init_resolution

    #point_cloud = None

    while reso > 0:
        # subdivide the grid
        grid_mask[0:resolution[0]:reso, 0:resolution[1]:reso, 0:resolution[2]:reso] = True
        # test samples in this iteration
        test_mask = np.logical_and(grid_mask, dirty)
        #print('step size:', reso, 'test sample size:', test_mask.sum())
        points = coords[:, test_mask]

        sdf[test_mask] = batch_eval(points, model,c, num_samples=num_samples)
        dirty[test_mask] = False

        '''if point_cloud is None:
            inside_points = sdf[test_mask]>0.5
            point_cloud = points[:,inside_points]
        else:
            inside_points = sdf[test_mask]>0.5
            point_cloud = np.append(point_cloud,points[:,inside_points],axis=1)'''

        # do interpolation
        if reso <= 1:
            break
        for x in range(0, resolution[0] - reso, reso):
            for y in range(0, resolution[1] - reso, reso):
                for z in range(0, resolution[2] - reso, reso):
                    # if center marked, return
                    if not dirty[x + reso // 2, y + reso // 2, z + reso // 2]:
                        continue
                    v0 = sdf[x, y, z]
                    v1 = sdf[x, y, z + reso]
                    v2 = sdf[x, y + reso, z]
                    v3 = sdf[x, y + reso, z + reso]
                    v4 = sdf[x + reso, y, z]
                    v5 = sdf[x + reso, y, z + reso]
                    v6 = sdf[x + reso, y + reso, z]
                    v7 = sdf[x + reso, y + reso, z + reso]
                    v = np.array([v0, v1, v2, v3, v4, v5, v6, v7])
                    v_min = v.min()
                    v_max = v.max()
                    # this cell is all the same
                    if (v_max - v_min) < threshold:
                        sdf[x:x + reso, y:y + reso, z:z + reso] = (v_max + v_min) / 2
                        dirty[x:x + reso, y:y + reso, z:z + reso] = False
        reso //= 2

    return sdf   


def generate_mesh(model, device, color, K, R, C, scaling, origin, mask, bb_min, bb_max,fname='test_mesh'):
	#Based on the function of the same name from https://github.com/shunsukesaito/PIFu
	batch_size=1000
	res=color.shape[2]
	coords=np.mgrid[:res,:res,:res]
	coords=coords.reshape(3,-1)
	coords=torch.from_numpy(coords)
	coords_matrix = torch.eye(4)
	#coords_matrix=coords_matrix.to(device)
	#coords = coords.to(device)

	length=bb_max-bb_min
	coords_matrix[0,0] = length[0]/res
	coords_matrix[1,1] = length[1]/res
	coords_matrix[2,2] = length[2]/res
	coords_matrix[0:3,3] = bb_min
	coords = coords_matrix[:3,:3]@coords.float() + coords_matrix[:3,3:4]

	coords=coords.T.to(device)
	#coords=coords.T

	sdf = torch.zeros((1,coords.shape[0]))
	logits = torch.zeros((1,coords.shape[0]))

	num_batches = int(math.ceil(coords.shape[0]/batch_size))

	c = model.encode_inputs(color)

	with torch.no_grad():
		#logits = eval_grid_octree(coords.view((res, res,res)), model,c,num_samples=batch_size)
		#print(logits.shape)
		for i in range(num_batches):
			sdf[:,i*batch_size:i*batch_size+batch_size] = model.decode(coords[np.newaxis,i*batch_size:i*batch_size+batch_size,:].float(),c=c).probs
	
	#sdf = dist.Bernoulli(logits=torch.tensor(logits)).probs
	sdf = sdf.view((res,res,res))

	# Finally we do marching cubes
	if torch.min(sdf)<=0.5 and torch.max(sdf)>=0.5:
		#try:
		verts, faces, _, _ = measure.marching_cubes_lewiner(sdf.cpu().numpy(), 0.50)
		#Convert verts into the world coordinate system
		verts = coords_matrix[:3,:3].cpu().numpy()@verts.T+coords_matrix[:3,3:4].cpu().numpy()
		verts=verts.T

		save_obj_mesh(fname+'.obj',verts, faces)
		#except:
		#	print("Error: cannot perform marching cubes")
	else:
		print("Error: cannot perform marching cubes")
		print("Min:%2.2f"%(torch.min(sdf)))
		print("Max:%2.2f"%(torch.max(sdf)))


def generate_mesh_color(model, device, color,points, K, R, C, scaling, origin, mask, bb_min, bb_max,fname='test_mesh'):
	#Based on the function of the same name from https://github.com/shunsukesaito/PIFu
	batch_size=1000
	res=color.shape[2]
	coords=np.mgrid[:res,:res,:res]
	coords=coords.reshape(3,-1)
	coords=torch.from_numpy(coords)
	coords_matrix = torch.eye(4)

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
			verts = coords_matrix[:3,:3]@torch.tensor(verts.T)+coords_matrix[:3,3:4]
			verts=verts.T
			verts_tensor = torch.from_numpy(verts).unsqueeze(0).to(device).float() # (B,N,3)
			color_output = torch.zeros(verts_tensor.shape)
			for i in range(color_output.shape[1]//batch_size):
				left = i*batch_size
				right = i*batch_size+batch_size
				if i ==len(color) //batch_size -1:
					right = -1
				temp = model.decode_color(verts_tensor[:,left:right,:],c=c) 
				color_output[:,left:right,:] = model.decode_color(verts_tensor[:,left:right,:],c=c)

			save_obj_mesh_with_color(fname+'.obj',verts.detach().cpu().numpy(), faces,color_output[0].detach().cpu().numpy())
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


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
	#Code from PIFU mesh_util.py
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    print("verts saved")
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    print("faces saved")
    file.close()




