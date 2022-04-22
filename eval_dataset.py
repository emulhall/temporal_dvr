import torch
from torch.utils.data.dataset import Dataset
import pickle
import numpy as np
from torchvision import transforms
from PIL import Image
import scipy
import cv2
import glob
import os
import matplotlib.pyplot as plt
import trimesh
import random
from geometry import world_to_camera

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )

class RenderPeopleDataset(Dataset):
	def __init__(self, dataroot, num_views=1,usage='train',num_samples=5000,scale=0.1,random_multiview=False):

		# Path setup
		self.root = dataroot
		self.RENDER = os.path.join(self.root, '0','color')
		self.MASK = os.path.join(self.root, '0','gr_mask')
		self.PARAM = os.path.join(self.root, '0','camera')
		self.BBOX = os.path.join(self.root, '0','boundingBox')
		self.DEPTH = os.path.join(self.root, '0', 'complete_depth')
		self.OBJ = os.path.join(self.root.replace('train_512_RenderPeople_all_sparse','meshes'))

		self.bb_min = np.array([0, -1, -0.4])
		self.bb_max = np.array([3.6, 3.0, 2.6])

		self.is_train = (usage == 'train')

		self.num_views =num_views

		self.num_samples = num_samples

		self.to_tensor=transforms.ToTensor()

		self.scale = scale

		views = sorted(glob.glob(os.path.join(self.root,'*')))
		self.views=[]
		for v in views:
			_,num = os.path.split(v)
			if int(num)==56 or int(num)==83:
				continue
			elif int(num)==77:
				self.views.append(num)
			else:
				#self.views.append(num)
				continue

		self.subjects = self.get_subjects()
		self.random_multiview = random_multiview


	def get_subjects(self):
		all_subjects = glob.glob(os.path.join(self.root,'0/color/*.png'))
		all_subjects = sorted([i.replace('.png','').replace(os.path.join(self.root,'0/color/'),'') for i in all_subjects])
		#all_subjects = ['%i'%i for i in range(0,1)]

		#var_subjects = ['%i'%i for i in range(330,345)]
		var_subjects=['330']
		if len(var_subjects) == 0:
			return all_subjects

		if self.is_train:
			return sorted(list(set(all_subjects) - set(var_subjects)))
		else:
			return sorted(list(var_subjects))


	def calculate_scaling_and_origin_rp(self,bbox, height):
		"""
		Calculates the origin and scaling of an image for the RenderPeople dataset

		Parameters
		---------
		bbox - the bounding box coords: ndarray of shape (4,2)
		height - the height of the final image: int
		"""
		bb1_t=bbox-1
		bbc1_t=bb1_t[2:4,0:3]
		origin = np.multiply([bb1_t[1,0]-bbc1_t[1,0],bb1_t[0,0]-bbc1_t[0,0]],2)
		squareSize = np.maximum(bb1_t[0,1]-bb1_t[0,0]+1,bb1_t[1,1]-bb1_t[1,0]+1)
		scaling = np.multiply(np.true_divide(squareSize,height),2)

		return origin, scaling


	def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
		'''
		Return the render data
		:param subject: subject name
		:param num_views: how many views to return
		:param view_id: the first view_id. If None, select a random one.
		:return:
			'img': [num_views, C, W, H] images
			'calib': [num_views, 4, 4] calibration matrix
			'extrinsic': [num_views, 4, 4] extrinsic matrix
			'mask': [num_views, 1, W, H] masks
		'''

		# The ids are an even distribution of num_views around view_id
		view_ids = [self.views[(yid + len(self.views) // num_views * offset) % len(self.views)] for offset in range(num_views)]

		if random_sample:
			view_ids = np.random.choice(self.views, num_views, replace=False)

		render_list = []
		mask_list = []
		K_list = []
		R_list = []
		C_list = []
		origin_list = []
		scale_list = []
		samples_list = []
		labels_list = []


		for vid in view_ids:
			K_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'K.txt')
			R_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'R.txt')
			C_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'cen.txt')
			render_path = os.path.join(self.RENDER.replace('/0/','/%s/'%(vid)), subject+'.png')
			mask_path = os.path.join(self.MASK.replace('/0/','/%s/'%(vid)), subject+'.png')
			bbox_path = os.path.join(self.BBOX.replace('/0/','/%s/'%(vid)),subject+'.txt')
			mesh_path = os.path.join('/media/mulha024/i/meshes/','%s.obj'%(subject))

			# loading calibration data
			K = torch.tensor(np.loadtxt(K_path,delimiter=',',dtype='f')).float()
			R = torch.tensor(np.loadtxt(R_path,delimiter=',',dtype='f')).float()
			center = np.loadtxt(C_path,delimiter=',',dtype='f')
			center = torch.tensor(center[...,np.newaxis]).float()
			bbox = np.loadtxt(bbox_path,delimiter=',',dtype='f')
			origin,scaling = self.calculate_scaling_and_origin_rp(bbox,512)
			origin = torch.tensor(origin[np.newaxis,...]).float()
			scaling = torch.tensor(scaling[np.newaxis,np.newaxis,...]).float()

			mask = Image.open(mask_path).convert('L')
			render = Image.open(render_path).convert('RGB')

			mask = transforms.ToTensor()(mask).float()
			mask = torch.where(mask>0,torch.ones_like(mask),torch.zeros_like(mask))
			mask_list.append(mask)

			render = self.to_tensor(render).float()
			render = mask.expand_as(render) * render

			samples, labels = self.sample(mask, K, R, center, origin, scaling, mesh_path)

			render_list.append(render)
			K_list.append(K)
			R_list.append(R)
			C_list.append(center)
			origin_list.append(origin)
			scale_list.append(scaling)
			samples_list.append(samples)
			labels_list.append(labels)

			#verts_tensor = torch.Tensor(self.mesh_dic[subject].vertices).T
			#faces = torch.Tensor(self.mesh_dic[subject].faces).T

			'''xyz_tensor = perspective(verts_tensor[np.newaxis,...], calib[np.newaxis,...], transforms_tensor[np.newaxis,...])
			uv = xyz_tensor[:, :2, :]
			color = index(render[np.newaxis,...], uv).detach().cpu().numpy()[0].T
			color = color * 0.5 + 0.5
			save_obj_mesh_with_color('/media/mulha024/i/PIFu-master/test_mesh.obj', verts_tensor.T, faces.T, color)
			print("Saved the mesh")'''



		return {
			'color': torch.stack(render_list, dim=0),
			'K': torch.stack(K_list, dim=0),
			'R': torch.stack(R_list, dim=0),
			'C': torch.stack(C_list, dim=0),
			'origin': torch.stack(origin_list, dim=0),
			'scaling': torch.stack(scale_list, dim=0),
			'mask': torch.stack(mask_list, dim=0),
			'samples': torch.stack(samples_list,dim=0),
			'labels': torch.stack(labels_list, dim=0)
			}

	def sample(self, mask, K, R, C, origin, scaling, mesh_path):
		if not self.is_train:
			random.seed(1991)
			np.random.seed(1991)
			torch.manual_seed(1991)

		mesh = trimesh.load(mesh_path,process=False, file_type='obj')
		mesh.fill_holes()
		vertices = torch.Tensor(mesh.vertices) # N,3

		xyz = world_to_camera(vertices.unsqueeze(0),K.unsqueeze(0).unsqueeze(1),R.unsqueeze(0).unsqueeze(1),C.unsqueeze(0).unsqueeze(1),origin.unsqueeze(0).unsqueeze(1),scaling.unsqueeze(0).unsqueeze(1))
		xy = xyz[:, :, :2]
		z = xyz[:, :, 2:3]
		in_img = torch.where((xy[..., 0] >= 0) & (xy[..., 0] <= 512.0) & (xy[..., 1] >= 0) & (xy[..., 1] <= 512))
		vertices = vertices[in_img[1],:]	#N,3

		surface_points = vertices[np.random.choice(np.arange(vertices.shape[0]), size=min(self.num_samples*6,vertices.shape[0]), replace=False),:]
		#surface_points, _ = trimesh.sample.sample_surface(mesh, self.num_samples*6)
		sample_points = surface_points + np.random.normal(scale=self.scale, size=surface_points.shape) # N,3

		length = self.bb_max- self.bb_min
		random_points = np.random.rand(self.num_samples, 3) * length + self.bb_min
		xyz = world_to_camera(torch.Tensor(random_points).unsqueeze(0),K.unsqueeze(0).unsqueeze(1),R.unsqueeze(0).unsqueeze(1),C.unsqueeze(0).unsqueeze(1),origin.unsqueeze(0).unsqueeze(1),scaling.unsqueeze(0).unsqueeze(1))
		xy = xyz[:, :, :2]
		z = xyz[:, :, 2:3]
		in_img = torch.where((xy[..., 0] >= 0) & (xy[..., 0] <= 512.0) & (xy[..., 1] >= 0) & (xy[..., 1] <= 512))
		random_points = random_points[in_img[1],:]	#N,3

		sample_points = np.concatenate([sample_points, random_points], 0)
		np.random.shuffle(sample_points)

		inside = mesh.contains(sample_points)
		inside_points = sample_points[inside]
		outside_points = sample_points[np.logical_not(inside)]



		nin=inside_points.shape[0]
		inside_points = inside_points[:self.num_samples//2] if nin > self.num_samples else inside_points
		outside_points = outside_points[:self.num_samples//2] if nin > self.num_samples else outside_points[:(self.num_samples-nin)]

		samples = np.concatenate([inside_points, outside_points],0)
		labels = np.concatenate([np.ones((inside_points.shape[0],1)), np.zeros((outside_points.shape[0],1))])

		samples = torch.Tensor(samples).float()
		labels = torch.Tensor(labels).float()

		save_samples_truncted_prob('test_samples.ply', samples, labels)

		del mesh

		return samples, labels


	def __len__(self):
		return len(self.subjects) * len(self.views)


	def get_item(self, index):
		# In case of a missing file or IO error, switch to a random sample instead
		# try:
		sid = index % len(self.subjects)
		tmp = index // len(self.subjects)
		yid = tmp % len(self.views)
		pid = tmp // len(self.views)

		# name of the subject 'rp_xxxx_xxx'
		subject = self.subjects[sid]
		res = {
			'name': subject,
			'sid': sid,
			'yid': yid,
			'pid': pid,
			'bb_min': self.bb_min,
			'bb_max': self.bb_max,
		}
		
		render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.random_multiview)
		res.update(render_data)
		
		#sample_data = self.sample(res['mask'], res['K'], res['R'], res['C'], res['origin'], res['scaling'], depth_img=res['gt_depth'])
		#res.update(sample_data)

		return res

	def __getitem__(self, index):
		return self.get_item(index)



class HUMBIDataset(Dataset):
	def __init__(self, dataroot, num_views=2,usage='train',num_samples=5000,scale=0.1,random_multiview=False):

		# Path setup
		self.root = dataroot
		self.RENDER = 'color'
		self.MASK = 'mask'
		self.PARAM = 'camera'
		self.BBOX = 'bbox'
		self.IUV = 'correspondences'

		self.bb_min = np.array([0, -1, -0.4])
		self.bb_max = np.array([3.6, 3.0, 2.6])

		self.is_train = (usage == 'train')

		self.num_views =num_views

		self.num_samples = num_samples

		self.to_tensor=transforms.ToTensor()

		self.scale = scale

		self.subjects = self.get_subjects()
		time_stamps = sorted(glob.glob(os.path.join(self.root,self.subjects[0],'*')))
		self.time_stamps = []
		for t in time_stamps:
			_, time_stamp = os.path.split(t)
			self.time_stamps.append(time_stamp)
		self.random_multiview = random_multiview

		views = sorted(glob.glob(os.path.join(self.root,self.subjects[0],self.time_stamps[0],'correspondences','*.pkl')))
		self.views=[]
		for v in views:
			_,num = os.path.split(v)
			num=num[:-4]
			if int(num)==106:
				self.views.append(int(num))

	def get_subjects(self):
		all_subjects_list = glob.glob(os.path.join(self.root,'*'))
		all_subjects = []
		for s in all_subjects_list:
			_, subject = os.path.split(s)
			all_subjects.append(subject)

		
		var_subjects=[]
		if len(var_subjects) == 0:
			return all_subjects

		if self.is_train:
			return sorted(list(set(all_subjects) - set(var_subjects)))
		else:
			return sorted(list(var_subjects))

	def calculate_scaling_and_origin(self, bbox, size):
		y_max = bbox[0]
		y_min = bbox[1]
		x_max = bbox[2]
		x_min = bbox[3]

		origin = [x_min, y_min]

		square_size = max((x_max-x_min),(y_max-y_min))
		scaling = np.true_divide(square_size, size)

		return np.asarray(origin), scaling


	def get_render(self, subject, time_stamp, num_views, yid=0, pid=0, random_sample=False):
		'''
		Return the render data
		:param subject: subject name
		:param num_views: how many views to return
		:param view_id: the first view_id. If None, select a random one.
		:return:
			'img': [num_views, C, W, H] images
			'calib': [num_views, 4, 4] calibration matrix
			'extrinsic': [num_views, 4, 4] extrinsic matrix
			'mask': [num_views, 1, W, H] masks
		'''

		# The ids are an even distribution of num_views around view_id
		view_id = self.views[yid]
		view_ids = [view_id]

		if random_sample:
			view_ids = np.random.choice(self.views, num_views, replace=False)

		render_list = []
		mask_list = []
		K_list = []
		R_list = []
		C_list = []
		origin_list = []
		scale_list = []

		K_path = os.path.join(self.root, subject, time_stamp, self.PARAM, '%i_K.txt'%(view_id))
		R_path = os.path.join(self.root, subject, time_stamp, self.PARAM, '%i_R.txt'%(view_id))			
		C_path = os.path.join(self.root, subject, time_stamp, self.PARAM, '%i_C.txt'%(view_id))
		render_path = os.path.join(self.root, subject, time_stamp, self.RENDER, '%07d.png'%(view_id))
		mask_path = render_path.replace('color','mask')
		bbox_path = render_path.replace('color', 'bbox').replace('.png','.txt')

		K = torch.tensor(np.loadtxt(K_path,delimiter=',',dtype='f')).float()
		R = torch.tensor(np.loadtxt(R_path,delimiter=',',dtype='f')).float()
		center = np.loadtxt(C_path,delimiter=',',dtype='f')
		center = torch.tensor(center[...,np.newaxis]).float()
		bbox = np.loadtxt(bbox_path,delimiter=',',dtype='f')
		origin, scaling = self.calculate_scaling_and_origin(bbox, 512)
		origin = torch.tensor(origin[np.newaxis,...]).float()
		scaling = torch.tensor(scaling[np.newaxis,np.newaxis,...]).float()		

		mask = Image.open(mask_path).convert('L')
		render = Image.open(render_path).convert('RGB')

		mask = transforms.ToTensor()(mask).float()
		mask = torch.where(mask>0,torch.ones_like(mask),torch.zeros_like(mask))

		render = self.to_tensor(render).float()
		render = mask.expand_as(render) * render

		render_list.append(render)
		mask_list.append(mask)
		origin_list.append(origin)
		scale_list.append(scaling)
		K_list.append(K)
		R_list.append(R)
		C_list.append(center)
			

		return {
			'color': torch.stack(render_list, dim=0),
			'K': torch.stack(K_list, dim=0),
			'R': torch.stack(R_list, dim=0),
			'C': torch.stack(C_list, dim=0),
			'origin': torch.stack(origin_list, dim=0),
			'scaling': torch.stack(scale_list, dim=0),
			'mask': torch.stack(mask_list, dim=0),
			}			


	def __len__(self):
		return len(self.subjects) * len(self.time_stamps) * len(self.views)


	def get_item(self, index):
		# In case of a missing file or IO error, switch to a random sample instead
		# try:
		sid = index % len(self.subjects)
		tmp = index // len(self.subjects)
		tid = tmp % len(self.time_stamps)
		yid = tmp % len(self.views)
		pid = tmp // len(self.views)

		subject = self.subjects[sid]
		time_stamp = self.time_stamps[tid]

		res = {
			'name': subject,
			'sid': sid,
			'yid': yid,
			'tid': tid,
			'bb_min': self.bb_min,
			'bb_max': self.bb_max,
		}

		render_data = self.get_render(subject, time_stamp, num_views=self.num_views, yid=yid, pid=pid, random_sample=self.random_multiview)
		res.update(render_data)

		return res

	def __getitem__(self, index):
		return self.get_item(index)		
