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
from utils import sample_n, get_freespace_points, get_occupancy_points, get_mask_points


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
			else:
				self.views.append(num)

		self.subjects = self.get_subjects()
		self.random_multiview = random_multiview


	def get_subjects(self):
		#all_subjects = glob.glob(os.path.join(self.root,'0/color/*.png'))
		#all_subjects = sorted([i.replace('.png','').replace(os.path.join(self.root,'0/color/'),'') for i in all_subjects])
		all_subjects = ['%i'%i for i in range(0,1)]

		#var_subjects = ['%i'%i for i in range(330,345)]
		var_subjects=[]
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
		depth_list = []
		p_list = []
		p_freespace_list = []
		p_occupancy_list = []
		p_mask_list = []

		for vid in view_ids:
			K_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'K.txt')
			R_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'R.txt')
			C_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'cen.txt')
			render_path = os.path.join(self.RENDER.replace('/0/','/%s/'%(vid)), subject+'.png')
			mask_path = os.path.join(self.MASK.replace('/0/','/%s/'%(vid)), subject+'.png')
			bbox_path = os.path.join(self.BBOX.replace('/0/','/%s/'%(vid)),subject+'.txt')
			gt_depth_path = os.path.join(self.DEPTH.replace('/0/','/%s/'%(vid)), subject+'.txt')

			# loading calibration data
			K = torch.tensor(np.loadtxt(K_path,delimiter=',',dtype='f')).float()
			R = torch.tensor(np.loadtxt(R_path,delimiter=',',dtype='f')).float()
			center = np.loadtxt(C_path,delimiter=',',dtype='f')
			center = torch.tensor(center[...,np.newaxis]).float()
			bbox = np.loadtxt(bbox_path,delimiter=',',dtype='f')
			origin,scaling = self.calculate_scaling_and_origin_rp(bbox,512)
			origin = torch.tensor(origin[np.newaxis,...]).float()
			scaling = torch.tensor(scaling[np.newaxis,np.newaxis,...]).float()
			gt_depth = torch.tensor(np.genfromtxt(gt_depth_path,delimiter=',',dtype='f')).float()

			valid_depth=self.to_tensor(np.where(gt_depth>=1.0, np.ones_like(gt_depth),np.zeros_like(gt_depth))*np.where(gt_depth<=8.0, np.ones_like(gt_depth),np.zeros_like(gt_depth)))

			mask = Image.open(mask_path).convert('L')
			render = Image.open(render_path).convert('RGB')

			mask = transforms.ToTensor()(mask).float()
			mask = torch.where(mask>0,torch.ones_like(mask),torch.zeros_like(mask))
			mask = mask*valid_depth
			mask_list.append(mask)

			render = self.to_tensor(render).float()
			render = mask.expand_as(render) * render

			p,p_freespace,p_occupancy,p_mask = self.sample(mask, K, R, center, origin, scaling, depth_img=gt_depth)

			render_list.append(render)
			K_list.append(K)
			R_list.append(R)
			C_list.append(center)
			origin_list.append(origin)
			scale_list.append(scaling)
			depth_list.append(gt_depth)
			p_list.append(p)
			p_freespace_list.append(p_freespace)
			p_occupancy_list.append(p_occupancy)
			p_mask_list.append(p_mask)

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
			'gt_depth': torch.stack(depth_list, dim=0),
			'p': torch.stack(p_list, dim=0),
			'p_freespace': torch.stack(p_freespace_list, dim=0),
			'p_occupancy': torch.stack(p_occupancy_list, dim=0),
			'p_mask': torch.stack(p_mask_list, dim=0)
			}

	def sample(self, mask, K, R, C, origin, scaling, depth_img=None):
		per_sample = self.num_samples//4
		p = sample_n(mask.unsqueeze(0), per_sample*3) # N,2
		p_freespace=None
		p_occupancy=None

		if depth_img is not None:
			p_freespace = get_freespace_points(p[:,:per_sample,:], K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0),depth_img=depth_img.unsqueeze(0).unsqueeze(1), scale=self.scale)
			p_occupancy = get_occupancy_points(p[:,per_sample:per_sample*2,:], K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0),depth_img=depth_img.unsqueeze(0).unsqueeze(1), scale=self.scale)
		else:
			p_freespace = get_freespace_points(p[:,:per_sample,:], K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0),scale=self.scale)
			p_occupancy = get_occupancy_points(p[:,per_sample:per_sample*2,:], K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0), scale=self.scale)
		
		p_mask = get_mask_points(mask.unsqueeze(0),K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0), origin.unsqueeze(0), scaling.unsqueeze(0), per_sample, self.bb_min, self.bb_max)
		p=p[:,per_sample*2:,:]

		return p[0], p_freespace[0], p_occupancy[0], p_mask[0]


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

class RenderPeopleDataset_temp(Dataset):
	def __init__(self, usage='train', pickle_file='./data/RenderPeople_dvr.pkl',skip_every_n_image=1, num_samples=5000, scale=0.1):

		self.to_tensor=transforms.ToTensor()
		self.num_samples = num_samples
		self.scale = scale
		self.bb_min = torch.FloatTensor([0,-1,-0.5])
		self.bb_max = torch.FloatTensor([3.6, 3,2.6])

		with open(pickle_file, 'rb') as f:
			self.data_info = pickle.load(f)[usage]
			self.idx = [i for i in range(0, len(self.data_info[0][0]), skip_every_n_image)]
			self.data_len = len(self.idx)

	def sample(self, mask, K, R, C, origin, scaling, depth_img=None):
		per_sample = self.num_samples//4
		p = sample_n(mask, per_sample*3) # N,2
		p_freespace=None
		p_occupancy=None

		if depth_img is not None:
			p_freespace = get_freespace_points(p[:per_sample,:].unsqueeze(0), K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0),depth_img=depth_img.unsqueeze(0), scale=self.scale)
			p_occupancy = get_occupancy_points(p[per_sample:per_sample*2,:].unsqueeze(0), K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0),depth_img=depth_img.unsqueeze(0), scale=self.scale)
		else:
			p_freespace = get_freespace_points(p[:per_sample,:].unsqueeze(0), K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0),scale=self.scale)
			p_occupancy = get_occupancy_points(p[per_sample:per_sample*2,:].unsqueeze(0), K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0),origin.unsqueeze(0),scaling.unsqueeze(0), scale=self.scale)
		
		p_mask = get_mask_points(mask.unsqueeze(0),K.unsqueeze(0),R.unsqueeze(0),C.unsqueeze(0), origin.unsqueeze(0), scaling.unsqueeze(0), per_sample, self.bb_min, self.bb_max)
		p=p[per_sample*2:,:]

		return p, p_freespace[0],p_occupancy[0],p_mask[0]
		



	def __getitem__(self, index):
		output = {}

		color = self.to_tensor(Image.open(self.data_info[0][0][index]))

		mask = cv2.imread(self.data_info[0][2][index])
		K = np.loadtxt(self.data_info[0][4][index],delimiter=',')
		R = np.loadtxt(self.data_info[0][5][index],delimiter=',')
		C = np.loadtxt(self.data_info[0][6][index],delimiter=',')

		K=self.to_tensor(np.asarray(K,dtype='f'))
		R=self.to_tensor(np.asarray(R,dtype='f'))
		C=self.to_tensor(np.asarray(C[...,np.newaxis],dtype='f'))

		gt_depth = np.genfromtxt(self.data_info[0][7][index],delimiter=',')

		#Build the normal from the 3 files
		gt_norm_1_0 = np.genfromtxt(self.data_info[0][8][index],delimiter=',')
		gt_norm_2_0 = np.genfromtxt(self.data_info[0][9][index],delimiter=',')
		gt_norm_3_0 = np.genfromtxt(self.data_info[0][10][index],delimiter=',')
		gt_norm = np.concatenate((gt_norm_1_0[...,np.newaxis], gt_norm_2_0[...,np.newaxis], gt_norm_3_0[...,np.newaxis]), axis=2)
		gt_norm = self.to_tensor(np.asarray(gt_norm,dtype='f'))

		#Refine the mask with valid depth
		valid_depth=np.where(gt_depth>=1.0, np.ones_like(gt_depth),np.zeros_like(gt_depth))*np.where(gt_depth<=8.0, np.ones_like(gt_depth),np.zeros_like(gt_depth))
		mask_refined = self.to_tensor(np.where(mask>100,np.ones_like(mask),np.zeros_like(mask))*valid_depth[...,np.newaxis])
		mask_refined_1D = mask_refined[0,...]

		gt_depth = self.to_tensor(np.asarray(gt_depth,dtype='f'))

		origin = self.data_info[0][13][index]
		origin = self.to_tensor(np.asarray(origin[np.newaxis,...],dtype='f'))

		scaling = np.asarray([self.data_info[0][14][index]],dtype='f')
		scaling=self.to_tensor(scaling[np.newaxis,...])

		#Refine color image, depth image and normal image using the refined mask
		color=torch.where(mask_refined>0,color,torch.ones_like(color))
		gt_depth=gt_depth*mask_refined_1D[np.newaxis,...]
		gt_norm=gt_norm*mask_refined

		p, p_freespace, p_occupancy, p_mask = self.sample(mask_refined, K, R, C, origin, scaling, depth_img=gt_depth)

		output['color'] = color

		output['mask'] = mask_refined

		output['K'] = K

		output['R'] = R

		output['C'] = C

		output['gt_depth'] = gt_depth

		output['gt_normal'] = gt_norm

		output['origin'] = origin

		output['scaling'] = scaling

		output['p'] = p

		output['p_freespace'] = p_freespace

		output['p_occupancy'] = p_occupancy

		output['p_mask'] = p_mask

		output['bb_min'] = self.bb_min

		output['bb_max'] = self.bb_max

		return output


	def __len__(self):
		return self.data_len



class HUMBIDataset(Dataset):
	def __init__(self, usage='train', pickle_file='./data/HUMBI_dvr.pkl',skip_every_n_image=1):

		self.to_tensor=transforms.ToTensor()

		with open(pickle_file, 'rb') as f:
			self.data_info = pickle.load(f)[usage]
			self.idx = [i for i in range(0, len(self.data_info[0][0]), skip_every_n_image)]
			self.data_len = len(self.idx)


	def __getitem__(self, index):
		output = {}

		color_1 = self.to_tensor(Image.open(self.data_info[0][0][index]))
		color_2 = self.to_tensor(Image.open(self.data_info[1][0][index]))

		dp_1 = self.to_tensor(Image.open(self.data_info[0][1][index]))
		dp_2 = self.to_tensor(Image.open(self.data_info[1][1][index]))	

		mask_1 =self.to_tensor(Image.open(self.data_info[0][2][index]))
		mask_2 = self.to_tensor(Image.open(self.data_info[1][2][index]))

		mask_1=torch.where(mask_1[-1,...]>0, torch.ones_like(mask_1[-1,...]),torch.zeros_like(mask_1[-1,...]))
		mask_2=torch.where(mask_2[-1,...]>0, torch.ones_like(mask_2[-1,...]),torch.zeros_like(mask_2[-1,...]))
		mask_1=mask_1[np.newaxis,...]
		mask_2=mask_2[np.newaxis,...]

		color_1 = color_1*mask_1
		color_2 = color_2*mask_2

		dp_1 = dp_1*mask_1
		dp_2 = dp_2*mask_2

		K = np.loadtxt(self.data_info[0][4][index],delimiter=',')
		R = np.loadtxt(self.data_info[0][5][index],delimiter=',')
		C = np.loadtxt(self.data_info[0][6][index],delimiter=',')

		iuv_1 = self.data_info[1][11][index]
		iuv_2 = self.data_info[1][12][index]

		origin_1 = self.data_info[0][13][index]
		origin_2 = self.data_info[1][13][index]		

		scaling_1 = 2*np.asarray([self.data_info[0][14][index]],dtype='f')
		scaling_2 = 2*np.asarray([self.data_info[1][14][index]],dtype='f')		

		output['color_1'] = color_1
		output['color_2'] = color_2

		output['dp_1'] = dp_1
		output['dp_2'] = dp_2

		output['mask_1'] = mask_1
		output['mask_2'] = mask_2		

		output['K'] = self.to_tensor(np.asarray(K,dtype='f'))	

		output['R'] = self.to_tensor(np.asarray(R,dtype='f'))	

		output['C'] = self.to_tensor(np.asarray(C[...,np.newaxis],dtype='f'))		

		output['origin_1'] = self.to_tensor(np.asarray(origin_1[np.newaxis,...],dtype='f'))
		output['origin_2'] = self.to_tensor(np.asarray(origin_2[np.newaxis,...],dtype='f'))		

		output['scaling_1'] = self.to_tensor(scaling_1[np.newaxis,...])
		output['scaling_2'] = self.to_tensor(scaling_2[np.newaxis,...])

		output['iuv_1'] = self.to_tensor(np.asarray(iuv_1,dtype='f'))
		output['iuv_2'] = self.to_tensor(np.asarray(iuv_2,dtype='f'))

		return output


	def __len__(self):
		return self.data_len