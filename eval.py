import argparse
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import distributions as dist
import math
import random

from eval_dataset import RenderPeopleDataset, HUMBIDataset
from config import get_model
from utils import get_tensor_values, save_3D, reshape_multiview_tensors, calc_error
from loss import calculate_photoconsistency_loss, calculate_depth_loss, calculate_normal_loss, calculate_freespace_loss, calculate_occupancy_loss, calculate_mask_loss
from geometry import depth_to_3D
from view_3D import visualize_3D_masked, visualize_3D
from render import render_img
from generate import generate_mesh, generate_mesh_color

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	#parser.add_argument('--input', type=str, default='/mars/mnt/oitstorage/emily_storage/train_512_RenderPeople_all_sparse')
	#parser.add_argument('--output', type=str, default='/mars/mnt/oitstorage/emily_storage/temporal_dvr_results/')
	parser.add_argument('--input', type=str, default='/media/mulha024/i/train_512_RenderPeople_all_sparse')
	parser.add_argument('--output', type=str, default='./results')	
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-4)
	#parser.add_argument('--epochs', type=int, default=10000000000000000)
	parser.add_argument('--epochs', type=int, default=151)
	parser.add_argument('--num_samples',type=int, default=5000)
	parser.add_argument('--name', type=str, default='pretrain')
	parser.add_argument('--scale',type=float, default=0.1)
	parser.add_argument('--random_multiview',action='store_true')
	parser.add_argument('--resume_epoch', type=int, default=0)
	parser.add_argument('--continue_train', action='store_true')
	parser.add_argument('--checkpoint_path', type=str, default='')

	return parser.parse_args()


if __name__ == '__main__':
	args=ParseCmdLineArguments()


	#Set up dataloaders
	'''train_dataset = RenderPeopleDataset(args.input,
		usage='train',
		num_samples=args.num_samples,
		scale=args.scale,
		random_multiview=args.random_multiview)'''

	val_dataset = HUMBIDataset(args.input,
		usage='val',
		num_samples=args.num_samples,
		scale=args.scale,
		random_multiview=args.random_multiview)

	'''train_dataloader = DataLoader(train_dataset, shuffle=True,
		batch_size=args.batch_size,
		num_workers=4)'''

	val_dataloader = DataLoader(val_dataset, shuffle=True,
		batch_size=args.batch_size,
		num_workers=4)


	is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda" if is_cuda else "cpu")


	#Initialize training
	net = get_model(device)
	optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate)

	if args.checkpoint_path != '':
		net.load_state_dict(torch.load(args.checkpoint_path))
		print("Checkpoint loaded")	

	net.eval()

	'''IOU, prec, recall = calc_error(net, device, val_dataset, 100)
	print('IOU: %f prec: %f recall: %f'%(IOU, prec, recall))	

	IOU, prec, recall = calc_error(net, device, train_dataset, 100)
	print('IOU: %f prec: %f recall: %f'%(IOU, prec, recall))'''

	data = random.choice(val_dataset)
	color = data['color'].to(device)
	mask = data['mask'].to(device)
	K = data['K'].to(device) 
	R = data['R'].to(device) 
	C = data['C'].to(device)
	origin = data['origin'].to(device)
	scaling = data['scaling'].to(device)
	bb_min = torch.Tensor(data['bb_min'])
	bb_max = torch.Tensor(data['bb_max'])
	#save_path = os.path.join(args.output, args.name, 'meshes','final_subject_%s_view_%s'%(data['name'], data['yid']))
	save_path = os.path.join(args.output, args.name, 'meshes','HUMBI_%s_view_%s'%(data['name'], data['yid']))

	generate_mesh(net,device,color, K, R, C, scaling, origin,mask,bb_min,bb_max,fname=save_path)