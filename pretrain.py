import argparse
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import distributions as dist

from dataset import RenderPeopleDataset
from config import get_model
from utils import sample, patch_sample, get_freespace_points, get_occupancy_points

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	parser.add_argument('--input', type=str, default='./data/RenderPeople.pkl')
	parser.add_argument('--output', type=str, default='./results')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-3)
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--num_sample_points',type=int, default=10)
	parser.add_argument('--sample_batch_size',type=int, default=100)

	return parser.parse_args()


if __name__ == '__main__':
	args=ParseCmdLineArguments()


	#Set up dataloaders
	train_dataset = RenderPeopleDataset(usage='train',
		pickle_file=args.input)

	val_dataset = RenderPeopleDataset(usage='val',
		pickle_file=args.input)

	train_dataloader = DataLoader(train_dataset, shuffle=True,
		batch_size=args.batch_size,
		num_workers=0)

	val_dataloader = DataLoader(val_dataset, shuffle=True,
		batch_size=args.batch_size,
		num_workers=0)

	#Set up directories to save results to
	if not (os.path.isdir(args.output)):
		os.mkdir(args.output)

	if not (os.path.isdir(os.path.join(args.output,'pretrain'))):
		os.mkdir(os.path.join(args.output,'pretrain'))

	if not (os.path.isdir(os.path.join(args.output,'pretrain','plots'))):
		os.mkdir(os.path.join(args.output,'pretrain','plots'))

	if not (os.path.isdir(os.path.join(args.output,'pretrain','meshes'))):
		os.mkdir(os.path.join(args.output,'pretrain','meshes'))

	is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda" if is_cuda else "cpu")


	#Initialize training
	net = get_model(device)
	optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate)

	net.train()
	for epoch in range(args.epochs):

		for train_idx, train_data in enumerate(train_dataloader):
			#Retrieve a batch of RenderPeople data
			color = train_data['color'].cuda(non_blocking=True) 
			dp = train_data['dp'].cuda(non_blocking=True) 
			mask = train_data['mask'].cuda(non_blocking=True) 
			K = train_data['K'].cuda(non_blocking=True) 
			R = train_data['R'].cuda(non_blocking=True) 
			C = train_data['C'].cuda(non_blocking=True) 
			gt_depth = train_data['gt_depth'].cuda(non_blocking=True) 
			gt_norm = train_data['gt_norm'].cuda(non_blocking=True) 
			origin = train_data['origin'].cuda(non_blocking=True)
			scaling = train_data['scaling'].cuda(non_blocking=True)

			#Sample pixels
			#TODO maybe play around with more random sampling?
			p = None
			if args.num_sample_points >=color.shape[2]*color.shape[3]:
				p_unscaled, p = sample(color,color.shape[0])
			else:
				p_unscaled, p = patch_sample(color, color.shape[0], args.num_sample_points) #(batch_size, n_points, 2)

			p=p.cuda(non_blocking=True)
			p_unscaled=p_unscaled.cuda(non_blocking=True)


			#Get the ground truth mask for the sampled points
			gt_mask = mask[torch.arange(mask.shape[0]).unsqueeze(-1),:,p_unscaled[...,1].long(),p_unscaled[...,0].long()] # (B,n_points,3)
			gt_mask_1D = gt_mask[...,0]

			#Get the 3D points for evaluating occupancy and freespace losses as mentioned in DVR paper
			p_freespace = get_freespace_points(p, K, R, C, origin, scaling, depth_range=[0,2.4]) #(B,n_points,3)
			p_occupancy = get_occupancy_points(p, p_unscaled,K, R, C, origin, scaling, gt_depth) #(B,n_points,3)


			#Forward pass
			p_world_hat, rgb_pred, logits_occupancy, logits_freespace, mask_pred, p_world_hat_sparse, mask_pred_sparse, normals=net(p, p_occupancy,p_freespace,color,K,R,C,origin,scaling)

			#Calculate loss
			#Photoconsistency loss
			mask_rgb = mask_pred & gt_mask
			




