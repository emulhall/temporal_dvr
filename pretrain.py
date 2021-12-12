import argparse
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import distributions as dist
import math

from dataset import RenderPeopleDataset
from config import get_model
from utils import sample, sample_n, get_freespace_points, get_occupancy_points, get_tensor_values
from loss import calculate_photoconsistency_loss, calculate_depth_loss, calculate_normal_loss, calculate_freespace_loss, calculate_occupancy_loss
from geometry import depth_to_3D
from view_3D import visualize_3D_masked, visualize_3D, visualize_prediction
from render import render_img
from generate import generate_mesh

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	parser.add_argument('--input', type=str, default='./data/RenderPeople.pkl')
	parser.add_argument('--output', type=str, default='./results')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-4)
	parser.add_argument('--epochs', type=int, default=51)
	parser.add_argument('--num_sample_points',type=int, default=1000)

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

	if not (os.path.isdir(os.path.join(args.output,'pretrain','render'))):
		os.mkdir(os.path.join(args.output,'pretrain','render'))	

	is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda" if is_cuda else "cpu")

	#Set lambda values
	lambda_rgb=1
	lambda_depth=1
	lambda_normal=0.5
	lambda_freespace=1
	lambda_occupancy=1

	#Set hyperparameters
	padding=1e-3

	#Set up loss lists for plots
	loss_list={'loss':[], 'loss_rgb':[], 'loss_depth':[], 'loss_normal':[],  'loss_freespace':[], 'loss_occupancy':[]}
	loss_val_list={'loss':[], 'loss_rgb':[], 'loss_depth':[], 'loss_normal':[],  'loss_freespace':[], 'loss_occupancy':[]}


	#Initialize training
	net = get_model(device)
	optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate)

	net.train()
	for epoch in range(args.epochs):

		val_iterator = iter(val_dataloader)

		for train_idx, train_data in enumerate(train_dataloader):
			#Retrieve a batch of RenderPeople data
			color = train_data['color'].cuda(non_blocking=True) 
			mask = train_data['mask'].cuda(non_blocking=True) 
			K = train_data['K'].cuda(non_blocking=True) 
			R = train_data['R'].cuda(non_blocking=True) 
			C = train_data['C'].cuda(non_blocking=True) 
			gt_depth = train_data['gt_depth'].cuda(non_blocking=True)
			origin = train_data['origin'].cuda(non_blocking=True)
			scaling = train_data['scaling'].cuda(non_blocking=True)

			#Sample pixels
			p_total = sample(mask) #(batch_size, n_points, 2)

			num_batches = math.ceil(p_total.shape[1]/args.num_sample_points)

			for n in range(num_batches):
				#Initialize loss dictionary
				loss={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_normal':0,  'loss_freespace':0, 'loss_occupancy':0}
				loss_val={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_normal':0,  'loss_freespace':0, 'loss_occupancy':0}

				#Get a sample of pixels
				p=p_total[:,n*args.num_sample_points:n*args.num_sample_points+args.num_sample_points,:].to(device)


				#Get the ground truth mask for the sampled points
				gt_mask = get_tensor_values(mask,p)[...,0]

				#Get the 3D points for evaluating occupancy and freespace losses as mentioned in DVR paper
				p_freespace = get_freespace_points(p, K, R, C, origin, scaling, depth_img=gt_depth) #(B,n_points,3)
				p_occupancy = get_occupancy_points(p, K, R, C, origin, scaling, depth_img=gt_depth) #(B,n_points,3)

				#visualize_3D(p_occupancy.squeeze(0).cpu().numpy(),fname='occupancy.ply')
				#visualize_3D(p_freespace.squeeze(0).cpu().numpy(),fname='freespace.ply')

				#Forward pass
				depth_range=[float(torch.min(gt_depth[gt_depth>0]))-padding,float(torch.max(gt_depth[gt_depth>0]))+padding]
				p_world_hat, rgb_pred, logits_occupancy, logits_freespace, mask_pred, normals=net(p, p_occupancy,p_freespace,color,K,R,C,origin,scaling,depth_range=depth_range)
				#visualize_3D(p_world_hat.detach().squeeze(0).cpu().numpy(),fname='p_world_hat%03d.ply'%(epoch))

				#Calculate loss
				#Photoconsistency loss
				mask_rgb = mask_pred & gt_mask.bool()
				calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, color, p, 'sum', loss)

				#Depth loss
				mask_depth = mask_pred & gt_mask.bool()
				calculate_depth_loss(lambda_depth,mask_depth, gt_depth, p, K, R, C, origin, scaling, p_world_hat, 'sum', loss)

				#Normal loss
				calculate_normal_loss(lambda_normal,normals, color.shape[0], loss)

				#Freespace loss
				calculate_freespace_loss(lambda_freespace,logits_freespace, 'sum', loss)

				#Occupancy loss
				calculate_occupancy_loss(lambda_occupancy,logits_occupancy, 'sum',loss)

				#Update the loss lists
				loss_list['loss'].append(loss['loss'].item())
				loss_list['loss_rgb'].append(loss['loss_rgb'])
				loss_list['loss_depth'].append(loss['loss_depth'])
				loss_list['loss_normal'].append(loss['loss_normal'])
				loss_list['loss_freespace'].append(loss['loss_freespace'])
				loss_list['loss_occupancy'].append(loss['loss_occupancy'])			


				total_loss = loss['loss']
				total_loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				#Test validation
				try:
					val_data = next(val_iterator)
				except StopIteration:
					val_iterator = iter(val_dataloader)
					val_data = next(val_iterator)

				color = val_data['color'].cuda(non_blocking=True) 
				mask = val_data['mask'].cuda(non_blocking=True) 
				K = val_data['K'].cuda(non_blocking=True) 
				R = val_data['R'].cuda(non_blocking=True) 
				C = val_data['C'].cuda(non_blocking=True) 
				gt_depth = val_data['gt_depth'].cuda(non_blocking=True)
				origin = val_data['origin'].cuda(non_blocking=True)
				scaling = val_data['scaling'].cuda(non_blocking=True)

				p = sample_n(mask, args.num_sample_points).to(device)

				gt_mask = get_tensor_values(mask,p)[...,0]

				#Get the 3D points for evaluating occupancy and freespace losses as mentioned in DVR paper
				p_freespace = get_freespace_points(p, K, R, C, origin, scaling, depth_img=gt_depth) #(B,n_points,3)
				p_occupancy = get_occupancy_points(p, K, R, C, origin, scaling, depth_img=gt_depth) #(B,n_points,3)

				depth_range=[float(torch.min(gt_depth[gt_depth>0]))-padding,float(torch.max(gt_depth[gt_depth>0]))+padding]

				with torch.no_grad():
					p_world_hat, rgb_pred, logits_occupancy, logits_freespace, mask_pred, normals=net(p, p_occupancy,p_freespace,color,K,R,C,origin,scaling,depth_range=depth_range)

					mask_rgb = mask_pred & gt_mask.bool()
					calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, color, p, 'mean', loss_val)

					#Depth loss
					mask_depth = mask_pred & gt_mask.bool()
					calculate_depth_loss(lambda_depth,mask_depth, gt_depth, p, K, R, C, origin, scaling, p_world_hat, 'mean', loss_val)

					#Normal loss
					calculate_normal_loss(lambda_normal,normals, color.shape[0], loss_val)

					#Freespace loss
					calculate_freespace_loss(lambda_freespace,logits_freespace, 'mean', loss_val)

					#Occupancy loss
					calculate_occupancy_loss(lambda_occupancy,logits_occupancy, 'mean',loss_val)

					#Update the loss lists
					loss_val_list['loss'].append(loss_val['loss'].item())
					loss_val_list['loss_rgb'].append(loss_val['loss_rgb'])
					loss_val_list['loss_depth'].append(loss_val['loss_depth'])
					loss_val_list['loss_normal'].append(loss_val['loss_normal'])
					loss_val_list['loss_freespace'].append(loss_val['loss_freespace'])
					loss_val_list['loss_occupancy'].append(loss_val['loss_occupancy'])



				if epoch%1==0 and train_idx%1==0 and n%10==0 or (epoch==0 and train_idx==0 and n==0):
					print('epoch: %d, iter: %d, n: %d,rgb loss: %2.4f, depth loss: %2.4f, normal loss: %2.4f, freespace loss: %2.4f, occupancy loss: %2.4f, total loss: %2.4f'%(epoch, train_idx, n, loss['loss_rgb'],
						loss['loss_depth'], loss['loss_normal'], loss['loss_freespace'], loss['loss_occupancy'],loss['loss'].item()))
					with torch.no_grad():
						#Plot the error
						plot_path=os.path.join(args.output, 'pretrain','plots','pretraining_error.png')

						val_plot_path=os.path.join(args.output, 'pretrain','plots','pretraining_validation_error.png')

						if sum(loss_list['loss'])>0:
							plt.plot(loss_list['loss'], label='Total loss')
						if sum(loss_list['loss_rgb'])>0:
							plt.plot(loss_list['loss_rgb'], label='RGB loss')
						if sum(loss_list['loss_depth'])>0:
							plt.plot(loss_list['loss_depth'], label='Depth loss')
						if sum(loss_list['loss_normal'])>0:
							plt.plot(loss_list['loss_normal'], label='Normal loss')
						if sum(loss_list['loss_freespace'])>0:
							plt.plot(loss_list['loss_freespace'], label='Freespace loss')
						if sum(loss_list['loss_occupancy'])>0:
							plt.plot(loss_list['loss_occupancy'], label='Occupancy loss')

						plt.xlabel("Iterations")
						plt.ylabel("Error")
						plt.title("Error over Iterations")
						plt.legend()
						plt.savefig(plot_path)
						plt.clf()

						if sum(loss_val_list['loss'])>0:
							plt.plot(loss_val_list['loss'], label='Total loss')
						if sum(loss_val_list['loss_rgb'])>0:
							plt.plot(loss_val_list['loss_rgb'], label='RGB loss')
						if sum(loss_val_list['loss_depth'])>0:
							plt.plot(loss_val_list['loss_depth'], label='Depth loss')
						if sum(loss_val_list['loss_normal'])>0:
							plt.plot(loss_val_list['loss_normal'], label='Normal loss')
						if sum(loss_val_list['loss_freespace'])>0:
							plt.plot(loss_val_list['loss_freespace'], label='Freespace loss')
						if sum(loss_val_list['loss_occupancy'])>0:
							plt.plot(loss_val_list['loss_occupancy'], label='Occupancy loss')

						plt.xlabel("Iterations")
						plt.ylabel("Error")
						plt.title("Error over Iterations")
						plt.legend()
						plt.savefig(val_plot_path)
						plt.clf()


				if (epoch%5==0 and train_idx%1==0 and n%10==0 and n!=0) or (epoch==0 and train_idx==0 and n==0):
					with torch.no_grad():
						#Save a ground truth point cloud for reference
						X,_ = depth_to_3D(gt_depth, K, R, C, scaling, origin, 1)
						visualize_3D_masked(X[0].cpu().numpy(), mask[0,0].cpu().numpy(),fname=os.path.join(args.output, 'pretrain', 'meshes','gt_epoch_%04d_iter_%04d_%04d.ply'%(epoch, train_idx,n)))
						generate_mesh(net,device,color,X,mask,fname=os.path.join(args.output, 'pretrain', 'meshes','pred_epoch_%04d_iter_%04d_%04d'%(epoch, train_idx,n)))









