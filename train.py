import argparse
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import distributions as dist
import math

from dataset import HUMBIDataset
from config import get_model
from utils import sample, sample_n, get_freespace_points, get_occupancy_points, get_tensor_values, save_3D, sample_correspondences, get_mask_points
from loss import calculate_photoconsistency_loss, calculate_depth_loss, calculate_normal_loss, calculate_freespace_loss, calculate_occupancy_loss, calculate_temporal_photoconsistency_loss, calculate_temporal_loss, calculate_mask_loss
from geometry import depth_to_3D
from view_3D import visualize_3D_masked, visualize_3D
from render import render_img
from generate import generate_mesh_unsupervised

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	parser.add_argument('--input', type=str, default='./data/HUMBI_dvr.pkl')
	parser.add_argument('--output', type=str, default='./results')
	#parser.add_argument('--input', type=str, default='/mars/mnt/oitstorage/emily_storage/pickles/RenderPeople_dvr.pkl')
	#parser.add_argument('--output', type=str, default='/mars/mnt/oitstorage/emily_storage/temporal_dvr_results/')	
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-4)
	parser.add_argument('--epochs', type=int, default=101)
	parser.add_argument('--num_sample_points',type=int, default=1000)

	return parser.parse_args()


if __name__ == '__main__':
	args=ParseCmdLineArguments()


	#Set up dataloaders
	train_dataset = HUMBIDataset(usage='train',
		pickle_file=args.input)

	train_dataloader = DataLoader(train_dataset, shuffle=True,
		batch_size=args.batch_size,
		num_workers=0)

	#Set up directories to save results to
	if not (os.path.isdir(args.output)):
		os.mkdir(args.output)

	if not (os.path.isdir(os.path.join(args.output,'plots'))):
		os.mkdir(os.path.join(args.output,'plots'))

	if not (os.path.isdir(os.path.join(args.output,'meshes'))):
		os.mkdir(os.path.join(args.output,'meshes'))

	if not (os.path.isdir(os.path.join(args.output,'color'))):
		os.mkdir(os.path.join(args.output,'color'))
	
	if not (os.path.isdir(os.path.join(args.output,'models'))):
		os.mkdir(os.path.join(args.output,'models'))

	is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda" if is_cuda else "cpu")

	#Set lambda values
	lambda_rgb=1
	lambda_depth=1
	lambda_normal=0.5
	lambda_freespace=0.25
	lambda_occupancy=1
	lambda_mask=1
	lambda_temporal=0
	lambda_temp_rgb=1

	#Set hyperparameters
	padding=1e-3

	#Set up loss lists for plots
	loss_list={'loss':[], 'loss_rgb':[], 'loss_depth':[], 'loss_normal':[],  'loss_freespace':[], 'loss_occupancy':[], 'loss_temporal_rgb':[], 'loss_temporal':[],'loss_mask':[]}

	#Initialize training
	net = get_model(device)
	net.load_state_dict(torch.load('./models/net_epoch_000000_iter_0001500.cpkt'))
	optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate)

	net.train()
	for epoch in range(args.epochs):

		for train_idx, train_data in enumerate(train_dataloader):
			#Retrieve a batch of RenderPeople data
			color_1 = train_data['color_1'].cuda(non_blocking=True)
			color_2 = train_data['color_2'].cuda(non_blocking=True)			 
			mask_1 = train_data['mask_1'].cuda(non_blocking=True)
			mask_2 = train_data['mask_2'].cuda(non_blocking=True)			 
			K = train_data['K'].cuda(non_blocking=True)		 
			R = train_data['R'].cuda(non_blocking=True)		 
			C = train_data['C'].cuda(non_blocking=True)			
			origin_1 = train_data['origin_1'].cuda(non_blocking=True)
			origin_2 = train_data['origin_2'].cuda(non_blocking=True)			
			scaling_1 = train_data['scaling_1'].cuda(non_blocking=True)
			scaling_2 = train_data['scaling_2'].cuda(non_blocking=True)
			corr_1 = train_data['iuv_1'].cuda(non_blocking=True)
			corr_2 = train_data['iuv_2'].cuda(non_blocking=True)
			dp_1 = train_data['dp_1'].cuda(non_blocking=True)
			dp_2 = train_data['dp_1'].cuda(non_blocking=True)			

			#Sample pixels
			p_total = sample(mask_1) #(batch_size, n_points, 2)

			num_batches = math.ceil(p_total.shape[1]/args.num_sample_points)

			for n in range(num_batches):
				#Initialize loss dictionary
				loss={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_normal':0,  'loss_freespace':0, 'loss_occupancy':0, 'loss_temporal_rgb': 0, 'loss_temporal':0, 'loss_mask':0}

				#Get a sample of pixels
				p=p_total[:,n*args.num_sample_points:n*args.num_sample_points+args.num_sample_points,:].to(device)

				#Get a sample of correspondences
				p_1, p_2 = sample_correspondences(corr_1, corr_2,args.num_sample_points)
				p_1=p_1.to(device)
				p_2=p_2.to(device)

				iuv_1 = get_tensor_values(dp_1, p_1)
				iuv_2 = get_tensor_values(dp_2, p_2)

				#Get the ground truth mask for the sampled points
				gt_mask = get_tensor_values(mask_1,p)[...,0]

				#Get the 3D points for evaluating occupancy and freespace losses as mentioned in DVR paper
				p_freespace = get_freespace_points(p, K, R, C, origin_1, scaling_1) #(B,n_points,3)
				p_occupancy = get_occupancy_points(p, K, R, C, origin_1, scaling_1) #(B,n_points,3)
				p_mask = get_mask_points(mask_1, K, R, C, origin_1, scaling_1,int(p.shape[1]))

				#visualize_3D(p_occupancy.squeeze(0).cpu().numpy(),fname='occupancy.ply')
				#visualize_3D(p_freespace.squeeze(0).cpu().numpy(),fname='freespace.ply')

				#Forward pass
				p_world_hat, rgb_pred, logits_occupancy, logits_freespace, logits_mask,mask_pred, normals, p_world_1, mask_pred_1, p_world_2, mask_pred_2, rgb_pred_1, rgb_pred_2=net(p, p_occupancy,p_freespace,p_mask,color_1,K,R,C,origin_1,scaling_1,p_temporal_1=p_1, p_temporal_2=p_2, origin_2=origin_2, scale_2=scaling_2, inputs_2=color_2)
				#visualize_3D(p_world_hat.detach().squeeze(0).cpu().numpy(),fname='p_world_hat%03d.ply'%(epoch))

				#Calculate loss
				#Photoconsistency loss
				mask_rgb = mask_pred & gt_mask.bool()
				calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, color_1, p, 'sum', loss)

				#Normal loss
				calculate_normal_loss(lambda_normal,normals, color_1.shape[0], loss)

				#Freespace loss
				calculate_freespace_loss(lambda_freespace,logits_freespace, 'sum', loss)

				#Occupancy loss
				calculate_occupancy_loss(lambda_occupancy,logits_occupancy, 'sum',loss)


				#Mask loss
				calculate_mask_loss(lambda_mask, logits_mask, 'sum',loss)

				#Temporal RGB loss
				calculate_temporal_photoconsistency_loss(lambda_temp_rgb, mask_pred_1, mask_pred_2, rgb_pred_1, rgb_pred_2, 'sum',loss)

				#Temporal loss
				calculate_temporal_loss(lambda_temporal, iuv_1, iuv_2, mask_pred_1, mask_pred_2, p_world_1, p_world_2,'sum',loss)

				#Update the loss lists
				loss_list['loss'].append(loss['loss'].item())
				try:
					loss_list['loss_rgb'].append(loss['loss_rgb'].item())
				except:
					loss_list['loss_rgb'].append(loss['loss_rgb'])
				try:
					loss_list['loss_depth'].append(loss['loss_depth'].item())
				except:
					loss_list['loss_depth'].append(loss['loss_depth'])
				try:
					loss_list['loss_normal'].append(loss['loss_normal'].item())
				except:
					loss_list['loss_normal'].append(loss['loss_normal'])
				try:
					loss_list['loss_freespace'].append(loss['loss_freespace'].item())
				except:
					loss_list['loss_freespace'].append(loss['loss_freespace'])
				try:
					loss_list['loss_occupancy'].append(loss['loss_occupancy'].item())
				except:
					loss_list['loss_occupancy'].append(loss['loss_occupancy'])
				try:
					loss_list['loss_temporal_rgb'].append(loss['loss_temporal_rgb'].item())
				except:
					loss_list['loss_temporal_rgb'].append(loss['loss_temporal_rgb'])
				try:
					loss_list['loss_temporal'].append(loss['loss_temporal'].item())
				except:
					loss_list['loss_temporal'].append(loss['loss_temporal'])
				try:
					loss_list['loss_mask'].append(loss['loss_mask'].item())
				except:
					loss_list['loss_mask'].append(loss['loss_mask'])	

				total_loss = loss['loss']
				total_loss.backward()
				optimizer.step()
				optimizer.zero_grad()



				if train_idx%1==0 and n==0 or (epoch==0 and train_idx==0 and n==0):
					print('epoch: %d, iter: %d, n: %d,rgb loss: %2.4f, normal loss: %2.4f, freespace loss: %2.4f, occupancy loss: %2.4f, mask loss: %2.4f, temporal rgb loss: %2.4f, total loss: %2.4f'%(epoch, train_idx, n, loss['loss_rgb'],
						loss['loss_normal'], loss['loss_freespace'], loss['loss_occupancy'],loss['loss_mask'],loss['loss_temporal_rgb'],loss['loss'].item()))

				if train_idx%1==0 and n==0 or (epoch==0 and train_idx==0 and n==0):
					with torch.no_grad():
						#Plot the error
						plot_path=os.path.join(args.output,'plots','error.png')

						val_plot_path=os.path.join(args.output, 'plots','validation_error.png')

						if sum(loss_list['loss'])>0:
							plt.plot(loss_list['loss'], label='Total loss')
						if sum(loss_list['loss_rgb'])>0:
							plt.plot(loss_list['loss_rgb'], label='RGB loss')
						if sum(loss_list['loss_normal'])>0:
							plt.plot(loss_list['loss_normal'], label='Normal loss')
						if sum(loss_list['loss_freespace'])>0:
							plt.plot(loss_list['loss_freespace'], label='Freespace loss')
						if sum(loss_list['loss_occupancy'])>0:
							plt.plot(loss_list['loss_occupancy'], label='Occupancy loss')
						if sum(loss_list['loss_temporal_rgb'])>0:
							plt.plot(loss_list['loss_temporal_rgb'], label='Temporal RGB loss')
						if sum(loss_list['loss_temporal'])>0:
							plt.plot(loss_list['loss_temporal'], label='Temporal loss')
						if sum(loss_list['loss_mask'])>0:
							plt.plot(loss_list['loss_mask'], label='Mask loss')								


						plt.xlabel("Iterations")
						plt.ylabel("Error")
						plt.title("Error over Iterations")
						plt.legend()
						plt.savefig(plot_path)
						plt.clf()

				
				if (epoch%10==0 and train_idx==0 and n==0):
					with torch.no_grad():
						#Save a ground truth point cloud for reference
						if mask_rgb.sum() >0:					
							generate_mesh_unsupervised(net,device,color_1,p_world_hat,mask_rgb,padding=2.,fname=os.path.join(args.output, 'meshes','pred_epoch_%06d_iter_%07d'%(epoch,train_idx)))
							plt.imsave(os.path.join(args.output, 'color','pred_epoch_%06d_iter_%07d.png'%(epoch,train_idx)), color_1[0].permute(1,2,0).cpu().numpy())
						else:
							print("Not enough points to generate mesh")

						#Save a model
						model_path = os.path.join(args.output,'pretrain','models','net_epoch_%06d_iter_%07d.cpkt'%(epoch,train_idx))
						torch.save(net.state_dict(),model_path)








