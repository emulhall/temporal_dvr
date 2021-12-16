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
from utils import sample, sample_n, get_freespace_points, get_occupancy_points, get_tensor_values, save_3D, get_mask_points
from loss import calculate_photoconsistency_loss, calculate_depth_loss, calculate_normal_loss, calculate_freespace_loss, calculate_occupancy_loss, calculate_mask_loss
from geometry import depth_to_3D
from view_3D import visualize_3D_masked, visualize_3D
from render import render_img
from generate import generate_mesh

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	parser.add_argument('--input', type=str, default='./data/RenderPeople.pkl')
	parser.add_argument('--output', type=str, default='./results')
	#parser.add_argument('--input', type=str, default='/mars/mnt/oitstorage/emily_storage/pickles/RenderPeople_dvr.pkl')
	#parser.add_argument('--output', type=str, default='/mars/mnt/oitstorage/emily_storage/temporal_dvr_results/')	
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-4)
	#parser.add_argument('--epochs', type=int, default=10000000000000000)
	#parser.add_argument('--num_sample_points',type=int, default=10000)
	parser.add_argument('--epochs', type=int, default=1001)
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

	if not (os.path.isdir(os.path.join(args.output,'pretrain','models'))):
		os.mkdir(os.path.join(args.output,'pretrain','models'))	

	is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda" if is_cuda else "cpu")

	#Set lambda values
	lambda_rgb=1
	lambda_depth=1
	lambda_normal=0.5
	lambda_freespace=0.25
	lambda_occupancy=1
	lambda_mask=1

	#Set hyperparameters
	padding=1e-3

	#Set up loss lists for plots
	loss_list={'loss':[], 'loss_rgb':[], 'loss_depth':[], 'loss_normal':[],  'loss_freespace':[], 'loss_occupancy':[], 'loss_mask':[]}
	loss_val_list={'loss':[], 'loss_rgb':[], 'loss_depth':[], 'loss_normal':[],  'loss_freespace':[], 'loss_occupancy':[],'loss_mask':[]}


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
				loss={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_normal':0,  'loss_freespace':0, 'loss_occupancy':0, 'loss_mask':0}
				loss_val={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_normal':0,  'loss_freespace':0, 'loss_occupancy':0,'loss_mask':0}

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
				p_mask = get_mask_points(mask, K, R, C, origin, scaling,int(p.shape[1]))
				#visualize_3D(p_mask.squeeze(0).cpu().numpy(),fname='mask.ply')

				p_world_hat, rgb_pred, logits_occupancy, logits_freespace, logits_mask,mask_pred, normals,_,_,_,_,_,_=net(p, p_occupancy,p_freespace,p_mask,color,K,R,C,origin,scaling)
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

				#Mask loss
				calculate_mask_loss(lambda_mask, logits_mask, 'sum',loss)

				#Update the loss lists
				#if train_idx%10==0 and n==0 or (epoch==0 and train_idx==0 and n==0):
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
					loss_list['loss_mask'].append(loss['loss_mask'].item())
				except:
					loss_list['loss_mask'].append(loss['loss_mask'])


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

				color_val = val_data['color'].cuda(non_blocking=True) 
				mask_val = val_data['mask'].cuda(non_blocking=True) 
				K_val = val_data['K'].cuda(non_blocking=True) 
				R_val = val_data['R'].cuda(non_blocking=True) 
				C_val = val_data['C'].cuda(non_blocking=True) 
				gt_depth_val = val_data['gt_depth'].cuda(non_blocking=True)
				origin_val = val_data['origin'].cuda(non_blocking=True)
				scaling_val = val_data['scaling'].cuda(non_blocking=True)

				p = sample_n(mask_val, args.num_sample_points).to(device)

				gt_mask = get_tensor_values(mask_val,p)[...,0]

				#Get the 3D points for evaluating occupancy and freespace losses as mentioned in DVR paper
				p_freespace = get_freespace_points(p, K_val, R_val, C_val, origin_val, scaling_val, depth_img=gt_depth_val) #(B,n_points,3)
				p_occupancy = get_occupancy_points(p, K_val, R_val, C_val, origin_val, scaling_val, depth_img=gt_depth_val) #(B,n_points,3)
				p_mask = get_mask_points(mask_val, K_val, R_val, C_val, origin_val, scaling_val, int(args.num_sample_points/2))

				depth_range=[float(torch.min(gt_depth_val[gt_depth_val>0]))-padding,float(torch.max(gt_depth_val[gt_depth_val>0]))+padding]

				with torch.no_grad():
					p_world_hat, rgb_pred, logits_occupancy, logits_freespace, logits_mask,mask_pred, normals,_,_,_,_,_,_=net(p, p_occupancy,p_freespace,p_mask,color_val,K_val,R_val,C_val,origin_val,scaling_val)

					mask_rgb = mask_pred & gt_mask.bool()
					calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, color_val, p, 'mean', loss_val)

					#Depth loss
					mask_depth = mask_pred & gt_mask.bool()
					calculate_depth_loss(lambda_depth,mask_depth, gt_depth_val, p, K_val, R_val, C_val, origin_val, scaling_val, p_world_hat, 'mean', loss_val)

					#Normal loss
					calculate_normal_loss(lambda_normal,normals, color_val.shape[0], loss_val)

					#Freespace loss
					calculate_freespace_loss(lambda_freespace,logits_freespace, 'mean', loss_val)

					#Occupancy loss
					calculate_occupancy_loss(lambda_occupancy,logits_occupancy, 'mean',loss_val)

					#Mask loss
					calculate_mask_loss(lambda_mask, logits_mask, 'mean', loss_val)

					#Update the loss lists
					#if train_idx%10==0 and n==0 or (epoch==0 and train_idx==0 and n==0):
					loss_val_list['loss'].append(loss_val['loss'].item())
					try:
						loss_val_list['loss_rgb'].append(loss_val['loss_rgb'].item())
					except:
						loss_val_list['loss_rgb'].append(loss_val['loss_rgb'])
					try:
						loss_val_list['loss_depth'].append(loss_val['loss_depth'].item())
					except:
						loss_val_list['loss_depth'].append(loss_val['loss_depth'])
					try:
						loss_val_list['loss_normal'].append(loss_val['loss_normal'].item())
					except:
						loss_val_list['loss_normal'].append(loss_val['loss_normal'])
					try:
						loss_val_list['loss_freespace'].append(loss_val['loss_freespace'].item())
					except:
						loss_val_list['loss_freespace'].append(loss_val['loss_freespace'])
					try:
						loss_val_list['loss_occupancy'].append(loss_val['loss_occupancy'].item())
					except:
						loss_val_list['loss_occupancy'].append(loss_val['loss_occupancy'])
					try:
						loss_val_list['loss_mask'].append(loss['loss_mask'].item())
					except:
						loss_val_list['loss_mask'].append(loss['loss_mask'])



				if train_idx%5==0 and n==0 or (epoch==0 and train_idx==0 and n==0):
					print('epoch: %d, iter: %d, n: %d,rgb loss: %2.4f, depth loss: %2.4f, normal loss: %2.4f, freespace loss: %2.4f, occupancy loss: %2.4f, mask loss: %2.4f, total loss: %2.4f'%(epoch, train_idx, n, loss['loss_rgb'],
						loss['loss_depth'], loss['loss_normal'], loss['loss_freespace'], loss['loss_occupancy'],loss['loss_mask'],loss['loss'].item()))

				if train_idx%10==0 and n==0 or (epoch==0 and train_idx==0 and n==0):
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
						if sum(loss_list['loss_mask'])>0:
							plt.plot(loss_list['loss_mask'], label='Mask loss')							

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
						if sum(loss_val_list['loss_mask'])>0:
							plt.plot(loss_val_list['loss_mask'], label='Mask loss')									

						plt.xlabel("Iterations")
						plt.ylabel("Error")
						plt.title("Error over Iterations")
						plt.legend()
						plt.savefig(val_plot_path)
						plt.clf()

				
				#if (train_idx%100==0 and n==0):
				if (epoch%10==0 and n==0):
					with torch.no_grad():
						#Save a ground truth point cloud for reference
						X,_ = depth_to_3D(gt_depth, K, R, C, scaling, origin, 1)
						visualize_3D_masked(X[0].cpu().numpy(), mask[0,0].cpu().numpy(),fname=os.path.join(args.output, 'pretrain', 'meshes','gt_epoch_%06d_iter_%07d.ply'%(epoch, train_idx)))
						
						generate_mesh(net,device,color,X,mask,fname=os.path.join(args.output, 'pretrain', 'meshes','pred_epoch_%06d_iter_%07d'%(epoch,train_idx)))

						#Save a model
						#model_path = os.path.join(args.output,'pretrain','models','net_epoch_%06d_iter_%07d.cpkt'%(epoch,train_idx))
						#torch.save(net.state_dict(),model_path)








