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
from utils import sample, patch_sample, get_freespace_points, get_occupancy_points, get_tensor_values, rescale_origin
from loss import calculate_photoconsistency_loss, calculate_depth_loss, calculate_normal_loss, calculate_freespace_loss, calculate_occupancy_loss
from geometry import depth_to_3D
from view_3D import visualize_3D_masked, visualize_3D, visualize_prediction
from render import render_img

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	parser.add_argument('--input', type=str, default='./data/RenderPeople.pkl')
	parser.add_argument('--output', type=str, default='./results')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-4)
	parser.add_argument('--epochs', type=int, default=100)
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
	lambda_image_gradients=0
	lambda_depth=1
	lambda_normal=0
	lambda_freespace=1
	lambda_occupancy=1

	#Set hyperparameters
	patch_size=1

	#Set up loss lists for plots
	loss_list={'loss':[], 'loss_rgb':[], 'loss_image_gradient':[], 'loss_rgb_eval': [], 'loss_depth':[], 'loss_depth_eval':[], 'loss_normal':[], 'loss_normal_eval':[], 'loss_freespace':[], 'loss_occupancy':[]}


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

			#Initialize loss dictionary
			loss={'loss':0, 'loss_rgb':0, 'loss_image_gradient':0, 'loss_rgb_eval': 0, 'loss_depth':0, 'loss_depth_eval':0, 'loss_normal':0, 'loss_normal_eval':0, 'loss_freespace':0, 'loss_occupancy':0}

			#Sample pixels
			p = None
			if args.num_sample_points >=color.shape[2]*color.shape[3]:
				p_unscaled, p = sample(color,color.shape[0])
			else:
				p_unscaled, p = patch_sample(color, color.shape[0], args.num_sample_points, patch_size=patch_size) #(batch_size, n_points, 2)

			p=p.cuda(non_blocking=True)
			p_unscaled=p_unscaled.cuda(non_blocking=True)

			#Get the ground truth mask for the sampled points
			gt_mask = get_tensor_values(mask,p)
			gt_mask = gt_mask[...,0].bool()

			#Get the 3D points for evaluating occupancy and freespace losses as mentioned in DVR paper
			p_freespace = get_freespace_points(p, K, R, C, origin, scaling, depth_range=[0,4.]) #(B,n_points,3)
			p_occupancy = get_occupancy_points(p, K, R, C, origin, scaling, depth_img=gt_depth) #(B,n_points,3)

			#visualize_3D(p_occupancy.squeeze(0).cpu().numpy(),fname='occupancy.ply')
			#visualize_3D(p_freespace.squeeze(0).cpu().numpy(),fname='freespace.ply')

			#Forward pass
			p_world_hat, rgb_pred, logits_occupancy, logits_freespace, mask_pred, p_world_hat_sparse, mask_pred_sparse, normals=net(p, p_occupancy,p_freespace,color,K,R,C,origin,scaling)
			#visualize_3D(p_world_hat.detach().squeeze(0).cpu().numpy(),fname='p_world_hat%03d.ply'%(epoch))

			#Calculate loss
			#Photoconsistency loss
			mask_rgb = mask_pred & gt_mask
			calculate_photoconsistency_loss(lambda_rgb, lambda_image_gradients,mask_rgb, rgb_pred, color, p, 'sum', loss, patch_size, eval_mode=False)

			#Depth loss
			mask_depth = mask_pred & gt_mask
			calculate_depth_loss(lambda_depth,mask_depth, gt_depth, p, K, R, C, origin, scaling, p_world_hat, 'sum', loss, eval_mode=False)

			#Normal loss
			calculate_normal_loss(lambda_normal,normals, color.shape[0], loss)

			#TODO sparse depth?

			#Freespace loss
			mask_freespace = (gt_mask==0)
			calculate_freespace_loss(lambda_freespace,logits_freespace, mask_freespace, 'sum', loss)

			#Occupancy loss
			mask_occupancy = (mask_pred ==0)&gt_mask
			calculate_occupancy_loss(lambda_occupancy,logits_occupancy, mask_occupancy, 'sum',loss)

			#Update the loss lists
			loss_list['loss'].append(loss['loss'].item())
			loss_list['loss_rgb'].append(loss['loss_rgb'])
			loss_list['loss_image_gradient'].append(loss['loss_image_gradient'])
			loss_list['loss_depth'].append(loss['loss_depth'])
			loss_list['loss_normal'].append(loss['loss_normal'])
			loss_list['loss_freespace'].append(loss['loss_freespace'])
			loss_list['loss_occupancy'].append(loss['loss_occupancy'])			


			total_loss = loss['loss']
			total_loss.backward()
			optimizer.step()
			optimizer.zero_grad()


			if epoch%1==0 and train_idx%1==0 or (epoch==0 and train_idx==0):
				print('epoch: %d, iter: %d, rgb loss: %2.4f, image gradient loss:%2.4f, depth loss: %2.4f, normal loss: %2.4f, freespace loss: %2.4f, occupancy loss: %2.4f, total loss: %2.4f'%(epoch, train_idx, loss['loss_rgb'],
					loss['loss_image_gradient'], loss['loss_depth'], loss['loss_normal'], loss['loss_freespace'], loss['loss_occupancy'],loss['loss'].item()))
				with torch.no_grad():
					#Plot the error
					plot_path=os.path.join(args.output, 'pretrain','plots','training_error.png')

					if sum(loss_list['loss'])>0:
						plt.plot(loss_list['loss'], label='Total loss')
					if sum(loss_list['loss_rgb'])>0:
						plt.plot(loss_list['loss_rgb'], label='RGB loss')
					if sum(loss_list['loss_image_gradient'])>0:
						plt.plot(loss_list['loss_image_gradient'], label='Image grad loss')
					if sum(loss_list['loss_depth'])>0:
						plt.plot(loss_list['loss_depth'], label='Depth loss')
					if sum(loss_list['loss_normal'])>0:
						plt.plot(loss_list['loss'], label='Normal loss')
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


					#Save a ground truth point cloud for reference
					#X,_ = depth_to_3D(gt_depth, K, R, C, scaling, origin, 1)
					#visualize_3D_masked(X[0].cpu().numpy(), mask[0,0].cpu().numpy(),fname=os.path.join(args.output, 'pretrain', 'meshes','gt_rescaled_epoch_%04d_iter_%04d.ply'%(epoch, train_idx)))
					visualize_prediction(net, color.shape[0], device,color,fname=os.path.join(args.output, 'pretrain', 'meshes','pred_epoch_%04d_iter_%04d'%(epoch, train_idx)))
					#render_img(device, color, net, K, R, C, origin, scaling, fname='color_epoch_%04d_iter_%04d.png'%(epoch, train_idx))
					#render_img(device, gt_depth, net, K, R, C, origin, scaling, fname='depth_epoch_%04d_iter_%04d.png'%(epoch, train_idx))









