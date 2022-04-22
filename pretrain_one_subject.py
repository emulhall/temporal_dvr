import argparse
import torch
from torch.utils.data import DataLoader
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import distributions as dist
import math

from dataset_one_subject import RenderPeopleDataset
from config import get_model
from utils import get_tensor_values, save_3D, reshape_multiview_tensors
from loss import calculate_photoconsistency_loss, calculate_depth_loss, calculate_normal_loss, calculate_freespace_loss, calculate_occupancy_loss, calculate_mask_loss
from geometry import depth_to_3D
from view_3D import visualize_3D_masked, visualize_3D
from render import render_img
from generate import generate_mesh, generate_mesh_color

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	parser.add_argument('--input', type=str, default='/media/mulha024/i/train_512_RenderPeople_all_sparse')
	parser.add_argument('--output', type=str, default='./results')
	#parser.add_argument('--input', type=str, default='/mars/mnt/oitstorage/emily_storage/train_512_RenderPeople_all_sparse')
	#parser.add_argument('--output', type=str, default='/mars/mnt/oitstorage/emily_storage/temporal_dvr_results/')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-4)
	#parser.add_argument('--epochs', type=int, default=10000000000000000)
	#parser.add_argument('--num_sample_points',type=int, default=10000)
	parser.add_argument('--epochs', type=int, default=151)
	parser.add_argument('--num_samples',type=int, default=5000)
	parser.add_argument('--name', type=str, default='pretrain')
	parser.add_argument('--scale',type=float, default=0.1)
	parser.add_argument('--random_multiview',action='store_true')
	parser.add_argument('--resume_epoch', type=int, default=0)
	parser.add_argument('--continue_train', action='store_true')

	return parser.parse_args()


if __name__ == '__main__':
	args=ParseCmdLineArguments()


	#Set up dataloaders
	train_dataset = RenderPeopleDataset(args.input,
		usage='train',
		num_samples=args.num_samples,
		scale=args.scale,
		random_multiview=args.random_multiview)

	val_dataset = RenderPeopleDataset(args.input,
		usage='val',
		num_samples=args.num_samples,
		scale=args.scale,
		random_multiview=args.random_multiview)

	train_dataloader = DataLoader(train_dataset, shuffle=True,
		batch_size=args.batch_size,
		num_workers=0)

	val_dataloader = DataLoader(val_dataset, shuffle=True,
		batch_size=args.batch_size,
		num_workers=0)

	#Set up directories to save results to
	if not (os.path.isdir(args.output)):
		os.mkdir(args.output)

	if not (os.path.isdir(os.path.join(args.output,args.name))):
		os.mkdir(os.path.join(args.output,args.name))

	if not (os.path.isdir(os.path.join(args.output,args.name,'plots'))):
		os.mkdir(os.path.join(args.output,args.name,'plots'))

	if not (os.path.isdir(os.path.join(args.output,args.name,'meshes'))):
		os.mkdir(os.path.join(args.output,args.name,'meshes'))

	if not (os.path.isdir(os.path.join(args.output,args.name,'models'))):
		os.mkdir(os.path.join(args.output,args.name,'models'))	

	is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda" if is_cuda else "cpu")

	#Set lambda values
	lambda_rgb=1
	lambda_depth=1
	lambda_normal=0
	lambda_freespace=1
	lambda_occupancy=1
	lambda_mask=1

	#Set up loss lists for plots
	loss_list={'loss':[], 'loss_rgb':[], 'loss_depth':[], 'loss_normal':[],  'loss_freespace':[], 'loss_occupancy':[], 'loss_mask':[]}
	loss_val_list={'loss':[], 'loss_rgb':[], 'loss_depth':[], 'loss_normal':[],  'loss_freespace':[], 'loss_occupancy':[],'loss_mask':[]}


	#Initialize training
	net = get_model(device)
	optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate)
	start_epoch=0

	if args.continue_train:
		if args.resume_epoch>0:
			net.load_state_dict(torch.load(os.path.join(args.output,args.name,'models','net_epoch_%06d.cpkt'%(args.resume_epoch))))
			start_epoch = args.resume_epoch
		else:
			net.load_state_dict(torch.load(os.path.join(args.output,args.name,'models','latest.cpkt')))


	net.train()
	for epoch in range(start_epoch, args.epochs):

		val_iterator = iter(val_dataloader)
		
		avg_loss = 0
		rgb_loss = 0
		depth_loss = 0
		freespace_loss = 0
		occupancy_loss = 0
		normal_loss = 0
		mask_loss = 0

		val_avg_loss = 0
		val_rgb_loss = 0
		val_depth_loss = 0
		val_freespace_loss = 0
		val_occupancy_loss = 0
		val_normal_loss = 0
		val_mask_loss = 0
		
		iters = 0


		for train_idx, train_data in enumerate(train_dataloader):
			#Retrieve a batch of RenderPeople data
			color = train_data['color'].to(device)
			mask = train_data['mask'].to(device)
			K = train_data['K'].to(device) 
			R = train_data['R'].to(device) 
			C = train_data['C'].to(device)
			gt_depth = train_data['gt_depth'].to(device)
			origin = train_data['origin'].to(device)
			scaling = train_data['scaling'].to(device)
			p = train_data['p'].to(device).squeeze(1)
			p_freespace =train_data['p_freespace'].to(device).squeeze(1)
			p_occupancy = train_data['p_occupancy'].to(device).squeeze(1)
			p_mask = train_data['p_mask'].to(device).squeeze(1)
			bb_min = train_data['bb_min'].to(device)[0]
			bb_max = train_data['bb_max'].to(device)[0]

			color = reshape_multiview_tensors(color)
			mask = reshape_multiview_tensors(mask)

			#Initialize loss dictionary
			loss={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_normal':0,  'loss_freespace':0, 'loss_occupancy':0, 'loss_mask':0}
			loss_val={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_normal':0,  'loss_freespace':0, 'loss_occupancy':0,'loss_mask':0}

			#Get the ground truth mask for the sampled points
			gt_mask = get_tensor_values(mask,p)[...,0]

			#visualize_3D(p_occupancy.squeeze(0).cpu().numpy(),fname='occupancy.ply')
			#visualize_3D(p_freespace.squeeze(0).cpu().numpy(),fname='freespace.ply')
			#visualize_3D(p_mask.squeeze(0).cpu().numpy(),fname='mask.ply')

			p_world_hat, rgb_pred, logits_occupancy, logits_freespace, logits_mask,mask_pred, normals,d_pred,_,_,_,_,_,_=net(p, p_occupancy,p_freespace,p_mask,color,K,R,C,origin,scaling)
			#visualize_3D(p_world_hat.detach().squeeze(0).cpu().numpy(),fname='p_world_hat%03d.ply'%(epoch))

			#Calculate loss
			#Photoconsistency loss
			mask_rgb = mask_pred & gt_mask.bool()
			calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, color, p, 'sum', loss)

			#Depth loss
			mask_depth = mask_pred & gt_mask.bool()
			calculate_depth_loss(lambda_depth,mask_depth, gt_depth, p, d_pred, 'sum', loss)

			#Normal loss
			calculate_normal_loss(lambda_normal,normals, color.shape[0], loss)

			#Freespace loss
			calculate_freespace_loss(lambda_freespace,logits_freespace, 'sum', loss)

			#Occupancy loss
			calculate_occupancy_loss(lambda_occupancy,logits_occupancy, 'sum',loss)

			#Mask loss
			calculate_mask_loss(lambda_mask, logits_mask, 'sum',loss)

			#Update the loss lists
			avg_loss+=loss['loss'].item()
			try:
				rgb_loss+=loss['loss_rgb'].item()
			except:
				rgb_loss+=loss['loss_rgb']
			try:
				depth_loss+=loss['loss_depth'].item()
			except:
				depth_loss+=loss['loss_depth']
			try:
				freespace_loss+=loss['loss_freespace'].item()
			except:
				freespace_loss+=loss['loss_freespace']
			try:
				occupancy_loss+=loss['loss_occupancy'].item()
			except:
				occupancy_loss+=loss['loss_occupancy']
			try:
				normal_loss+=loss['loss_normal'].item()
			except:
				normal_loss+=loss['loss_normal']
			try:
				mask_loss+=loss['loss_mask'].item()
			except:
				mask_loss+=loss['loss_mask']


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

			color_val = val_data['color'].to(device)
			mask_val = val_data['mask'].to(device)
			K_val = val_data['K'].to(device)
			R_val = val_data['R'].to(device)
			C_val = val_data['C'].to(device)
			gt_depth_val = val_data['gt_depth'].to(device)
			origin_val = val_data['origin'].to(device)
			scaling_val = val_data['scaling'].to(device)
			p = val_data['p'].to(device).squeeze(1)
			p_freespace =val_data['p_freespace'].to(device).squeeze(1)
			p_occupancy = val_data['p_occupancy'].to(device).squeeze(1)
			p_mask = val_data['p_mask'].to(device).squeeze(1)
			bb_min = train_data['bb_min'].to(device)[0]

			color_val = reshape_multiview_tensors(color_val)
			mask_val = reshape_multiview_tensors(mask_val)

			gt_mask = get_tensor_values(mask_val,p)[...,0]

			with torch.no_grad():
				p_world_hat, rgb_pred, logits_occupancy, logits_freespace, logits_mask,mask_pred, normals,d_pred,_,_,_,_,_,_=net(p, p_occupancy,p_freespace,p_mask,color_val,K_val,R_val,C_val,origin_val,scaling_val)

				mask_rgb = mask_pred & gt_mask.bool()
				calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, color_val, p, 'mean', loss_val)

				#Depth loss
				mask_depth = mask_pred & gt_mask.bool()
				calculate_depth_loss(lambda_depth,mask_depth, gt_depth_val, p, d_pred, 'mean', loss)

				#Normal loss
				calculate_normal_loss(lambda_normal,normals, color_val.shape[0], loss_val)

				#Freespace loss
				calculate_freespace_loss(lambda_freespace,logits_freespace, 'mean', loss_val)

				#Occupancy loss
				calculate_occupancy_loss(lambda_occupancy,logits_occupancy, 'mean',loss_val)

				#Mask loss
				calculate_mask_loss(lambda_mask, logits_mask, 'mean', loss_val)

				#Update the loss lists
				val_avg_loss+=loss_val['loss'].item()
				try:
					val_rgb_loss+=loss_val['loss_rgb'].item()
				except:
					val_rgb_loss+=loss_val['loss_rgb']
				try:
					val_depth_loss+=loss_val['loss_depth'].item()
				except:
					val_depth_loss+=loss_val['loss_depth']
				try:
					val_freespace_loss+=loss_val['loss_freespace'].item()
				except:
					val_freespace_loss+=loss_val['loss_freespace']
				try:
					val_occupancy_loss+=loss_val['loss_occupancy'].item()
				except:
					val_occupancy_loss+=loss_val['loss_occupancy']
				try:
					val_normal_loss+=loss_val['loss_normal'].item()
				except:
					val_normal_loss+=loss_val['loss_normal']					
				try:
					val_mask_loss+=loss_val['loss_mask'].item()
				except:
					val_mask_loss+=loss_val['loss_mask']

				if train_idx%10==0 or (epoch==0 and train_idx==0):
					print('epoch: %d, iter: %d, rgb loss: %2.4f, depth loss: %2.4f, freespace loss: %2.4f, occupancy loss: %2.4f, normal loss: %2.4f, mask loss: %2.4f, total loss: %2.4f'%(epoch, train_idx, loss['loss_rgb'],
						loss['loss_depth'], loss['loss_freespace'], loss['loss_occupancy'], loss['loss_normal'], loss['loss_mask'],loss['loss'].item()))

				
				#if (train_idx%100==0 and n==0):
				if (epoch%1==0 and epoch!=0 and train_idx==0):
					with torch.no_grad():
						#Save a ground truth point cloud for reference
						X,_ = depth_to_3D(gt_depth[:1], K[:1], R[:1], C[:1], scaling[:1], origin[:1])
						visualize_3D_masked(X[0].cpu().numpy(), mask[0,0].cpu().numpy(),fname=os.path.join(args.output, args.name, 'meshes','gt_epoch_%06d_subject_%s_view_%s.ply'%(epoch,train_data['name'][0], train_data['yid'][0].item())))	
						generate_mesh(net,device,color[:1], K[:1], R[:1], C[:1], scaling[:1], origin[:1],mask[:1],bb_min,bb_max,fname=os.path.join(args.output, args.name, 'meshes','train_epoch_%06d_subject_%s_view_%s'%(epoch,train_data['name'][0], train_data['yid'][0].item())))
						print("Train mesh generated")
											
						#generate_mesh(net,device,color_val[:1], K_val[:1], R_val[:1], C_val[:1], scaling_val[:1], origin_val[:1],mask_val[:1],bb_min,bb_max,fname=os.path.join(args.output, args.name, 'meshes','val_epoch_%06d_subject_%s_view_%s'%(epoch,val_data['name'][0], val_data['yid'][0].item())))
						#print("Validation mesh generated")

						#Save a model
						model_path = os.path.join(args.output,args.name,'models','net_epoch_%06d.cpkt'%(epoch))
						torch.save(net.state_dict(),model_path)

						model_path = os.path.join(args.output,args.name,'models','latest.cpkt')
						torch.save(net.state_dict(),model_path)
		
		with torch.no_grad():
			loss_list['loss'].append(avg_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_rgb'].append(rgb_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_depth'].append(depth_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_normal'].append(normal_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_freespace'].append(freespace_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_occupancy'].append(occupancy_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_mask'].append(mask_loss / (train_dataset.__len__()/args.batch_size))

			loss_val_list['loss'].append(val_avg_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_rgb'].append(val_rgb_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_depth'].append(val_depth_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_freespace'].append(val_freespace_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_occupancy'].append(val_occupancy_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_normal'].append(val_normal_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_mask'].append(val_mask_loss / (train_dataset.__len__()/args.batch_size))


			#Plot the error
			plot_path=os.path.join(args.output, args.name,'plots','error.png')

			val_plot_path=os.path.join(args.output, args.name,'plots','validation_error.png')

			error_path=os.path.join(args.output, args.name,'plots','error.txt')

			val_error_path=os.path.join(args.output, args.name,'plots','validation_error.txt')

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

			np.savetxt(val_error_path, np.reshape(loss_val_list['loss'],(-1)))
			np.savetxt(error_path, np.reshape(loss_list['loss'],(-1)))








