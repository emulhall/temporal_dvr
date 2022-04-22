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
from utils import get_tensor_values,  reshape_multiview_tensors
from loss import calculate_photoconsistency_loss, calculate_depth_loss, calculate_freespace_loss, calculate_occupancy_loss, calculate_mask_loss, calculate_multi_rgb_loss, calculate_multi_depth_loss
from geometry import depth_to_3D
from view_3D import visualize_3D_masked, visualize_3D
from generate import generate_mesh

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	#parser.add_argument('--input', type=str, default='/mars/mnt/oitstorage/emily_storage/train_512_RenderPeople_all_sparse')
	#parser.add_argument('--output', type=str, default='/mars/mnt/oitstorage/emily_storage/temporal_dvr_results/')
	parser.add_argument('--input', type=str, default='/media/mulha024/i/HUMBI_512/')
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
	parser.add_argument('--checkpoint_load_path', type=str, default='')
	parser.add_argument('--num_views', type=int, default=2)
	return parser.parse_args()


if __name__ == '__main__':
	args=ParseCmdLineArguments()

	#Set up dataloaders
	train_dataset = HUMBIDataset(args.input,
		usage='train',
		num_samples=args.num_samples,
		scale=args.scale,
		random_multiview=args.random_multiview,
		num_views=args.num_views)

	val_dataset = HUMBIDataset(args.input,
		usage='val',
		num_samples=args.num_samples,
		scale=args.scale,
		random_multiview=args.random_multiview,
		num_views = args.num_views)

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

	error_path=os.path.join(args.output, args.name,'plots','error.txt')
	val_error_path=os.path.join(args.output, args.name,'plots','validation_error.txt')			

	is_cuda = (torch.cuda.is_available())
	device = torch.device("cuda" if is_cuda else "cpu")

	#Set lambda values
	lambda_rgb=1
	lambda_freespace=0
	lambda_occupancy=0
	lambda_mask=1
	lambda_multi_rgb = 0
	lambda_multi_depth = 0

	#Initialize training
	net = get_model(device)
	optimizer=torch.optim.Adam(net.parameters(),lr=args.learning_rate)
	start_epoch=0

	if args.continue_train or args.resume_epoch>0:
		train_error_list = np.loadtxt(error_path)
		val_error_list = np.loadtxt(val_error_path)
		if args.resume_epoch>0:
			net.load_state_dict(torch.load(os.path.join(args.output,args.name,'models','net_epoch_%06d.cpkt'%(args.resume_epoch))))
			print("Checkpoint loaded")
			start_epoch = args.resume_epoch
			loss_list={'train':train_error_list[:args.resume_epoch].tolist(), 'valid':val_error_list[:args.resume_epoch].tolist()}
		else:
			net.load_state_dict(torch.load(os.path.join(args.output,args.name,'models','latest.cpkt')))
			print("Checkpoint loaded")
			loss_list={'train':train_error_list, 'valid':val_error_list}

	elif args.checkpoint_load_path != '':
		net.load_state_dict(torch.load(args.checkpoint_load_path))
		print("Checkpoint loaded")
	else:
		loss_list={'train':[], 'valid':[]}

	#Set up loss lists for plots
	loss_list={'loss':[], 'loss_rgb':[], 'loss_freespace':[], 'loss_occupancy':[], 'loss_mask':[], 'loss_multi_rgb': [], 'loss_multi_depth': []}
	loss_val_list={'loss':[], 'loss_rgb':[], 'loss_freespace':[], 'loss_occupancy':[],'loss_mask':[], 'loss_multi_rgb':[], 'loss_multi_depth':[]}	

	net.train()
	for epoch in range(args.epochs):

		val_iterator = iter(val_dataloader)
		
		avg_loss = 0
		rgb_loss = 0
		depth_loss = 0
		freespace_loss = 0
		occupancy_loss = 0
		mask_loss = 0
		multi_rgb_loss = 0
		multi_depth_loss = 0

		val_avg_loss = 0
		val_rgb_loss = 0
		val_depth_loss = 0
		val_freespace_loss = 0
		val_occupancy_loss = 0
		val_mask_loss = 0
		val_multi_rgb_loss=0
		val_multi_depth_loss = 0
		
		iters = 0

		for train_idx, train_data in enumerate(train_dataloader):
			#Retrieve a batch of RenderPeople data
			color = train_data['color'].to(device)
			mask = train_data['mask'].to(device)
			K = train_data['K'].to(device) 
			R = train_data['R'].to(device) 
			C = train_data['C'].to(device)
			origin = train_data['origin'].to(device)
			scaling = train_data['scaling'].to(device)
			p = train_data['p'].to(device).squeeze(1)
			p_freespace =train_data['p_freespace'].to(device).squeeze(1)
			p_occupancy = train_data['p_occupancy'].to(device).squeeze(1)
			p_mask = train_data['p_mask'].to(device).squeeze(1)
			bb_min = train_data['bb_min'].to(device)[0]
			bb_max = train_data['bb_max'].to(device)[0]
			p_corr_1 = train_data['p_corr_1'].to(device)
			p_corr_2 = train_data['p_corr_2'].to(device)			

			color = reshape_multiview_tensors(color)
			mask = reshape_multiview_tensors(mask)
			K = K.permute(1,0,2,3)
			R = R.permute(1,0,2,3)
			C = C.permute(1,0,2,3)
			origin = origin.permute(1,0,2,3)
			scaling = scaling.permute(1,0,2,3)

			p = p.view((p.shape[0]*p.shape[1],p.shape[2],p.shape[3]))
			p_mask = p_mask.view((p_mask.shape[0]*p_mask.shape[1],p_mask.shape[2],p_mask.shape[3]))
			p_freespace = p_freespace.view((p_freespace.shape[0]*p_freespace.shape[1],p_freespace.shape[2],p_freespace.shape[3]))
			p_occupancy = p_occupancy.view((p_occupancy.shape[0]*p_occupancy.shape[1],p_occupancy.shape[2],p_occupancy.shape[3]))

			p_corr_1 = p_corr_1.view((p_corr_1.shape[0]*p_corr_1.shape[1], p_corr_1.shape[2], p_corr_1.shape[3]))
			p_corr_2 = p_corr_2.view((p_corr_2.shape[0]*p_corr_2.shape[1], p_corr_2.shape[2], p_corr_2.shape[3]))

			#Initialize loss dictionary
			loss={'loss':0, 'loss_rgb':0, 'loss_depth':0,   'loss_freespace':0, 'loss_occupancy':0, 'loss_mask':0, 'loss_multi_rgb':0, 'loss_multi_depth': 0}
			loss_val={'loss':0, 'loss_rgb':0, 'loss_depth':0,  'loss_freespace':0, 'loss_occupancy':0,'loss_mask':0, 'loss_multi_rgb':0, 'loss_multi_depth': 0}

			#Get the ground truth mask for the sampled points
			gt_mask = get_tensor_values(mask,p)[...,0]

			#visualize_3D(p_occupancy[0].cpu().numpy(),fname='occupancy.ply')
			#visualize_3D(p_freespace[0].cpu().numpy(),fname='freespace.ply')
			#visualize_3D(p_mask[0].cpu().numpy(),fname='mask.ply')

			#Forward pass
			p_world_hat, rgb_pred, logits_freespace, logits_occupancy, logits_mask,mask_pred, d_pred, p_world_1, rgb_pred_1, mask_pred_1, p_world_2, rgb_pred_2, mask_pred_2 = net(p, p_freespace,p_occupancy,p_mask,color,K,R,C,origin,scaling, p_corr_1, p_corr_2)
			#visualize_3D(p_world_hat.detach().squeeze(0).cpu().numpy(),fname='p_world_hat%03d.ply'%(epoch))

			#Calculate loss
			#Photoconsistency loss
			mask_rgb = mask_pred & gt_mask.bool()
			calculate_photoconsistency_loss(lambda_rgb, mask_rgb, rgb_pred, color, p, 'sum', loss)

			#Freespace loss
			calculate_freespace_loss(lambda_freespace,logits_freespace, 'sum', loss)

			#Occupancy loss
			calculate_occupancy_loss(lambda_occupancy,logits_occupancy, 'sum',loss)			

			#Mask loss
			calculate_mask_loss(lambda_mask, logits_mask, 'sum',loss)

			#Multi RGB loss
			calculate_multi_rgb_loss(lambda_multi_rgb, mask_pred_1.bool(), mask_pred_2.bool(), rgb_pred_1, rgb_pred_2,'sum',loss)

			#Multi depth loss
			calculate_multi_depth_loss(lambda_multi_depth, mask_pred_1.bool(), mask_pred_2.bool(), p_world_1, p_world_2, 'sum', loss)


			#Update the loss lists
			avg_loss+=loss['loss'].item()
			try:
				rgb_loss+=loss['loss_rgb'].item()
			except:
				rgb_loss+=loss['loss_rgb']
			try:
				freespace_loss+=loss['loss_freespace'].item()
			except:
				freespace_loss+=loss['loss_freespace']
			try:
				occupancy_loss+=loss['loss_occupancy'].item()
			except:
				occupancy_loss+=loss['loss_occupancy']
			try:
				mask_loss+=loss['loss_mask'].item()
			except:
				mask_loss+=loss['loss_mask']
			try:
				multi_rgb_loss+=loss['loss_multi_rgb'].item()
			except:
				multi_rgb_loss+=loss['loss_multi_rgb']
			try:
				multi_depth_loss+=loss['loss_multi_depth'].item()
			except:
				multi_depth_loss+=loss['loss_multi_depth']				


			total_loss = loss['loss']
			total_loss.backward()
			optimizer.step()
			optimizer.zero_grad()



			if train_idx%10==0 or (epoch==0 and train_idx==0):
				print('epoch: %d, iter: %d, rgb loss: %2.4f, freespace loss: %2.4f, occupancy loss: %2.4f, mask loss: %2.4f, multi rgb loss: %2.4f, multi depth loss: %2.4f, total loss: %2.4f'%(epoch, train_idx, loss['loss_rgb'],
					loss['loss_freespace'], loss['loss_occupancy'],loss['loss_mask'],loss['loss_multi_rgb'],loss['loss_multi_depth'],loss['loss'].item()))

		with torch.no_grad():
			loss_list['loss'].append(avg_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_rgb'].append(rgb_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_freespace'].append(freespace_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_occupancy'].append(occupancy_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_mask'].append(mask_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_multi_rgb'].append(multi_rgb_loss / (train_dataset.__len__()/args.batch_size))
			loss_list['loss_multi_depth'].append(multi_depth_loss / (train_dataset.__len__()/args.batch_size))						

			loss_val_list['loss'].append(val_avg_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_rgb'].append(val_rgb_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_freespace'].append(val_freespace_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_occupancy'].append(val_occupancy_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_mask'].append(val_mask_loss / (train_dataset.__len__()/args.batch_size))
			loss_val_list['loss_multi_rgb'].append(val_multi_rgb_loss / (train_dataset.__len__()/args.batch_size))	
			loss_val_list['loss_multi_depth'].append(val_multi_depth_loss / (train_dataset.__len__()/args.batch_size))						


			#Plot the error
			plot_path=os.path.join(args.output, args.name,'plots','error.png')

			val_plot_path=os.path.join(args.output, args.name,'plots','validation_error.png')

			error_path=os.path.join(args.output, args.name,'plots','error.txt')

			val_error_path=os.path.join(args.output, args.name,'plots','validation_error.txt')

			if sum(loss_list['loss'])>0:
				plt.plot(loss_list['loss'], label='Total loss')
			if sum(loss_list['loss_rgb'])>0:
				plt.plot(loss_list['loss_rgb'], label='RGB loss')
			if sum(loss_list['loss_freespace'])>0:
				plt.plot(loss_list['loss_freespace'], label='Freespace loss')
			if sum(loss_list['loss_occupancy'])>0:
				plt.plot(loss_list['loss_occupancy'], label='Occupancy loss')
			if sum(loss_list['loss_mask'])>0:
				plt.plot(loss_list['loss_mask'], label='Mask loss')
			if sum(loss_list['loss_multi_rgb'])>0:
				plt.plot(loss_list['loss_multi_rgb'], label='Multi RGB loss')
			if sum(loss_list['loss_multi_depth'])>0:
				plt.plot(loss_list['loss_multi_depth'], label='Multi depth loss')															

			plt.xlabel("Iterations")
			plt.ylabel("Error")
			plt.title("Error over Iterations")
			plt.legend()
			plt.savefig(plot_path)
			plt.clf()

			np.savetxt(val_error_path, np.reshape(loss_val_list['loss'],(-1)))
			np.savetxt(error_path, np.reshape(loss_list['loss'],(-1)))


		if epoch%10==0:
				#visualize_3D_masked(X[0].cpu().numpy(), mask[0,0].cpu().numpy(),fname=os.path.join(args.output, args.name, 'meshes','gt_epoch_%06d_subject_%s_view_%s.ply'%(epoch,train_data['name'][0], train_data['yid'][0].item())))	
				generate_mesh(net,device,color[:1], K[:1], R[:1], C[:1], scaling[:1], origin[:1],mask[:1],bb_min,bb_max,fname=os.path.join(args.output, args.name, 'meshes','train_epoch_%06d_%s_view_%s'%(epoch,train_data['name'][0], train_data['yid'][0].item())))
				print("Train mesh generated")








