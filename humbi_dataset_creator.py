import argparse
import pickle
import glob
import os
import numpy as np
from natsort import natsorted
from PIL import Image
from utils import calculate_scaling_and_origin_rp, get_dp_correspondences
import copy


def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Render People Pickle Creator')
	#parser.add_argument('--input', type=str, default='/mars/mnt/oitstorage/emily_storage/train_512_RenderPeople_all_sparse')
	#parser.add_argument('--output', type=str, default='/mars/mnt/oitstorage/emily_storage/pickles/')
	parser.add_argument('--input', type=str, default='/media/mulha024/i/HUMBI_1_80_256')
	parser.add_argument('--output', type=str, default='./data/')

	return parser.parse_args()

def create_split(ROOT_dir,train):
	final_split=[[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]], [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]], [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]]
	total_added=0

	cam_nums=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,23,24,26,27,28,29,30,31,32,33,34,36,38,39,40,41,42,43,44,45,46,47,48,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,66,67,68,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106]


	subjects=natsorted(glob.glob(os.path.join(ROOT_dir,'*')))

	for s in subjects[:1]:
		time_stamps = glob.glob(os.path.join(s, '0*'))

		for t in time_stamps[:1]:
			#Select a random other time in the video
			temp_time_stamps = copy.deepcopy(time_stamps)
			temp_time_stamps.remove(t)

			choice = np.random.choice(np.asarray(temp_time_stamps),size=1)[0]

			color_path_1 = glob.glob(os.path.join(t,'image_256','*.png'))[0]
			color_path_2 = glob.glob(os.path.join(choice, 'image_256','*.png'))[0]

			dp_path_1 = color_path_1.replace('image_256','densepose_256')
			dp_path_2 = color_path_2.replace('image_256','densepose_256')

			mask_path_1 = color_path_1.replace('image_256','mask_256')
			mask_path_2 = color_path_2.replace('image_256','mask_256')

			K = os.path.join(s,'K.txt')

			R = os.path.join(s,'R.txt')

			C = os.path.join(s, 'cen.txt')

			dp_1=np.asarray(Image.open(dp_path_1))
			mask_1=np.asarray(Image.open(mask_path_1))
			mask_temp_1 = np.where(mask_1[...,0]>0,1.,0.)+np.where(mask_1[...,1]>0,1.,0.)+np.where(mask_1[...,2]>0,1.,0.)
			mask_1 = np.where(mask_temp_1>0,1.,0)

			dp_2=np.asarray(Image.open(dp_path_2))
			mask_2=np.asarray(Image.open(mask_path_2))
			mask_temp_2 = np.where(mask_2[...,0]>0,1.,0.)+np.where(mask_2[...,1]>0,1.,0.)+np.where(mask_2[...,2]>0,1.,0.)
			mask_2 = np.where(mask_temp_2>0,1.,0)

			x_1,x_2,warning=get_dp_correspondences(dp_1,dp_2,mask_1[...,np.newaxis],mask_2[...,np.newaxis],num_matches=2000,threshold=1e-6,visualize=False)

			scale_and_origin_path_1 = color_path_1.replace('image_256', 'scaling_and_origin_256').replace('image','').replace('png','txt')
			scale_and_origin_path_2 = color_path_2.replace('image_256', 'scaling_and_origin_256').replace('image','').replace('png','txt')

			scaling_and_origin_1=np.loadtxt(scale_and_origin_path_1)
			
			scaling_1=scaling_and_origin_1[0,0]
			origin_1=scaling_and_origin_1[1,:]

			scaling_and_origin_2=np.loadtxt(scale_and_origin_path_2)
			
			scaling_2=scaling_and_origin_2[0,0]
			origin_2=scaling_and_origin_2[1,:]

			#Append the paths to the correct spots in the split
			final_split[0][0].append(color_path_1)
			final_split[1][0].append(color_path_2)

			final_split[0][1].append(dp_path_1)
			final_split[1][1].append(dp_path_2)

			final_split[0][2].append(mask_path_1)
			final_split[1][2].append(mask_path_2)

			final_split[0][4].append(K)

			final_split[0][5].append(R)

			final_split[0][6].append(C)

			final_split[1][11].append(np.array(x_1,dtype='f'))
			final_split[1][12].append(np.array(x_2,dtype='f'))

			final_split[0][13].append(np.array(origin_1,dtype='f'))
			final_split[1][13].append(np.array(origin_2,dtype='f'))			

			final_split[0][14].append(np.array(scaling_1,dtype='f'))
			final_split[1][14].append(np.array(scaling_2,dtype='f'))

			total_added+=1

	print(str(total_added)+' added to split.')
	return final_split

if __name__ == '__main__':
	args=ParseCmdLineArguments()
	#Uncomment to create the pickle file for only one view
	'''final_dict={'train': create_split(args.input,True)}
	with open(os.path.join(args.output, 'RenderPeople_ind.pkl'), 'wb') as f:
		pickle.dump(final_dict, f)'''


	if not (os.path.isdir(args.output)):
		os.mkdir(args.output)

	final_dict={'train': create_split(args.input,True), 'val':create_split(args.input,False)}
	with open(os.path.join(args.output, 'HUMBI_dvr.pkl'), 'wb') as f:
		pickle.dump(final_dict, f)