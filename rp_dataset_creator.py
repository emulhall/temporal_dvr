import argparse
import pickle
import glob
import os
import numpy as np
from natsort import natsorted
from PIL import Image
from utils import calculate_scaling_and_origin_rp

def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Render People Pickle Creator')
	#parser.add_argument('--input', type=str, default='/mars/mnt/oitstorage/emily_storage/train_512_RenderPeople_all_sparse')
	#parser.add_argument('--output', type=str, default='/mars/mnt/oitstorage/emily_storage/pickles/')
	parser.add_argument('--input', type=str, default='/media/mulha024/i/train_512_RenderPeople_all_sparse')
	parser.add_argument('--output', type=str, default='./data/')

	return parser.parse_args()

def create_split(ROOT_dir,train):
	final_split=[[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]], [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]], [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]]
	total_added=0

	cameras=natsorted(glob.glob(os.path.join(ROOT_dir,'*')))

	camera_pairs=np.loadtxt('./camera_pairs.txt',delimiter=',',dtype=np.int32)
	#camera_pairs=np.loadtxt('/mars/mnt/oitstorage/emily_storage/camera_pairs.txt',delimiter=',',dtype=np.int32)

	cam_nums=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,23,24,26,27,28,29,30,31,32,33,34,36,38,39,40,41,42,43,44,45,46,47,48,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,66,67,68,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106]
	
	for i in range(len(camera_pairs)-98):
		#Get the two cameras closest to the camera 
		i1=camera_pairs[i][0]
		i2=camera_pairs[i][1]

		#Get the actual camera numbers
		c=cam_nums[i]

		print("Processing camera: " + str(c))

		#Get subjects
		subjects=natsorted(glob.glob(os.path.join(cameras[i],'color/*.png')))

		if train:
			subjects=subjects[:int(len(subjects)*0.8)]
		else:
			subjects=subjects[int(len(subjects)*0.8):]

		head,cam=os.path.split(cameras[i])

		for s in subjects[:1]:
			#Missing data for camera 56 and 83
			if c==56 or c==83:
				continue

			_,subject=os.path.split(s)
			subject=subject[:-4]

			#Get color paths for camera and two closest cameras
			color_path=s

			#Get mask paths for camera nad two closest cameras
			mask_path=s.replace('color','gr_mask')

			#Get depth paths for camera and two closest cameras
			gt_depth_path=s.replace('color','complete_depth').replace('.png','.txt')

			#Get the bounding box coordinates for camera and two closest cameras
			bbox=s.replace('color','boundingBox').replace('.png','.txt')

			#Get the camera intrinsics for camera and two closest cameras
			K=os.path.join(head, str(c),'camera','K.txt')

			#Get the camera rotation matrix for camera and two closest cameras
			R=os.path.join(head, str(c),'camera','R.txt')

			#Get the camera center matrix for camera and two closest cameras
			C=os.path.join(head, str(c),'camera','cen.txt')

			origin,scaling=calculate_scaling_and_origin_rp(np.loadtxt(bbox, delimiter=','),512)

			if origin.shape!=(2,):
				print(origin.shape)
				print(c)
			if scaling.shape!=():
				print(scaling.shape)
				print(c)

			#Append the paths to the correct spots in the split
			final_split[0][0].append(color_path)

			final_split[0][2].append(mask_path)

			final_split[0][3].append(bbox)

			final_split[0][4].append(K)

			final_split[0][5].append(R)

			final_split[0][6].append(C)

			final_split[0][7].append(gt_depth_path)

			final_split[0][13].append(np.array(origin,dtype='f'))

			final_split[0][14].append(np.array(scaling,dtype='f'))

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
	with open(os.path.join(args.output, 'RenderPeople_dvr.pkl'), 'wb') as f:
		pickle.dump(final_dict, f)