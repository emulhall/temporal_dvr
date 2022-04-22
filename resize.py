import pickle
import glob
import os
import numpy as np
import math
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

	ROOT = '/media/mulha024/i/Body_1_80_update'
	size = 512
	new_root = ROOT.replace('Body_1_80_update','HUMBI_%i'%(size))
	#Set up directories to save results to
	if not (os.path.isdir(new_root)):
		os.mkdir(new_root)
	
	subjects = natsorted(glob.glob(os.path.join(ROOT, '*')))

	for s in subjects[2:3]:
		_, subject = os.path.split(s)
		time_stamps = natsorted(glob.glob(os.path.join(s, 'body','0*')))
		
		#Set up directories to save results to
		subject_path = os.path.join(new_root, subject)
		if not (os.path.isdir(subject_path)):
			os.mkdir(subject_path)


		for t in time_stamps[:1]:
			_, time = os.path.split(t)

			time_path = os.path.join(subject_path, time)
			if not os.path.isdir(time_path):
				os.mkdir(time_path)

			bbox_path = os.path.join(time_path, 'bbox')
			color_path = os.path.join(time_path, 'color')
			mask_path = os.path.join(time_path, 'mask')

			if not os.path.isdir(bbox_path):
				os.mkdir(bbox_path)

			if not os.path.isdir(color_path):
				os.mkdir(color_path)

			if not os.path.isdir(mask_path):
				os.mkdir(mask_path)


			remove_bgs = natsorted(glob.glob(os.path.join(t,'removeBg','image*.png')))

			for r in remove_bgs:
				_,camera = os.path.split(r)
				camera = int(camera.replace('.png','').replace('image',''))

				image = np.asarray(Image.open(r))

				mask = np.where(image[...,3]>0,255,0)

				#mask = Image.fromarray(np.uint8(mask))

				#mask.save(r.replace('removeBg','mask'))

				#mask = np.asarray(Image.open(m)) # 1080,1920
				color = np.asarray(Image.open(m.replace('mask','image').replace('.png','.jpg')))
				color = color*np.where(mask[...,np.newaxis]>0,1.0,0.0)

				valid = np.where(mask>0)

				max_x = np.max(valid[1])
				min_x = np.min(valid[1])
				max_y = np.max(valid[0])
				min_y = np.min(valid[0])

				height = max_y - min_y
				width = max_x - min_x

				if height > width:
					difference = height - width
					max_x+= math.ceil(difference/2)
					min_x-= math.floor(difference/2)
					width = max_x - min_x
				elif width>height:
					difference = width - height
					max_y+= math.ceil(difference/2)
					min_y-= math.floor(difference/2)
					height = max_y - min_y

				assert(width==height)


				mask = Image.fromarray(np.uint8(mask[min_y:max_y,min_x:max_x]))
				color = Image.fromarray(np.uint8(color[min_y:max_y,min_x:max_x]))
				
				mask = mask.resize((size,size))
				color = color.resize((size, size))

				bbox = np.asarray([max_y, min_y, max_x, min_x])
				np.savetxt(os.path.join(bbox_path,'%07d.txt'%(camera)),bbox)

				mask.save(os.path.join(mask_path, '%07d.png'%(camera)))
				color.save(os.path.join(color_path,'%07d.png'%(camera)))





