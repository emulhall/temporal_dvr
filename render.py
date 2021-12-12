from utils import sample
import torch
import numpy as np
from PIL import Image
from geometry import world_to_camera

def render_img(device, img,model,K, R, C, origin, scaling, c=None, fname='test.png'):
	h=img.shape[2]
	w=img.shape[3]

	p_loc, pixels = sample(img, img.shape[0])
	pixels = pixels.to(device)

	with torch.no_grad():
		p_world_hat, mask_pred, mask_zero_occupied = model.pixels_to_world(pixels, K, R, C, origin, scaling,c=c, sampling_accuracy=[1024,1025])
	p_loc=p_loc[mask_pred]

	with torch.no_grad():
		if img.shape[1]==3:
			img_out = (255 * np.ones((h,w,3))).astype(np.uint8)
			if mask_pred.sum() > 0:
				rgb_hat = model.decode_color(pixels_to_world, c=c)
				rgb_hat = rgb_hat[mask_pred].cpu().numpy()
				rgb_hat = (rgb_hat*255).astype(np.uint8)
				img_out[p_loc[:,1],p_loc[:,0]] = rgb_hat
			img_out = Image.fromarray(img_out).convert('RGB')
		elif img.shape[1]==1:
			img_out = (255*np.ones((h,w))).astype(np.uint8)
			if mask_pred.sum() > 0:
				p_world_hat = p_world_hat[mask_pred].unsqueeze(0)
				d_values = world_to_camera(p_world_hat, K, R, C, origin, scaling).squeeze(0)[...,-1].cpu().numpy()
				m = d_values[d_values != np.inf].min()
				M = d_values[d_values != np.inf].max()
				d_values = 0.5+0.45 *(d_values - m) / (M-m)
				d_image_vales = d_values*255
				img_out[p_loc[:,1],p_loc[:,0]] = d_image_vales.astype(np.uint8)

			img_out = Image.fromarray(img_out).convert("L")

		img_out.save(fname)

