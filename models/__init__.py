#Modified version of https://github.com/autonomousvision/differentiable_volumetric_rendering
import torch
import torch.nn as nn
from torch import distributions as dist

from models import depth_function, decoder
from utils import points_to_world, normalize_tensor, get_mask

class DVR(nn.Module):
	def __init__(self, decoder, encoder=None, device=None,depth_function_kwargs={}):
		super().__init__()

		self.decoder=decoder.to(device)
		if encoder is not None:
			self.encoder = encoder.to(device)
		else:
			self.encoder=None

		self._device=device

		self.call_depth_function = depth_function.DepthModule(**depth_function_kwargs)


	def forward(self, p, p_occupancy, p_freespace, inputs, K, R, C, origin, scale, it=None, sparse_depth=None,calc_normals=False, **kwargs):

		#Encode inputs
		c=self.encode_inputs(inputs) #(1,0)
		c=None

		p_world, mask_pred, mask_zero_occupied = self.pixels_to_world(p, K, R, C, origin, scale, c,it)

		rgb_pred = self.decode_color(p_world, c=c)

		#eval occ at sampled p
		logits_occupancy = self.decode(p_occupancy, c=c).logits

		# eval freespace at p and
		# fill in predicted world points

		p_freespace[mask_pred] = p_world[mask_pred].detach()
		logits_freespace = self.decode(p_freespace,c=c).logits

		if calc_normals:
			normals = self.get_normals(p_world.detach(), mask_pred, c=c)
		else:
			normals = None

		#if spase_depth is not None:
		mask_pred_sparse=None
		p_world_sparse=None

		return (p_world, rgb_pred, logits_occupancy, logits_freespace, mask_pred, p_world_sparse, mask_pred_sparse, normals)

	def get_normals(self, points, mask, c=None, h_sample=1e-3, h_finite_difference=1e-3):

		if mask.sum() > 0:
			c = c.unsqueeze(1).repeat(1,points.shape[1],1)[mask]
			points = points[mask]
			points_neighbor = points + (torch.rand_like(points)*h_sample - (h_sample /2.))

			normals_p = normalize_tensor(self.get_central_difference(points, c=c, h=h_finite_difference))
			normals_neighbor = normalize_tensor(self.get_central_difference(points_neighbor, c=c, h=h_finite_difference))

		else:
			normals_p = torch.empty(0,3).cuda(non_blocking=True)
			normals_neighbor = torch.empty(0,3).cuda(non_blocking=True)

		return [normals_p, normals_neighbor]


	def get_central_difference(self, points, c=None, h=1e-3):
		n_points,_ = points.shape

		if c.shape[-1]!=0:
			c=c.unsqueeze(1).repeat(1,6,1).view(-1,c.shape[-1])

		#calculate steps x + h/2 and x - h/2 for all 3 dimensions
		step = torch.cat([
			torch.tensor([1.,0,0]).view(1,1,3).repeat(n_points,1,1),
			torch.tensor([-1.,0,0]).view(1,1,3).repeat(n_points,1,1),
			torch.tensor([0,1.,0]).view(1,1,3).repeat(n_points,1,1),
			torch.tensor([0,-1.,0]).view(1,1,3).repeat(n_points,1,1),
			torch.tensor([0,0,1.]).view(1,1,3).repeat(n_points,1,1),
			torch.tensor([0,0,-1.].view(1,1,3).repeat(n_points,1,1))],dim=1).cuda(non_blocking=True) * h/2

		points_eval = (points.unsqueeze(1).repeat(1,6,1)+step).view(-1,3)

		#Eval decoder at these points
		f = self.decoder(points_eval, c=c, only_occupancy=True, batchwise=False).view(n_points, 6)

		#Get approximate derivative as f(x+h/2)-f(x-h/2)
		df_dx = torch.stack([
			(f[:,0]-f[:,1]),
			(f[:,2]-f[:,3]),
			(f[:,4]-f[:,5])],dim=-1)

		return df_dx


	def encode_inputs(self, inputs):
		if self.encoder is not None:
			c = self.encoder(inputs)
		else:
			c = torch.empty(inputs.size(0),0).cuda(non_blocking=True)
		return c

	def decode_color(self, p_world, c=None, **kwargs):
		rgb_hat = self.decoder(p_world, c=c, only_texture=True)
		rgb_hat = torch.sigmoid(rgb_hat)
		return rgb_hat


	def pixels_to_world(self,p, K, R, C, origin, scale, c,it=None, sampling_accuracy=None):
		p_world = points_to_world(p, K, R, C, origin, scale) # B,n_points,3

		c_world = C.squeeze(1).permute(0,2,1).repeat(1,p.shape[1],1)

		ray = p_world - c_world

		d_hat, mask_pred, mask_zero_occupied = self.march_along_ray(c_world, ray, c, it,sampling_accuracy)

		p_world_hat = c_world + ray*d_hat.unsqueeze(-1)

		return p_world_hat, mask_pred, mask_zero_occupied


	def march_along_ray(self,origin, ray_direction, c=None, it=None, sampling_accuracy=None):
		d_i = self.call_depth_function(origin, ray_direction, self.decoder, c=c, it=it, n_steps=sampling_accuracy)

		#Get mask for where first evaluation point is occupied
		mask_zero_occupied = d_i==0

		#Get mask for predicted depth
		mask_pred = get_mask(d_i).detach()

		d_hat = torch.ones_like(d_i).cuda(non_blocking=True)
		d_hat[mask_pred] = d_i[mask_pred]
		d_hat[mask_zero_occupied] = 0.

		return d_hat, mask_pred, mask_zero_occupied


	def to(self, device):
		model = super().to(device)
		model._device = device
		return model

	def decode(self, p, c=None, **kwargs):
		logits = self.decoder(p,c,only_occupancy=True, **kwargs)
		p_r = dist.Bernoulli(logits=logits)
		return p_r