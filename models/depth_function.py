#Modified version of https://github.com/autonomousvision/differentiable_volumetric_rendering
import torch
import numpy as np
import torch.nn as nn

from utils import get_logits_from_prob, get_proposal_points_in_unit_cube


class DepthModule(nn.Module):

	def __init__(self, tau=0.5, n_steps=[128,129], n_secant_steps=8, depth_range=[0.,2,4],
		method='secant',check_cube_intersection=True,max_points=3700000, schedule_ray_sampling=True,
		schedule_milestones=[50000,100000,25000], init_resolution=16):
		super().__init__()
		self.tau = tau
		self.n_steps = n_steps
		self.n_secant_steps = n_secant_steps
		self.depth_range = depth_range
		self.calc_depth=DepthFunction.apply
		self.method = method
		self.check_cube_intersection = check_cube_intersection
		self.max_points = max_points
		self.schedule_ray_sampling = schedule_ray_sampling
		self.schedule_milestones = schedule_milestones
		self.init_resolution = init_resolution


	def get_sampling_accuracy(self,it):
		if len(self.schedule_milestones)==0:
			return [128,129]
		else:
			res = self.init_resolution
			for i, milestone in enumerate(self.schedule_milestones):
				if it <milestone:
					return [res,res+1]
				res = res*2
			return [res, res+1]

	def forward(self, origin, ray_direction, decoder, c=None, it=None, n_steps=None):
		if n_steps is None:
			if self.schedule_ray_sampling and it is not None:
				n_steps = self.get_sampling_accuracy(it)
			else:
				n_steps = self.n_steps

		if n_steps[1] > 1:
			inputs = [origin, ray_direction, decoder, c, n_steps,
			self.n_secant_steps, self.tau, self.depth_range, self.method,
			self.check_cube_intersection, self.max_points] + [k for k in decoder.parameters()]

			d_hat = self.calc_depth(*inputs)

		else:
			d_hat = torch.full((origin.shape[0],origin.shape[2]),np.inf)

		return d_hat




class DepthFunction(torch.autograd.Function):

	@staticmethod
	def run_secant_method(f_low, f_high, d_low, d_high, n_secant_steps, origin_masked, ray_direction_masked, decoder, c, logit_tau):
		d_pred = -f_low *(d_high - d_low) / (f_high -f_low) + d_low
		for i in range(n_secant_steps):
			p_mid = origin_masked + d_pred.unsqueeze(-1) * ray_direction_masked
			with torch.no_grad():
				f_mid = decoder(p_mid, c, batchwise = False, only_occupancy=True)-logit_tau

			ind_low = f_mid <0
			if ind_low.sum() >0:
				d_low[ind_low] = d_pred[ind_low]
				f_low[ind_low] = f_mid[ind_low]
			if (ind_low == 0).sum() > 0:
				d_high[ind_low==0] = d_pred[ind_low==0]
				f_high[ind_low==0] = f_mid[ind_low==0]

			d_pred = -f_low*(d_high - d_low) / (f_high - f_low) + d_low

		return d_pred


	@staticmethod
	def perform_ray_marching(origin, ray_direction, decoder, c=None, tau=0.5, n_steps=[128,129], n_secant_steps=8, depth_range=[0.,2.4], method='secant', check_cube_intersection=True, max_points=3500000):
		batch_size, n_points, D = origin.shape
		device=origin.device
		logit_tau = get_logits_from_prob(tau)
		n_steps = torch.randint(n_steps[0],n_steps[1],(1,)).item()

		#Prepare the proposal depth values and corresponding 3D points
		d_proposal = torch.linspace(depth_range[0],depth_range[1],steps=n_steps).view(1,1,n_steps,1).to(device)
		d_proposal = d_proposal.repeat(batch_size, n_points, 1,1)

		if check_cube_intersection:
			d_proposal_cube, mask_inside_cube = get_proposal_points_in_unit_cube(origin, ray_direction, padding=0.1, eps=1e-6, n_steps=n_steps)
			d_proposal[mask_inside_cube] = d_proposal_cube[mask_inside_cube]

		p_proposal = origin.unsqueeze(2).repeat(1,1,n_steps,1)+ray_direction.unsqueeze(2).repeat(1,1,n_steps,1)*d_proposal

		#Evaluate all proposal points
		with torch.no_grad():
			val = torch.cat([(decoder(p_split,c,only_occupancy=True)-logit_tau) for p_split in torch.split(p_proposal.view(batch_size,-1,3),int(max_points/batch_size),dim=1)],dim=1).view(batch_size,-1,n_steps)


		#Create mask for valid points where the first point is not occupied
		mask_0_not_occupied = val[...,0] <0

		#Calculate if sign change occured
		sign_matrix = torch.cat([torch.sign(val[...,:-1]*val[...,1:]),torch.ones(batch_size,n_points,1).to(device)],dim=-1)

		cost_matrix = sign_matrix*torch.arange(n_steps,0,-1).float().to(device)

		#Get fist sign chance and mask values where a sign change occured and a neg to pos sign change occurred
		values, indices = torch.min(cost_matrix,-1)
		mask_sign_change = values<0
		mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),torch.arange(n_points).unsqueeze(-0),indices]<0
		mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

		#Get depth values and function values for the interval to which we want to apply the secant method
		n = batch_size*n_points
		d_low = d_proposal.view(n,n_steps,1)[torch.arange(n),indices.view(n)].view(batch_size,n_points)[mask]
		f_low = val.view(n, n_steps,1)[torch.arange(n),indices.view(n)].view(batch_size,n_points)[mask]
		indices = torch.clamp(indices+1,max=n_steps-1)
		d_high = d_proposal.view(n,n_steps,1)[torch.arange(n),indices.view(n)].view(batch_size,n_points)[mask]
		f_high = val.view(n, n_steps,1)[torch.arange(n),indices.view(n)].view(batch_size,n_points)[mask]

		origin_masked = origin[mask]
		ray_direction_masked = ray_direction[mask]

		#write c in pointwise format
		if c is not None and c.shape[-1] !=0:
			c=c.unsqueeze(1).repeat(1,n_points,1)[mask]

		#Apply surface depth refinement step (Secant method)
		if method == 'secant' and mask.sum() >0:
			d_pred = DepthFunction.run_secant_method(f_low,f_high,d_low, d_high, n_secant_steps, origin_masked, ray_direction_masked, decoder, c, logit_tau)
		elif method == 'bisection' and mask.sum() >0:
			d_pred = DepthFunction.run_bisection_method(d_low, d_high, n_secant_steps, origin_masked, ray_direction_masked, decoder, c, logit_tau)
		else:
			d_pred = torch.ones(ray_direction_masked.shape[0]).to(device)

		pt_pred = torch.ones(batch_size,n_points,3).to(device)
		pt_pred[mask] = origin_masked + d_pred.unsqueeze(-1)*ray_direction_masked
		d_pred_out = torch.ones(batch_size,n_points).to(device)
		d_pred_out[mask] = d_pred

		return d_pred_out, pt_pred, mask, mask_0_not_occupied

	@staticmethod
	def forward(ctx, *input):
		origin, ray_direction, decoder, c, n_steps, n_secant_steps, tau, depth_range,method, check_cube_intersection, max_points = input[:11]

		#Get depth values
		with torch.no_grad():
			d_pred, p_pred, mask, mask_0_not_occupied = DepthFunction.perform_ray_marching(origin, ray_direction, decoder, c, tau, n_steps, 
																							n_secant_steps, depth_range, method, check_cube_intersection, max_points)

		d_pred[mask==0] = np.inf
		d_pred[mask_0_not_occupied==0] = 0

		ctx.save_for_backward(origin, ray_direction, d_pred, p_pred,c)
		ctx.decoder = decoder
		ctx.mask = mask

		return d_pred


	@staticmethod
	def backward(ctx, grad_output):
		origin, ray_direction, d_pred, p_pred, c = ctx.saved_tensors
		decoder = ctx.decoder
		mask = ctx.mask
		eps = 1e-3

		with torch.enable_grad():
			p_pred.requires_grad = True
			f_p = decoder(p_pred, only_occupancy=True)
			f_p_sum = f_p.sum()
			grad_p = torch.autograd.grad(f_p_sum, p_pred, retain_graph=True)[0]
			grad_p_dot_v = (grad_p*ray_direction).sum(-1)

			if mask.sum() >0:
				grad_p_dot_v[mask==0]=1
				grad_p_dot_v[abs(grad_p_dot_v)<eps] = eps
				grad_outputs = -grad_output.squeeze(-1)
				grad_outputs = grad_outputs/grad_p_dot_v
				grad_outputs = grad_outputs*mask.float()

			#Gradients for latent code c
			if c is None or c.shape[-1]==0 or mask.sum()==0:
				gradc = None
			else:
				torch.autograd.grad(f_p, c, retain_graph=True, grad_outputs=grad_outputs)[0]

			#Gradients for network paameters phi
			if mask.sum() >0:
				grad_phi = torch.autograd.grad(f_p,[k for k in decoder.parameters()],grad_outputs=grad_outputs, retain_graph=True)
			else:
				grad_phi = [None for i in decoder.parameters()]

			#Return gradients for c,z and network parameters and none for all other inputs
			out = [None, None, None, gradc, None, None, None, None, None, None, None] + list(grad_phi)
			return tuple(out)