import torch.nn as nn
import torch.nn.functional as F
from layers import ResnetBlockFC


class Decoder(nn.Module):

	def __init__(self, dim=3,c_dim=128,hidden_size=512, leaky=False, n_blocks=5, out_dim=4):
		super().__init__()
		self.c_dim=c_dim
		self.n_blocks=n_blocks
		self.out_dim = out_dim

		self.fc_p = nn.Linear(dim,hidden_size)
		self.fc_out = nn.Linear(hidden_size,out_dim)

		if c_dim != 0:
			self.fc_c = nn.ModuleList([
				nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
				])

		self.blocks = nn.ModuleList([
			ResnetBlockFC(hidden_size) for i in range(n_blocks)
			])

		if not leaky:
			self.actvn = F.relu
		else:
			self.actvn = lambda x: F.leaky_relu(x, 0.2)

	def forward(self, p, c=None, only_occupancy=False, only_texture=False, batchwise=True):
		assert(len(p.shape)==3 or len(p.shape)==2)

		net = self.fc_p(p.float())
		for n in range(self.n_blocks):
			if self.c_dim != 0 and c is not None:
				net_c = self.fc_c[n](c)
				if batchwise:
					net_c = net_c.unsqueeze(1)
				net = net + net_c

			net = self.blocks[n](net)

		out = self.fc_out(self.actvn(net))
		if only_occupancy:
			if len(p.shape) == 3:
				out = out[:, :, 0]
			elif len(p.shape) == 2:
				out = out[:, 0]
		elif only_texture:
			if len(p.shape) == 3:
				out = out[:, :, 1:4]
			elif len(p.shape) == 2:
				out = out[:, 1:4]

		out = out.squeeze(-1)

		return out

