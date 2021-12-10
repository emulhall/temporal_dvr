#modified from https://github.com/autonomousvision/differentiable_volumetric_rendering
import torch.nn as nn

class ResnetBlockFC(nn.Module):

	def __init__(self, size_in, size_out=None, size_h=None):
		super().__init__()

		#Attributes
		if size_out is None:
			size_out = size_in

		if size_h is None:
			size_h = min(size_in, size_out)

		self.size_in = size_in
		self.size_h = size_h
		self.size_out = size_out

		#Submodules
		self.fc_0 = nn.Linear(size_in, size_h)
		self.fc_1 = nn.Linear(size_h, size_out)
		self.actvn = nn.ReLU()

		if size_in == size_out:
			self.shortcut = None
		else:
			self.shortcut = nn.Linear(size_in, size_out, bias=False)

		nn.init.zeros_(self.fc_1.weight)

	def forward(self, x):
		net = self.fc_0(self.actvn(x))
		dx = self.fc_1(self.actvn(x))

		if self.shortcut is not None:
			x_s = self.shortcut(x)
		else:
			x_s= x

		return x_s + dx