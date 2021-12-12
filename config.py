import models
import encoder
def get_model(device):
	'''
		Returns the model
	'''

	#decoder = 
	#encoder = 
	#depth_function =
	#depth_range

	model = models.DVR(models.decoder.Decoder(), encoder=encoder.Resnet18(c_dim=128), device=device)

	return model