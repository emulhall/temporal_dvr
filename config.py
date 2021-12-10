import models
def get_model(device):
	'''
		Returns the model
	'''

	#decoder = 
	#encoder = 
	#depth_function =
	#depth_range

	model = models.DVR(models.decoder.Decoder(), encoder=None, device=device)

	return model