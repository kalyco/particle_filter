import numpy as np

class ImgMap:
	def __init__(self, img, distance_unit):
		self.img = img
		self.x_range = np.floor(img.shape[1]/2)
		self.y_range = np.floor(img.shape[0]/2)
		print(self.x_range)
		print(self.y_range)
		self.du = distance_unit
		self.x = self.get_range(self.x_range)
		self.y = self.get_range(self.y_range)

	def get_plot(self, v):
		p = np.linspace(-v, v, v*2/self.du)
		print(p)