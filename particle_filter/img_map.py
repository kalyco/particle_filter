import numpy as np

class ImgMap:
	def __init__(self, img, distance_unit):
		self.img = img
		self.x_range = np.floor(img.shape[1]/2)
		self.y_range = np.floor(img.shape[0]/2)
		self.du = distance_unit
		self.x = self.get_plot(self.x_range)
		self.y = self.get_plot(self.y_range)

	def get_plot(self, v):
		return np.linspace(-v, v, v*2/self.du)

	def offset(self, s):
		x_mod =  1500 if s[0][1] < 1500 else -1500
		y_mod = 1500 if s[1][1] < 1500 else -1500
		return [
			[s[0][0]+x_mod, s[0][1]+x_mod],
			[s[1][0]+y_mod, s[1][1]+y_mod]]
