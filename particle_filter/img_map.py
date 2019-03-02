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
		delta = s[0][1] - s[0][0]
		if (s[1][1] >= 1500):
			x = [s[1][0]-1500,s[1][0]-1500+delta]
		else:
		  x = [(-1500+delta)+s[1][0], (-1500-delta)+s[1][1]]
		if (s[0][0] <= 1500):
			y = [1500-s[0][0], 1500-s[0][1]+2*delta]
		else: 
			y = [1500-s[0][0], 1500-s[0][1]]
		o = [x,y]
		return o

	def reset_origin(self, s):
		x1,x2,y1,y2 = s[0][0],s[0][1],s[1][0],s[1][1]
		

	def selection(self, row, col, r):
		s = [[row,row+r], [col,col+r]]
		return s
