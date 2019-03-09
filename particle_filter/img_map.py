import numpy as np

class ImgMap:
	def __init__(self, img, DU):
		self.img = img
		self.x_range = np.floor(img.shape[1]/2)
		self.y_range = np.floor(img.shape[0]/2)
		self.du = DU
		self.x = self.get_plot(self.x_range)
		self.y = self.get_plot(self.y_range)

	def get_plot(self, v):
		return np.linspace(-v, v, v*2/self.du)

	def offset_matrix(self, s):
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

	def offset_vector(self, row, col):
		if (col <= 1500):
			x = -1500+col
		else:
			x = col-1500
		y = 1500-row
		return [x,y]

	def to_image(self, x, y):
		col = x+1500
		if (y <= 0):
			row = 1500-y
		else:
			row = 1500-y
		return [row,col]		

	def selection(self, row, col, r):
		s = [[row,row+r], [col,col+r]]
		return s