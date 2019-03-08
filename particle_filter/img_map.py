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
		x = col-1500 if col >= 1500 else -1500+col
		y = 1500-row
		return [x,y]

	def to_image(self, x, y):
		col = x+1500
		if y >= 0:
			row = y+1500 
		else:
			row = 1500-y
		return [row,col]		

	def image_w_padding(self, x, y):
		i = self.to_image(x, y)
		i[0] = i[0] if i[0] > 0 else i[0] + DU
		i[1] = i[1] if i[1] > 0 else i[1] + DU
		i[0] = i[0] if i[0] < 3000 else i[0] - DU
		i[1] = i[1] if i[1] < 3000 else i[1] - DU
		return i

	def selection(self, row, col, r):
		s = [[row,row+r], [col,col+r]]
		return s