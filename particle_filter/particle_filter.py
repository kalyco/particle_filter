import cv2 as cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import math
import time
from img_map import ImgMap

IMAGE = [25,25]
DISTANCE_UNIT = 50
M = 2000 # number of particles
# Img Size: 27000000
# Shape: [3000, 3000, 3]

class ParticleFilter():
	def __init__(self, name, img):
		self.name = name
		self.img = img
		self.orig_img = img
		self.rows = img.shape[0]
		self.cols = img.shape[1]
		self.iMap = ImgMap(self.img, DISTANCE_UNIT)
		self.state = self.set_state()
		self.time_step = 0
		self.X_t = self.distribute_particles_randomly()

	def set_state(self):
		col_range = random.randint(0,self.cols-DISTANCE_UNIT)
		row_range = random.randint(0,self.rows-DISTANCE_UNIT)
		prev_state = [[col_range,col_range+DISTANCE_UNIT], [row_range,row_range+DISTANCE_UNIT]]
		s = self.iMap.offset(prev_state)
		self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
		return prev_state

	def distribute_particles_randomly(self):
		particles = []
		for p in range(M):
			p = [random.randint(0,self.cols), random.randint(0,self.rows)]
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = [0, 0, 0]
			particles.append(p)
		return particles

	def draw_world(self):
			cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.name, self.cols, self.rows)
			self.get_measurement()
			cv2.imshow(self.name, self.img)
			cv2.waitKey(0)
			# self.make_moves()

	def get_measurement(self):
		v = self.iMap.offset([
			[self.state[0][0], self.state[0][1]],
			[self.state[1][0], self.state[1][1]],
		])
		self.img[v[0][0]:v[0][1]+IMAGE[0], v[1][0]:v[1][0]+IMAGE[1]] = [0, 0 ,0]
		# cv2.imshow("crop", imCrop)
		# cv2.waitKey(1)
		# cv2.destroyAllWindows()

	def resample(self):
		for p in self.X_t:
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = self.orig_img[p[0]:p[0]+10, p[1]:p[1]+10] = [100, 100, 100]
		self.distribute_particles_randomly()


	def make_moves(self):
		# for t in range(5):
			time.sleep(5)
			self.resample()
		# 	self.get_measurement()
		# 	self.draw_world()


		# s = self.state
		# r = 100.0 / self.img.shape[1]
		# dim = (25, int(self.img.shape[0]*r))
		# crop_img = self.img[s[0][0]:s[0][1]+IMAGE[0], s[1][0]:s[1][1]+IMAGE[1]]
		# cv2.namedWindow('cropped', cv2.WINDOW_NORMAL)
		# print(s)
		# cv2.resizeWindow(self.img, dim) # remember to offset
		# self.cImg = cv2.imshow("cropped", crop_img)

	# def show_origin(self):
	# 	prev_state = [[0,DISTANCE_UNIT], [0,DISTANCE_UNIT]]
	# 	s = self.iMap.offset(prev_state)
	# 	state = self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
	# 	return state, prev_state
