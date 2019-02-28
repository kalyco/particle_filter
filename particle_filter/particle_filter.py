import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import math
from img_map import ImgMap

IMAGE = [25,25]
DISTANCE_UNIT = 50
M = 1000 # number of particles
# Img Size: 27000000
# Shape: [3000, 3000, 3]

class ParticleFilter():
	def __init__(self, name, img):
		self.name = name
		self.img = img
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
			particles.append(p)
		return particles

	def draw_world(self):
		cv.namedWindow(self.name, cv.WINDOW_NORMAL)
		cv.resizeWindow(self.name, self.cols, self.rows)
		cv.imshow(self.name, self.img)
		cv.waitKey(1)
		cv.destroyAllWindows()

	def get_measurement(self):
		s = self.state
		crop_img = self.img[s[0][0]:s[0][1]+IMAGE[0], s[1][0]:s[1][1]+IMAGE[1]]
		cv.namedWindow('cropped', cv.WINDOW_NORMAL)
		print(crop_img[0])
		cv.resizeWindow(self.img, crop_img[0][1], crop_img[1][1])
		self.cImg = cv.imshow("cropped", crop_img)

	# def show_origin(self):
	# 	prev_state = [[0,DISTANCE_UNIT], [0,DISTANCE_UNIT]]
	# 	s = self.iMap.offset(prev_state)
	# 	state = self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
	# 	return state, prev_state
