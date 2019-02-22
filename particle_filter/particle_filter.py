import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import math
from img_map import ImgMap

IMAGE = [100,100]
DISTANCE_UNIT = 49
M = 1000 # number of particles
# Img Size: 27000000
# Shape: [3000, 3000, 3]

class ParticleFilter():
	def __init__(self, name, img):
		self.name = name
		self.img = img
		self.rows = img.shape[0]
		self.cols = img.shape[1]
		self.prior_state = self.set_state()
		self.X_t = self.distribute_particles_randomly()
		self.map = ImgMap(self.img, DISTANCE_UNIT)
		# todo: set map point

	def set_state(self):
		col_range = random.randint(0,self.cols-DISTANCE_UNIT)
		row_range = random.randint(0,self.rows-DISTANCE_UNIT)
		prior_state = [[col_range,col_range+DISTANCE_UNIT], [row_range,row_range+DISTANCE_UNIT]]
		self.img[prior_state[0][0]:prior_state[0][1], prior_state[1][0]:prior_state[1][1]] = [74, 69, 255]
		return prior_state

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
		cv.waitKey(0)
		cv.destroyAllWindows()
