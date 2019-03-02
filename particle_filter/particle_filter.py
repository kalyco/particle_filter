#!/usr/bin/env python3

import cv2 as cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import math
import time
from .img_map import ImgMap

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
		self.state = [[],[]]
		self.set_state()
		self.time_step = 0
		self.X_t = self.distribute_particles_randomly()

	def set_state(self):
		col = random.randint(0,self.cols-DISTANCE_UNIT)
		row = random.randint(0,self.rows-DISTANCE_UNIT)
		s = self.iMap.selection(row, col, DISTANCE_UNIT)
		self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
		self.state = self.iMap.origin(s)

	def distribute_particles_randomly(self):
		# TODO: give the particles a weight
		particles = []
		for p in range(M):
			p = [random.randint(0,self.cols), random.randint(0,self.rows)]
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = [0, 0, 0]
			particles.append(p)
		return particles

	def draw_world(self):
		while True:
			cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.name, self.cols, self.rows)
			self.get_measurement()
			cv2.imshow(self.name, self.img)
			self.action_step()
			cv2.waitKey(0)

	def get_measurement(self):
		v = self.state
		self.img[v[0][0]:v[0][1]+IMAGE[0], v[1][0]:v[1][1]+IMAGE[1]] = [0, 0 ,0]
		self.measurement = [
			[self.state[0][0], self.state[0][1]],
			[self.state[1][0], self.state[1][1]],
		]

	def resample(self):
		for p in self.X_t:
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = self.orig_img[p[0]:p[0]+10, p[1]:p[1]+10] = [100, 100, 100]
		self.distribute_particles_randomly()

	def action_step(self):
		print('action step')
		self.revert_colors(self.state)
					
	def revert_colors(self, area):
		old_state = self.iMap.reset_origin(area)

		# if len(self.state[0]):
		# 	for y in area[0]:
		# 		for x in area[1]:
