#!/usr/bin/env python3

from .img_map import ImgMap
import time
import math
import cv2 as cv2
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

IMAGE = [25, 25]
DISTANCE_UNIT = 50
M = 2000  # number of particles
# Img Size: 27000000
# Shape: [3000, 3000, 3]


class ParticleFilter():
	def __init__(self, name, img):
		self.name = name
		self.img = img
		self.orig_img = img.copy()
		self.rows = img.shape[0]
		self.cols = img.shape[1]
		self.iMap = ImgMap(self.img, DISTANCE_UNIT)
		self.state = [[], []]
		self.set_state()
		self.time_step = 0
		self.get_reference_images()
		self.distribute_particles_randomly()
		self.store_histograms()

	def set_state(self):
		col = random.randint(0, self.cols-DISTANCE_UNIT)
		row = random.randint(0, self.rows-DISTANCE_UNIT)
		s = self.iMap.selection(row, col, DISTANCE_UNIT)
		self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
		self.state = s

	def distribute_particles_randomly(self):
		particles = []
		for p in range(M):
			p = [random.randint(0, self.cols),random.randint(0, self.rows), 1/M]
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = [0, 0, 0]
			particles.append(p)
		self.X_t = particles

	def draw_world(self):
		self.action_step()
		self.get_measurement(self.state)
		self.compare_grams()
		# self.ref_histogram(self.measurement)
		while True:
			cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.name, 1000, 1500)
			cv2.imshow(self.name, self.img)
			wait_key = 33
			k = cv2.waitKey(wait_key) & 0xff
			if chr(k) == 's': # start running
				wait_key = 33
			elif chr(k) == 'p': # pause between frames
				wait_key = 0
			elif k == 27:  # end processing
				cv2.destroyAllWindows()
				break
			# elif chr(k) == 'm':  # show reference image
			#   cv2.imshow('measurement', self.measurement)
			else:
				k = 0

	def resample(self):
		# insert gaussian dist here
		for p in self.X_t:
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = self.orig_img[p[0]
					:p[0]+10, p[1]:p[1]+10] = [100, 100, 100]
		# self.distribute_particles_randomly()

	def action_step(self):
		print('action step')
		self.revert_colors(self.state)

	def revert_colors(self, area):
		old_state = self.iMap.reset_origin(area)

	def get_measurement(self, v):
		self.measurement = self.orig_img[
			v[0][0]-IMAGE[0]:v[0][1]+IMAGE[0],
			v[1][0]-IMAGE[0]:v[1][1]+IMAGE[1]].copy()

	def get_reference_images(self):
		self.rImages = []
		N = int(self.rows/DISTANCE_UNIT)
		for row in range(N):
			for col in range(N):
				s = [[row*DISTANCE_UNIT, (row*DISTANCE_UNIT)+DISTANCE_UNIT],
						[col*DISTANCE_UNIT, (col*DISTANCE_UNIT)+DISTANCE_UNIT]]
				self.rImages.append(s)

	def ref_histogram(self, s):
		plt.hist(s.ravel(),256,[0,256])
		# plt.draw()
		# plt.show()
		# plt.pause(0.03)

	def store_histograms(self):
		self.grams = []
		for i in self.rImages:
			img = self.img[i[0][0]:i[0][1], i[1][0]:i[1][1]]	
			hist1 = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			hist1 = cv2.normalize(hist1, hist1).flatten()
			self.grams.append(hist1)
		print(len(self.grams))
		print(self.grams[0])

	def compare_grams(self):
		oG = cv2.calcHist([self.measurement], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		oG = cv2.normalize(oG, oG).flatten()
		results = []
		for g in self.grams:
			results.append(cv2.compareHist(oG, g, cv2.HISTCMP_CORREL))
		return results

	def get_origin(self, s):
		return self.iMap.offset(s)
