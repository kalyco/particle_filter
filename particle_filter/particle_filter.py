#!/usr/bin/env python3
from .img_map import ImgMap
import time
import math
import cv2 as cv2
import numpy as np
import random
import bisect
import collections
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal 

REF_U, DU, M, mu, sigma = 25, 50, 1000, 0, 1.0

class ParticleFilter():
	def __init__(self, name, img):
		self.name = name
		self.img = img
		self.rows = self.img.shape[0]
		self.cols = self.img.shape[1]
		self.orig_img = img.copy()
		self.iMap = ImgMap(self.img, DU)
		self.dt = 0

	def draw_world(self):
		while True:
			cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.name, 1000, 1500)
			cv2.imshow(self.name, self.img)
			wait_key = 33
			k = cv2.waitKey(wait_key) & 0xff
			if k == 27: # kill switch
				cv2.destroyAllWindows()
				break
			elif chr(k) == '1': # Set robot location
				self.update_state(True)
			elif chr(k) == '2': # Scatter particles randomly
				self.dist_particles()
			elif chr(k) == '3': # Robot gets reference image
				cv2.imshow('measurement', self.get_measurement(self.state))
			elif chr(k) == '4': # Inflate particles
				self.compare_grams()
				self.inflate()
			elif chr(k) == '5': # Resample set 
				self.resample()
			elif chr(k) == '6': # Move bot
				self.move()
			# elif chr(k) == 'p': # show best particle
			# 	# cv2.imshow('p', self.P[0]['image'])
			elif k == 13: 
				self.move()
			else:
				k = 0

	def update_state(self, init=False): # Step 1
		if (init):
			col = random.randint(0, self.cols-DU)
			row = random.randint(0, self.rows-DU)
			self.state = self.iMap.offset_vector(row,col)
			s = self.iMap.selection(row, col, REF_U)
		else:
			row,col = self.iMap.to_image(self.state[0], self.state[1])
			s = self.iMap.selection(row, col, REF_U)
		self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
		cv2.circle(self.img, (s[1][0], s[0][0]), 4*REF_U, (0,0,0), thickness=4)

	def dist_particles(self, resampling=False): #Step 2
		self.P = []
		for i in range(M):
			if (resampling):
				col = self.samples[i][0]
				row = self.samples[i][1]
			else: 
				col = random.randint(0, self.cols-DU)
				row = random.randint(0, self.rows-DU)
			state = self.iMap.offset_vector(row, col)
			self.P.append({'state': state, 'prior': state, 'weight': 1/M})
			cv2.circle(self.img, (col, row), 8, (0,0,0), thickness=8)

	def get_measurement(self, v): # Step 3
		i = self.iMap.to_image(v[0], v[1])
		ref = self.orig_img[i[0]-REF_U:i[0]+REF_U,i[1]-REF_U:i[1]+REF_U].copy()
		return ref

	def inflate(self): # Step 4
		for p in self.P:
			coeff = p['weight']
			[row,col] = self.iMap.to_image(p['state'][0], p['state'][1])
			radius = 24
			if (-1.0 < coeff and coeff <= -0.75):
				radius = 1
			elif (-0.75 < coeff and coeff <= -0.50):
				radius = 2
			elif (-0.50 < coeff and coeff <= -0.25):
				radius = 3
			elif (-0.25 < coeff and coeff <= 0):
				radius = 4
			elif (0 < coeff and coeff <= 0.25):
				radius = 5
			elif (0.25 < coeff and coeff >= 0.50):
				radius = 6
			elif (0.50 < coeff and coeff >= 0.75):
				radius = 8
			cv2.circle(self.img, (col, row), radius, (0, 102, 255), thickness=-1)

	def get_weights(self): # Step 5.a
		self.samples = []
		for m in range(M):
			self.samples.append(self.roulette_wheel_resample())

	def roulette_wheel_resample(self): # Step 5.b
		wheelVals = []
		weightSum = 0
		total = sum(list(map(lambda x: x['weight'], self.P)))
		for i in self.P:
			weightSum += i['weight']
			wheelVals.append(weightSum/total)
		x = random.random()
		idx = bisect.bisect(wheelVals, x)
		return self.P[idx]['state']

	def resample(self): # Step 5.c
		self.get_weights()
		self.img = self.orig_img.copy()
		self.update_state()
		self.dist_particles(True)

	def move(self): # Step 6
		self.compare_grams()
		self.inflate()
		self.dt += 1
		self.img = self.orig_img.copy()
		# Extra Credit: random_speed = random.uniform(minSpeed, maxSpeed)
		angle = random.uniform(0, 2.0*math.pi)
		self.control = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))]
		# self.state[0] += int(self.control[0] + np.random.normal(0, 1.0, 3000)[0]) # agent does not have access to noise
		# self.state[1] += int(self.control[1] + np.random.normal(0, 1.0, 3000)[0])
		self.update_state()

	def get_grams(self, p):
		img = self.get_measurement(p['state'])
		hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		return cv2.normalize(hist, hist).flatten()

	def compare_grams(self):
		oG = cv2.calcHist([self.get_measurement(self.state)],
		 [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		oG = cv2.normalize(oG, oG).flatten()
		for p in self.P:
			v = self.get_grams(p)
			coeff = cv2.compareHist(oG, v, cv2.HISTCMP_CORREL)
			p.update({'state': p['state'], 'prior': p['prior'], 'weight': coeff})
		self.P.sort(key=lambda e: e['weight'], reverse=True)

	def move_particles(self):
		xd = self.state
		negRes = 0
		posRes = 0
		for p in self.P:
			diff = (abs(p['state'][0] - self.state[0]) +
			 abs(p['state'][1] - self.state[1])) / 2
			d = 1 - p['weight']
			state = self.iMap.to_image(p['state'][0], p['state'][1])
			# if (d > 1):
			# 	cv2.circle(self.img, (state[0], state[1]), 8, (0,0,0), thickness=-1)
			# 	negRes += 1
			# else:
			# 	posRes += 1