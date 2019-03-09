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

REF_U, DU, M, mu, sigma = 25, 50, 2000, 0, 1.0

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
			if k == 13: 
				self.step()
			elif chr(k) == '0': # Set robot location
				self.update_state(True)
			elif chr(k) == '1': # Scatter particles
				self.dist_particles()
			elif chr(k) == '2': # Inflate particles
				self.inflate()
			elif chr(k) == '3': # Resample set 
				self.resample()
			elif chr(k) == '4': # Move bot and shift particles
				self.move_agent()
				self.shift_particles()
			elif chr(k) == 'm': # Get reference image
				cv2.imshow('measurement', self.get_measurement(self.state))
			elif k == 27: # kill switch
				cv2.destroyAllWindows()
				break

	def step(self):
		self.update_state(self.dt == 0)
		self.dist_particles()
		self.inflate()
		self.resample()
		self.move_agent()
		self.shift_particles()

################### Step 0 #######################

	def update_state(self, init=False): # 0
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

################### Step 1 #######################

	def dist_particles(self, resampling=False): # 1
		self.P = []
		for i in range(M):
			if (resampling):
				row, col = self.iMap.to_image(self.samples[i][0], self.samples[i][1])
			else: 
				col = random.randint(0, self.cols-DU)
				row = random.randint(0, self.rows-DU)
			state = self.iMap.offset_vector(row, col)
			self.P.append({'state': state, 'prior': state, 'weight': 1/M})
			cv2.circle(self.img, (col, row), 8, (0,0,0), thickness=8)

################### Step 2 #######################

	def get_measurement(self, v): # 2.a
		i = self.iMap.to_image(v[0], v[1])
		ref = self.orig_img[i[0]-REF_U:i[0]+REF_U,i[1]-REF_U:i[1]+REF_U].copy()
		return ref

	def compare_grams(self): # 2.b
		oG = cv2.calcHist([self.get_measurement(self.state)],
		 [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		oG = cv2.normalize(oG, oG).flatten()
		for p in self.P:
			v = self.get_grams(p)
			coeff = cv2.compareHist(oG, v, cv2.HISTCMP_CORREL)
			p.update({'state': p['state'], 'prior': p['prior'], 'weight': coeff})
		self.P.sort(key=lambda e: e['weight'], reverse=True)

	def get_grams(self, p): # 2.c
		img = self.get_measurement(p['state'])
		hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		return cv2.normalize(hist, hist).flatten()

	def inflate(self): # 2.d
		self.compare_grams()
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

#################### Step 3 #######################

	def roulette_wheel_resample(self): # 3.a
		wheelVals = []
		weightSum = 0
		self.samples = []
		total = sum(list(map(lambda x: x['weight'], self.P)))
		for i in self.P:
			weightSum += i['weight']
			wheelVals.append(weightSum/total)
			x = random.random()
			idx = bisect.bisect(wheelVals, x)
			self.samples.append(self.P[idx]['state'])

	def resample(self): # 3.b
		self.roulette_wheel_resample()
		self.img = self.orig_img.copy()
		self.update_state()
		self.dist_particles(True)

################### Step 4 #######################

	def gen_control(self): # 4.a
		# ExCred: random_speed = random.uniform(minSpeed, maxSpeed)
		angle = random.uniform(0, 2.0*math.pi)
		self.control = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))]

	def move_agent(self): # 4.b
		self.gen_control()
		self.dt += 1
		self.state = self.get_movement(self.state)
		self.update_state()

	def get_movement(self, state): # 4.c
		state[0] += int(self.control[0] + np.random.normal(0, 1.0, 3000)[0]) # agent does not have access to noise
		state[1] += int(self.control[1] + np.random.normal(0, 1.0, 3000)[0])
		return state

	def shift_particles(self): # 4.d
		self.img = self.orig_img.copy()
		self.update_state()
		for p in self.P:
			p['state'] = self.get_movement(p['state'])
			row,col = self.iMap.to_image(p['state'][0], p['state'][1])
			cv2.circle(self.img, (col, row), 8, (0,0,0), thickness=8)	
