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
			elif chr(k) == '0': # Set robot location (x,y)
				self.update_state(True)
			elif chr(k) == '1': # Scatter M particles
				self.dist_particles()
			elif chr(k) == '2': # Inflate particles
				self.inflate()
			elif chr(k) == '3': # Resample set 
				self.roulette_wheel_resample()
				# self.distance_based_resample()
			elif chr(k) == '4': # Move bot
				self.move_agent()
			elif chr(k) == '5': # Shift particles	
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
		self.roulette_wheel_resample()
		self.move_agent()
		self.shift_particles()

################### Step 0: Draw Map ####################

	def update_state(self, init=False): # 0
		if (init):
			col = random.randint(REF_U, self.cols-DU)
			row = random.randint(REF_U, self.rows-DU)
			self.state = self.iMap.offset_vector(row,col)
			s = self.iMap.selection(row, col, REF_U)
		else:
			row,col = self.iMap.to_image(self.state[0], self.state[1])
			s = self.iMap.selection(row, col, REF_U)
		self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
		cv2.circle(self.img, (s[1][0], s[0][0]), 4*REF_U, (0,0,0), thickness=4)

################# Step 1: Random Particles ####################

	def dist_particles(self): # 1
		self.redraw_world()
		self.P = []
		for i in range(M):
			col = random.randint(0, self.cols-DU)
			row = random.randint(0, self.rows-DU)
			state = self.iMap.offset_vector(row, col)
			self.P.append({'state': state, 'prior': state, 'weight': 1/M})
			cv2.circle(self.img, (col, row), 8, (0,0,0), thickness=8)

################ Step 2: Inflate Particles #######################

	def distance_based_resample(self):
		self.redraw_world()
		weightSum = 0
		for m in self.P:
			diff = self.avg_dist(m['state'], self.state)
			m['weight'] = 1 - diff
		wheelVals = []
		total = sum(list(map(lambda x: x['weight'], self.P)))
		noise = np.random.normal(0, 1.0, M)
		for i in range(M):	
			weightSum += self.P[i]['weight']
			wheelVals.append(weightSum/total)
			x = random.random()
			idx = bisect.bisect(wheelVals, x)
			self.P[i] = self.P[idx]
			self.P[i]['state'][0] += int(noise[i])
			self.P[i]['state'][1] += int(noise[i])
			self.redraw_point(self.P[i]['state'], 8)

	def avg_dist(self, s1, s2):
		return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])/2

	def compare_grams(self): # 2.a
		oG = cv2.calcHist([self.get_measurement(self.state)],
		 [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		oG = cv2.normalize(oG, oG).flatten()
		for p in self.P:
			v = self.get_grams(p)
			coeff = cv2.compareHist(oG, v, cv2.HISTCMP_CORREL)
			if (p['state'][0] > 1500 or p['state'][0] < -1500 or
			 p['state'][1] > 1500 or p['state'][1] < -1500):
				coeff = 0
			p.update({'state': p['state'], 'prior': p['prior'], 'weight': coeff})
		# self.P.sort(key=lambda e: e['weight'])

	def get_grams(self, p): # 2.b
		img = self.get_measurement(p['state'])
		hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		return cv2.normalize(hist, hist).flatten()

	def inflate(self): # 2.c
		self.compare_grams()
		self.redraw_world()
		for p in self.P:
			mid_rad = 40
			radius = np.floor(mid_rad + p['weight'] * mid_rad)
			self.redraw_point(p['state'], int(radius))

#################### Step 3: Resample particles #######################

	def roulette_wheel_resample(self): # 3.a
		self.redraw_world()
		wheelVals = []
		weightSum = 0
		total = sum(list(map(lambda x: x['weight'], self.P)))
		noise = np.random.normal(0, 1.0, M)
		for i in range(M):
			weightSum += self.P[i]['weight']
			wheelVals.append(weightSum/total)
			x = random.random()
			idx = bisect.bisect(wheelVals, x)
			self.P[i] = self.P[idx]
			self.P[i]['state'][0] += int(noise[i])
			self.P[i]['state'][1] += int(noise[i])
			self.redraw_point(self.P[i]['state'], 8)

################### Step 4: Move agent #######################

	def gen_control(self): # 4.a
		# ExCred: random_speed = random.uniform(minSpeed, maxSpeed)
		angle = random.uniform(0, 2.0*math.pi)
		self.control = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))]

	def move_agent(self): # 4.b
		self.gen_control()
		self.dt += 1
		self.state = self.get_movement(self.state)
		self.actual_state = [ # agent does not have access to noise 
			self.state[0] + int(np.random.normal(0, 1.0)),
			self.state[0] + int(np.random.normal(0, 1.0)),
		]
		self.redraw_world()
		for p in self.P:
			self.redraw_point(p['state'], 8)

################### Step 5: Shift particles #######################

	def shift_particles(self): # 5
		self.redraw_world()
		for p in self.P:
			p['state'] = self.get_movement(p['state'])
			self.redraw_point(p['state'], 8)


################### Helper Functions #######################

	def get_measurement(self, v):
		i = self.iMap.to_image(v[0], v[1])
		ref = self.orig_img[i[0]-REF_U:i[0]+REF_U,i[1]-REF_U:i[1]+REF_U].copy()
		return ref
	
	def redraw_point(self, s, radius):
		row,col = self.iMap.to_image(s[0], s[1])
		cv2.circle(self.img, (col, row), radius, (0,0,0), thickness=8)

	def get_movement(self, state):
		state[0] += int(self.control[0])
		state[1] += int(self.control[1])
		return state

	def redraw_world(self):
		self.img = self.orig_img.copy()
		self.update_state()