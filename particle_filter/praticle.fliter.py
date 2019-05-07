#!/usr/bin/env python3
import sys
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
from scipy.stats import norm

REF_U, DU, M, mu, sigma = 75, 50, 1000, 0, 1.0

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
			cv2.resizeWindow(self.name, 1000, 1000)
			cv2.imshow(self.name, self.img)
			wait_key = 33
			k = cv2.waitKey(wait_key) & 0xff
			if k == 13: 
				self.step()
			elif chr(k) == '0': # Set drone state (x,y)
				self.update_state(True)
			elif chr(k) == '1': # Scatter M particles
				self.dist_particles()
			elif chr(k) == '2': # Inflate particles
				self.inflate()
			elif chr(k) == '3': # Resample set 
				self.roulette_index_resample()
			elif chr(k) == '4': # Move bot
				self.move_agent()
			elif chr(k) == '5': # Control resample p(x_t|x_t-1,y_t)
				self.control_resample()
			elif chr(k) == 'm': # Get reference image
				cv2.imshow('measurement', self.get_measurement(self.state))
			elif chr(k) == 'p': # Get first particle reference image
				cv2.imshow('p_measurement', self.get_measurement(self.P[0]['state']))	
				cv2.destroyAllWindows()
				break

	def step(self):
		self.update_state(self.dt == 0)
		self.dist_particles()
		self.inflate()
		self.roulette_index_resample()
		self.move_agent()
		self.control_resample()

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

	def compare_images(self):
		state_img = self.get_measurement(self.state)
		pixel_ct = float(state_img.shape[0] * state_img.shape[1])
		relevant_particles = []
		for i in range(len(self.P)):
			m = self.P[i]
			diffs = []
			p = self.get_measurement(m['state'])
			diffs.append((self.state[0] - m['state'][0]) ** 2) #x
			diffs.append((self.state[1] - m['state'][1]) ** 2) #y
			try:
				pixel_diffs = np.sum((state_img.astype('float') - p.astype('float')) ** 2)
				diffs.append(pixel_diffs)
				sse = np.sum(diffs) / pixel_ct
				m['sse'] = sse
				relevant_particles.append(m)
			except:
				pass
		relevant_particles.sort(key=lambda e: e['sse'])
		max_err = relevant_particles[-1]['sse']
		total = sum(list(map(lambda x: max_err - x['sse'], relevant_particles)))
		for m in self.P:
			try:
				ratio = (max_err - m['sse']) / total
				m['weight'] = ratio
			except:
				m['weight'] = 0

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


	def get_grams(self, p): # 2.b
		img = self.get_measurement(p['state'])
		hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		return cv2.normalize(hist, hist).flatten()

	def inflate(self): # 2.c
		# self.compare_grams() for histogram comparison
		self.compare_images()
		self.redraw_world()
		for p in self.P:
			mid_rad = 40000
			radius = np.floor(p['weight'] * mid_rad)
			self.redraw_point(p['state'], int(radius))

#################### Step 3: Resample particles #######################

	def roulette_index_resample(self): # 3.a
		self.redraw_world()
		wheelVals = []
		weightSum = 0
		total = sum(list(map(lambda x: x['weight'], self.P)))
		noise = np.random.normal(0, 10.0, M)
		particles = self.P
		self.P = []
		for i in range(len(particles)):
			p = particles[i]
			weightSum += p['weight']
			wheelVals.append(weightSum/total)
			x = random.random()
			idx = bisect.bisect(wheelVals, x)
			new_p = particles[idx]
			new_p['prior'] = particles[idx]['state'] 
			self.P.append(particles[idx])
			self.redraw_point(self.P[-1]['state'], 8)

################### Step 4: Move agent #######################

	def gen_control(self): # 4.a
		# ExCred: random_speed = random.uniform(minSpeed, maxSpeed)
		angle = random.uniform(0, 2.0*math.pi)
		self.control = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))]

	def move_agent(self): # 4.b
		self.gen_control()
		self.dt += 1
		self.get_movement(self.state)
		self.actual_state = [ # agent does not have access to noise 
			self.state[0] + int(np.random.normal(0, 1.0)),
			self.state[0] + int(np.random.normal(0, 1.0)),
		]
		self.redraw_world()
		for p in self.P:
			self.redraw_point(p['state'], 8)

################### Step 5: Shift particles #######################

	def control_resample(self): # 5
		self.bayes_rule()
		self.redraw_world()
		for p in self.P:
			self.get_movement(p['state'])
			self.redraw_point(p['state'], 8)

	def bayes_rule(self):
		for m in self.P:
			# p(x|x_prior,u) = p(x_prior|x,u)*p(x|u)/p(x_prior|u)
			x = m['state'] 
			x_prior = m['prior'] 
			u = self.control
			xu = [(x[0]*u[0]), (x[1]*u[1])]	

			x1 = self.conditional_probability(x_prior[0],xu[0]) # p(x_prior|x,u)
			y1 = self.conditional_probability(x_prior[1],xu[1]) # p(x_prior|x,u)
			x2 = self.conditional_probability(x[0],x_prior[0]) # p(x|x_prior)
			y2 = self.conditional_probability(u[1],x_prior[1]) # p(u|x_prior)
			x3 = self.conditional_probability(x[0],x_prior[0]) # p(x|x_prior)
			y3 = self.conditional_probability(u[1],x_prior[1]) # p(u|x_prior)
			new_state = [int((x1*x2)/x3), int((y1*y2)/y3)]
			m['prior'] = m['state']
			m['state'] = new_state

	def conditional_probability(self, a, b): # p(a|b)
		return (np.random.normal(a) * np.random.normal(b)) / np.random.normal(b)

################### Helper Functions #######################

	def get_measurement(self, v):
		i = self.iMap.to_image(v[0], v[1])
		ref = self.orig_img[i[0]-REF_U:i[0]+REF_U,i[1]-REF_U:i[1]+REF_U].copy()
		return ref
	
	def redraw_point(self, s, radius):
		if (np.sign(radius) == 1):
			row,col = self.iMap.to_image(s[0], s[1])
			cv2.circle(self.img, (col, row), radius, (0,0,0), thickness=8)

	def get_movement(self, state):
		state[0] += int(self.control[0])
		state[1] += int(self.control[1])

	def redraw_world(self):
		self.img = self.orig_img.copy()
		self.update_state()