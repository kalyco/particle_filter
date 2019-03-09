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
		self.update_state(True) # [x,y]
		self.distribute_particles_randomly()

	def update_state(self, init=False):
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

	def draw_world(self):
		self.compare_grams()
		self.inflate()
		self.move_partices()
		self.move()
		self.bayes_rule()
		while True:
			cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.name, 1000, 1500)
			# cv2.imshow(self.name, self.img)
			wait_key = 33
			k = cv2.waitKey(wait_key) & 0xff
			if k == 27: # end processing
				cv2.destroyAllWindows()
				break
			elif chr(k) == 'm': # show measurement
				cv2.imshow('measurement', self.get_measurement(self.state))
			elif chr(k) == 'p': # show best particle
				self.compare_grams()
				cv2.imshow('p', self.P[0]['image'])
			elif chr(k) == 'i':
				self.compare_grams()
				self.inflate()
			elif chr(k) == 'r':
				# self.resample()
				# self.roulette_wheel_resample()
				self.compare_grams()
				self.inflate()
				self.move_partices()
			elif k == 13: 
				self.move()
			else:
				k = 0

	def get_measurement(self, v):
		i = self.iMap.to_image(v[0], v[1])
		ref = self.orig_img[i[0]-REF_U:i[0]+REF_U,i[1]-REF_U:i[1]+REF_U].copy()
		return ref

	def move(self): #parametric representation of curves
		self.compare_grams()
		self.inflate()
		self.dt += 1
		self.img = self.orig_img.copy()
		self.move_partices()
		# Extra Credit: random_speed = random.uniform(minSpeed, maxSpeed)
		angle = random.uniform(0, 2.0*math.pi)
		self.control = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))]
		# self.state[0] += int(self.control[0] + np.random.normal(0, 1.0, 3000)[0]) # agent does not have access to noise
		# self.state[1] += int(self.control[1] + np.random.normal(0, 1.0, 3000)[0])
		self.update_state()


	def inflate(self):
		for p in self.P:
			coeff = p['weight']
			[row,col] = self.iMap.to_image(p['state'][0], p['state'][1])
			radius = 24
			if (-1.0 < coeff and coeff <= -0.75):
				radius = 0
			elif (-0.75 < coeff and coeff <= -0.50):
				radius = 0
			elif (-0.50 < coeff and coeff <= -0.25):
				radius = 0
			elif (-0.25 < coeff and coeff <= 0):
				radius = 1
			elif (0 < coeff and coeff <= 0.25):
				radius = 2
			elif (0.25 < coeff and coeff >= 0.50):
				radius = 3
			elif (0.50 < coeff and coeff >= 0.75):
				radius = 4
			cv2.circle(self.img, (col, row), 1, (0, 102, 255), thickness=6)

	def distribute_particles_randomly(self):
		self.P = []
		for p in range(M):
			col = random.randint(REF_U, self.cols-DU)
			row = random.randint(REF_U, self.rows-DU)
			state = self.iMap.offset_vector(row, col)
			p = {
				'state': state,
				'prior': state,
				'weight': 1/M,
				'image': self.get_measurement(state),
				'histogram': self.store_histogram(self.get_measurement(state))
			}
			cv2.circle(self.img, (col, row), 8, (0,0,0), thickness=-1)
			self.P.append(p)

	def bayes_rule(self):
		for m in self.P:
			print(m['prior'])
			# p(x|y,z) = p(y|x,z)*p(x|z)/p(y|z)
			x = multivariate_normal.pdf(self.state, 0, 1.0)
			y = multivariate_normal.pdf(self.control, 0, 1.0)
			z = multivariate_normal.pdf(m['prior'], 0, 1.0) 
			p1 = self.conditional_probability(y,(x*z)) # p(y|x,z)
			p2 = self.conditional_probability(x,z) # p(x|z)
			p3 = self.conditional_probability(y,z) # p(y|z)
			m['state'] = (p1*p2)/p3
			print(m['state'])

	def conditional_probability(self, a, b): # p(a|b)
		return (multivariate_normal.pdf(a, 0, 1.0)
		*multivariate_normal.pdf(b, 0, 1.0))
		/multivariate_normal.pdf(b, 0, 1.0)

	def move_partices(self):
		xd = self.state
		negRes = 0
		posRes = 0
		for p in self.P:
			diff = (abs(p['state'][0] - self.state[0]) +
			 abs(p['state'][1] - self.state[1])) / 2
			d = 1 - p['weight']
			state = self.iMap.to_image(p['state'][0], p['state'][1])
			if (d > 1):
				cv2.circle(self.img, (state[0], state[1]), 8, (0,0,0), thickness=-1)
				negRes += 1
			else:
				posRes += 1
		# print("neg quant")
		# print(negRes)
		# print("pos quant")
		# print(posRes)

	def roulette_wheel_resample(self):
		wheelVals = []
		weightSum = 0
		prevProbabilities = 0
		for p in self.P:
			weightSum += p['weight']
		for p in self.P:
			prevProbabilities += (p['weight'] / weightSum)
		nums = 0
		for i in range(M-1):
			num = np.random.uniform(0, 1.0)
			if (num > self.P[i]['weight'] and num > self.P[i+1]['weight']):
				nums += 1
		print(nums)


	def store_histogram(self, p):
			hist1 = cv2.calcHist([p], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			return cv2.normalize(hist1, hist1).flatten()

	def compare_grams(self):
		oG = cv2.calcHist([self.get_measurement(self.state)],
		 [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		oG = cv2.normalize(oG, oG).flatten()
		for p in self.P:
			v = p['histogram']
			coeff = cv2.compareHist(oG, v, cv2.HISTCMP_CORREL)
			p.update({
				'state': p['state'],
				'weight': coeff,
				'image': p['image'],
				'histogram': v,
			})
		self.P.sort(key=lambda e: e['weight'], reverse=True)



	# def convolutional_images(self):
	# 	template = self.get_measurement(self.state)
	# 	img = self.orig_img.copy()

	# 	res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
	# 	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	# 	# top_left = min_loc if match in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] else max_loc
	# 	top_left = max_loc 
	# 	bottom_right = (top_left[0] + REF_U, top_left[1] + REF_U)
	# 	i = self.img[top_left[0]-REF_U:v[0]+REF_U,v[1]-REF_U:v[1]+REF_U].copy()

			
	# def get_particle_reference(self, p):
	# 	possible_refs = []
	# 	row, col = self.iMap.to_image(p[0],p[1])
	# 	N = int((self.rows/REF_U)-DU)
	# 	r = abs(round(row/REF_U) - DU)
	# 	c = abs(round(col/REF_U) - DU)
	# 	ref_name ='ref' + str(r) + str(c)
	# 	ref = None
	# 	for r in self.refs:
	# 		if r['loc'] == ref_name:
	# 			p.append(ref_name)
	# 			break
	# 	if (not ref):


	# def ref_histogram(self, s):
	# 	plt.hist(s.ravel(),256,[0,256])
	# 	# plt.draw()
	# 	# plt.show()
	# 	# plt.pause(0.03)

	# def get_refs(self):
	# 	self.refs = []
	# 	N = int((self.rows/REF_U)-DU)
	# 	x = []
	# 	for row in range(N):
	# 		for col in range(N):
	# 			img = self.iMap.offset_vector(row*REF_U,col*REF_U)
	# 			x.append(img[0])
	# 			ob = {
	# 				'loc': 'ref' + str(row) + str(col),
	# 				'image': img
	# 			}
	# 			self.refs.append(ob)
	# 		self.store_histogram(ob['image'])
