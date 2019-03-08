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
			# row = random.randint(REF_U, self.rows-DU)		
			row = random.randint(REF_U, 1500)		
			self.state = self.iMap.offset_vector(row,col)
			s = self.iMap.selection(row, col, REF_U)
			# print(self.state)
		else:
			row,col = self.iMap.to_image(self.state[0], self.state[1])
			s = self.iMap.selection(row, col, REF_U)
		self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
		cv2.circle(self.img, (s[1][0], s[0][0]), 4*REF_U, (0,0,0), thickness=4)

	def draw_world(self):
		while True:
			cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
			cv2.resizeWindow(self.name, 1000, 1500)
			cv2.imshow(self.name, self.img)
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
			elif chr(k) == 'r':
				self.compare_grams()
				self.resample()
			elif k == 13: 
				self.move()
			else:
				k = 0

	def approximator(self):
		for p in self.P:
			print()

	def get_measurement(self, v):
		i = self.iMap.to_image(v[0], v[1])
		ref = self.orig_img[i[0]-REF_U:i[0]+REF_U,i[1]-REF_U:i[1]+REF_U].copy()
		return ref

	def move(self): #parametric representation of curves
		self.dt += 1
		self.img = self.orig_img.copy()
		# Extra Credit: random_speed = random.uniform(minSpeed, maxSpeed)
		angle = random.uniform(0, 2.0*math.pi)
		self.control = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))]
		self.state[0] += int(self.control[0] + np.random.normal(0, 1.0, 3000)[0]) # agent does not have access to noise
		self.state[1] += int(self.control[1] + np.random.normal(0, 1.0, 3000)[0])
		self.update_state()

	def compare_grams(self):
		oG = cv2.calcHist([self.get_measurement(self.state)],
		 [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		oG = cv2.normalize(oG, oG).flatten()
		for p in self.P:
			v = p['histogram']
			coeff = cv2.compareHist(oG, v, cv2.HISTCMP_CORREL)
			p.update({
				'state': p['state'],
				'weight': p['weight'],
				'image': p['image'],
				'histogram': v,
				'corr_coeff': coeff
			})
		self.P.sort(key=lambda e: e['corr_coeff'], reverse=True)

	def resample(self):
		for p in self.P:
			coeff = p['corr_coeff']
			[row,col] = self.iMap.to_image(p['state'][0], p['state'][1])
			print(row, col)
			if (-1.0 < coeff and coeff <= -0.75):
				cv2.circle(self.img, (col, row), 2, (0, 102, 255), thickness=6)
			elif (-0.75 < coeff and coeff <= -0.50):
				cv2.circle(self.img, (col, row), 4, (0, 102, 255), thickness=6)	
			elif (-0.50 < coeff and coeff <= -0.25):
				cv2.circle(self.img, (col, row), 6, (0, 102, 255), thickness=6)
			elif (-0.25 < coeff and coeff <= 0):
				cv2.circle(self.img, (col, row), 8, (0, 102, 255), thickness=6)
			elif (0 < coeff and coeff <= 0.25):
				cv2.circle(self.img, (col, row), 12, (0, 102, 255), thickness=6)
			elif (0.25 < coeff and coeff >= 0.50):
				cv2.circle(self.img, (col, row), 16, (0, 102, 255), thickness=6)
			elif (0.50 < coeff and coeff >= 0.75):
				cv2.circle(self.img, (col, row), 18, (0, 102, 255), thickness=6)
			else:
				cv2.circle(self.img, (col, row), 20, (0, 102, 255), thickness=6)
				# self.img[row-20:row+20, col-20:col+20] = [0, 0, 255]

	def distribute_particles_randomly(self):
		self.P = []
		for p in range(M):
			col = random.randint(REF_U, self.cols-DU)
			row = random.randint(REF_U, self.rows-DU)
			state = self.iMap.offset_vector(row, col)
			# print(state)
			p = {
				'state': state,
				'weight': 1/M,
				'image': self.get_measurement(state),
				'histogram': self.store_histogram(self.get_measurement(state))
			}
			cv2.circle(self.img, (col, row), 8, (0,0,0), thickness=-1)
			self.P.append(p)

	def store_histogram(self, p):
			hist1 = cv2.calcHist([p], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			return cv2.normalize(hist1, hist1).flatten()






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
