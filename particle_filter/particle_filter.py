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

REF_U = 25
DU = 50
M = 2000 # number of particles

class ParticleFilter():
	def __init__(self, name, img):
		self.name = name
		self.img = img
		self.orig_img = img.copy()
		self.rows = img.shape[0]
		self.cols = img.shape[1]
		self.iMap = ImgMap(self.img, DU)
		self.update_state(True) # [x,y]
		self.time_step = 0
		self.get_refs()
		self.distribute_particles_randomly()
		self.store_histograms()

	def update_state(self, init=False):
		if (init):
			col = random.randint(0, self.cols-DU)
			row = random.randint(0, self.rows-DU)
			s = self.iMap.selection(row, col, REF_U)
			self.state = self.iMap.offset_vector(row,col)
			print(self.state)
		else:
			y = self.state[1]
			x = self.state[0]
			m = self.iMap.to_image(x,y)
			s = [[m[0]-REF_U,m[0]+REF_U],[m[1]-REF_U,m[1]+REF_U]]
		self.img[s[0][0]:s[0][1], s[1][0]:s[1][1]] = [74, 69, 255]
		cv2.circle(
			self.img,
			(s[1][0],s[0][0]),
			4*REF_U,
			(0,0,0),
			thickness=4)

	def draw_world(self):
		self.action_step()
		self.compare_grams()
		# self.convolutional_images()
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
			elif k == 27: # end processing
				cv2.destroyAllWindows()
				break
			elif chr(k) == 'm': # show measurement
				cv2.imshow('measurement', self.get_measurement(self.state))
			elif chr(k) == 'r': # show best ref image
				cv2.imshow('ref', self.get_measurement(self.refs[0]['image']))
			elif k == 13: 
				self.move()
			elif chr(k) == 'c':
				image = self.convolutional_images()
				# cv2.namedWindow('conv', cv2.WINDOW_NORMAL)
				# cv2.resizeWindow(self.name, 1000, 1500)
				# cv2.imshow('conv', image)
			else:
				k = 0

################ REFS ######################

	def get_refs(self):
		self.refs = []
		N = int((self.rows/REF_U)-DU)
		for row in range(N):
			for col in range(N):
				img = self.iMap.offset_vector(row*REF_U,col*REF_U)
				ob = {
					'loc': 'ref' + str(row) + str(col),
					'image': img
				}
				self.refs.append(ob)
		self.store_histograms()

	def store_histograms(self):
		for k in self.refs:
			v = k['image']
			img = self.get_measurement(v)
			hist1 = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
			hist1 = cv2.normalize(hist1, hist1).flatten()
			k.update({
				'loc': k['loc'],
				'image': v,
				'histogram': hist1
			})

	def compare_grams(self):
		oG = cv2.calcHist([self.get_measurement(self.state)],
		 [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
		oG = cv2.normalize(oG, oG).flatten()
		for k in self.refs:
			v = k['histogram']
			coeff = cv2.compareHist(oG, v, cv2.HISTCMP_CORREL)
			k.update({
				'loc': k['loc'],
				'image': k['image'],
				'histogram': v,
				'corr_coeff': coeff
			})
		self.refs.sort(key=lambda e: e['corr_coeff'], reverse=True)

	def ref_histogram(self, s):
		plt.hist(s.ravel(),256,[0,256])
		# plt.draw()
		# plt.show()
		# plt.pause(0.03)

##################### /REFS ######################


################# PARTICLES ######################

	def distribute_particles_randomly(self):
		particles = []
		for p in range(M):
			p = [random.randint(0, self.cols),random.randint(0, self.rows), 1/M]
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = [0, 0, 0]
			particles.append(p)
		self.X_t = particles

	def resample(weights):
		n = len(weights)
		indices = []
		C = [0.] + [sum(weights[:i+1]) for i in range(n)]
		u0, j = random(), 0
		for u in [(u0+i)/n for i in range(n)]:
			while u > C[j]:
				j+=1
			indices.append(j-1)
		return indices

	def resample(self):
		# insert gaussian dist here
		for p in self.X_t:
			self.img[p[0]:p[0]+10, p[1]:p[1]+10] = self.orig_img[p[0]
					:p[0]+10, p[1]:p[1]+10] = [100, 100, 100]
		# self.distribute_particles_randomly()

	# def set_weights(self):
		


################# /PARTICLES ####################

################# MOVEMENT #####################

	def action_step(self):
		self.revert_colors(self.state)

	def revert_colors(self, area):
		return 0

	def get_measurement(self, v):
		i = self.iMap.to_image(v[0], v[1])
		i[0] = i[0] if i[0] > 0 else i[0] + DU
		i[1] = i[1] if i[1] > 0 else i[1] + DU
		i[0] = i[0] if i[0] < 3000 else i[0] - DU
		i[1] = i[1] if i[1] < 3000 else i[1] - DU
		ref = self.orig_img[i[0]-REF_U:i[0]+REF_U,i[1]-REF_U:i[1]+REF_U].copy()
		return ref

	def move(self): #parametric representation of curves
		self.img = self.orig_img
		# random_speed = random.uniform(minSpeed, maxSpeed)
		# hypoteneuse = opposite sq * adj sqr
		angle = random.uniform(0, 2.0*math.pi)
		self.movement_vector = [math.floor(DU * math.cos(angle)), math.floor(DU * math.sin(angle))]
		self.state[0] += self.movement_vector[0]
		self.state[1] += self.movement_vector[1]
		self.update_state()
		# dx^2 + dy^2 = 50

	def convolutional_images(self):
		# referencing https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
		# C_MATCHERS = ['cv2.TM_CCOEFF','cv2.TM_CCOEFF_NORMED','cv2.TM_CCORR',
		# 	'cv2.TM_CCORR_NORMED','cv2.TM_SQDIFF','cv2.TM_SQDIFF_NORMED']
		template = self.get_measurement(self.state)
		img = self.orig_img.copy()

		res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
		# top_left = min_loc if match in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'] else max_loc
		top_left = max_loc 
		bottom_right = (top_left[0] + REF_U, top_left[1] + REF_U)
		i = self.img[top_left[0]-REF_U:v[0]+REF_U,v[1]-REF_U:v[1]+REF_U].copy()
		# return i

################# /MOVEMENT ####################


################ PREDICTION #####################

################ /PREDICTION #####################