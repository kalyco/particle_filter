# def calc_distance(self):
# xd = self.state
# 		negRes = 0
# 		posRes = 0
# 		for p in self.P:
# 			diff = (abs(p['state'][0] - self.state[0]) +
# 			 abs(p['state'][1] - self.state[1])) / 2
# 			d = 1 - p['weight']
# 			state = self.iMap.to_image(p['state'][0], p['state'][1])
# 			# if (d > 1):
# 			# 	cv2.circle(self.img, (state[0], state[1]), 8, (0,0,0), thickness=-1)
# 			# 	negRes += 1
# 			# else:
# 			# 	posRes += 1

	# def conditional_probability(self, a, b): # p(a|b)
	# 	return (multivariate_normal.pdf(a, 0, 1.0)
	# 	* multivariate_normal.pdf(b, 0, 1.0))
	# 	/ multivariate_normal.pdf(b, 0, 1.0)

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

	# def roulette_wheel_test(self):
	# 	# for each point get its probability and append to an array
	# 	# draw from this distribution
	# 	wheelVals = []
	# 	# get sum of all weights
	# 	total = sum(list(map(lambda x: x['weight'], self.P)))
	# 	# get a random value 
	# 	randomVal = random.random() * total
	# 	for i in range(M):
	# 		# locate random value based on weights
	# 		randomVal -= self.P[i]['weight']
	# 		if (randomVal < 0):
	# 			return i
	# 	# when rounding errors occur, return the last item's index		
	# 	return self.P[-1]['weight']
	# 	# list(map(lambda x: wheelVals.append(x['weight']/total), self.P))