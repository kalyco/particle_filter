	# def bayes_rule(self):
	# 	for m in self.P:
	# 		print(m['prior'])
	# 		# p(x|y,z) = p(y|x,z)*p(x|z)/p(y|z)
	# 		x = multivariate_normal.pdf(self.state, 0, 1.0)
	# 		y = multivariate_normal.pdf(self.control, 0, 1.0)
	# 		z = multivariate_normal.pdf(m['prior'], 0, 1.0) 
	# 		# p1 = self.conditional_probability(y,(x*z)) # p(y|x,z)
	# 		p2 = self.conditional_probability(x,z) # p(x|z)
	# 		p3 = self.conditional_probability(y,z) # p(y|z)
	# 		m['state'] = (p1*p2)/p3
	# 		print(m['state'])

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
