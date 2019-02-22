import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import math

from particle_filter import ParticleFilter

def rotateImageToCenter(img, angle):
	img_center = tuple([img.shape[0]/2, img.shape[1]/2])
	rot_mat = cv.getRotationMatrix2D(img_center, angle, 1.0)
	result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
	return result


def main():
	img = cv.imread('MarioMap.png')
	cImg = rotateImageToCenter(img, 0)
	pf = ParticleFilter('mario', cImg)
	pf.draw_world()
	
	# constant= cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=BLUE)
	# plt.subplot(231),plt.imshow(constant,'gray'),plt.title('Mario World')

if __name__ == '__main__':
	main()
