import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt
import math

IMAGE = [100,100]
DISTANCE = 50
M = 1000 # number of particles
# Img Size: 27000000
# Shape: [3000, 3000, 3]

class Filter():
    def __init__(self, name, img):
        self.name = name
        self.img = img
        self.rows = img.shape[0]
        self.cols = img.shape[1]
        self.prior_state = self.set_state()
        self.X_t = self.distribute_particles_randomly()
        # todo: set map point

    def set_state(self):
        col_range = random.randint(0,self.cols-50)
        row_range = random.randint(0,self.rows-50)
        prior_state = [[col_range,col_range+50], [row_range,row_range+50]]
        self.img[prior_state[0][0]:prior_state[0][1], prior_state[1][0]:prior_state[1][1]] = [74, 69, 255]
        return prior_state

    def distribute_particles_randomly(self):
        particles = []
        for p in range(M):
            p = [random.randint(0,self.cols), random.randint(0,self.rows)]
            particles.append(p)
        return particles

    def draw_world(self):
        cv.namedWindow(self.name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.name, self.cols, self.rows)
        cv.imshow(self.name, self.img)
        cv.waitKey(0)
        cv.destroyAllWindows()

def rotateImageToCenter(img, angle):
    img_center = tuple([img.shape[0]/2, img.shape[1]/2])
    rot_mat = cv.getRotationMatrix2D(img_center, angle, 1.0)
    result = cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def main():
    img = cv.imread('MarioMap.png')
    cImg = rotateImageToCenter(img, 0)
    pf = Filter('mario', cImg)
    pf.draw_world()
    
    # constant= cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=BLUE)
    # plt.subplot(231),plt.imshow(constant,'gray'),plt.title('Mario World')

if __name__ == '__main__':
    main()
