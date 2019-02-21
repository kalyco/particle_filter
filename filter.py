import cv2 as cv2
import numpy as np
import random
from matplotlib import pyplot as plt

IMAGE = [100,100]
DISTANCE = 50

def centerImage(img):
    pivot = [img.shape[0]/2, img.shape[1]/2]
    print(pivot)
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant')
    return imgP


def main():
    img = cv2.imread('MarioMap.png')
    cImg = centerImage(img)
    cols = int(cImg.shape[0])
    rows = int(cImg.shape[1])

    screen_res = 100, 100
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', cols, rows)

    print(img.size)
    print(rows)
    print(cols)

    col_range = random.randint(0,cols-50)
    row_range = random.randint(0,rows-50)
    img[col_range:col_range+50, row_range:row_range+50] = [75, 0, 30]

    cv2.imshow('dst_rt', cImg)

    # constant= cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_CONSTANT,value=BLUE)
    # plt.subplot(231),plt.imshow(constant,'gray'),plt.title('Mario World')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.show()


if __name__ == '__main__':
    main()