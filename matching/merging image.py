import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

def resizer(image, newsize: tuple):
    newimage = cv.resize(image, newsize, interpolation=cv.INTER_LINEAR)

    return newimage

loadedImages = [cv.imread("box.png"), cv.imread("box_in_scene.png")]

for fname in glob.glob('*.png'):
    print(fname)
    loadedImages.append(cv.imread(fname))

# now, we have a list of images, lets do some things with these

img1 = loadedImages[0]
img2 = loadedImages[1]

img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
img2 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)

overlayedImg = cv.imread("box.png")

# try this, does this work?
img1 = resizer(img1, (300,300))
overlayedImg = resizer(overlayedImg, (300,300))

plt.imshow(overlayedImg)
plt.show()
