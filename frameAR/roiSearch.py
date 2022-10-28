
# coding: utf-8

# In[153]:


import sys
import cv2, os
import imageio
import numpy as np
from importlib import reload

import matplotlib.pyplot as plt
# %matplotlib inline

import matchImages as MI
reload(MI)

outPath = "./outData2/"
if not os.path.exists(outPath):
    os.makedirs(outPath)


# In[154]:


# create a dummy source video
def getRandomSourceImg():
    randImg = np.random.random_integers(0,255,500*500*3).reshape([500,500,3]).astype(np.uint8)
    r1,c1 = np.random.random_integers(0,300,2)
    refImgSmall = cv2.resize(refImg,None,None,0.2,0.2)
    rW, rH = refImgSmall.shape[1], refImgSmall.shape[0]
#     print(r1,c1,rW,rH)
    randImg[r1:r1+rH,c1:c1+rW,:] = refImgSmall.copy()
#     plt.imshow(randImg)
    return randImg


# In[155]:


# draw new image

def drawNewImg(oldIm,newIm,ptsWarped,Hin):
    print(ptsWarped.shape)
    y1,x1,y2,x2 = ptsWarped.flatten().astype(int)
    newImRsz = cv2.warpPerspective(newIm, Hin,(x2-x1,y2-y1))
#     newImRsz = cv2.resize(newIm,(int(x2-x1),int(y2-y1)),None)
    oldIm[y1:y2,x1:x2] = newImRsz.copy()
    return oldIm


# In[152]:

def drawOnTop(ptsWarped,wpPointsIn,imWithFoundROI):
    tl = np.min(ptsWarped,axis=0).astype(int)[::-1]
    br = np.max(ptsWarped,axis=0).astype(int)[::-1]
    imWithFoundROI = cv2.rectangle(imWithFoundROI,tl,br,(255,0,0),2)
    for k1 in range(wpPointsIn.shape[0]):
        rTo, cTo = int(ptsWarped[k1][0]), int(ptsWarped[k1][1])
        rFrom, cFrom = wpPointsIn[k1][0], wpPointsIn[k1][1]
        
        # imWithFoundROI[max(min(0,rTo),imWithFoundROI.shape[0]-1),max(min(0,cTo),imWithFoundROI.shape[1]-1)] = (0,0,0)# im[rFrom,cFrom]
        imWithFoundROI[min(max(0,rTo),imWithFoundROI.shape[0]-1),min(max(0,cTo),imWithFoundROI.shape[1]-1)] = (255,255,255)#im[rFrom,cFrom]
    return imWithFoundROI

def drawOnTop2(ptsWarped,wpPointsIn,imWithFoundROI,im):
    tl = np.min(ptsWarped,axis=0).astype(int)[::-1]
    br = np.max(ptsWarped,axis=0).astype(int)[::-1]
    imWithFoundROI = cv2.rectangle(imWithFoundROI,tl,br,(255,0,0),2)
    rTos, cTos = ptsWarped[:,0].astype(int), ptsWarped[:,1].astype(int)
    rFroms, cFroms = wpPointsIn[:,0], wpPointsIn[:,1]
    rTos = np.minimum(np.maximum(0,rTos),imWithFoundROI.shape[0]-1)
    cTos = np.minimum(np.maximum(0,cTos),imWithFoundROI.shape[1]-1)
    imWithFoundROI[rTos,cTos] = im[rFroms,cFroms]
    # imWithFoundROI[rTos,cTos] = (255,255,255)#im[rFroms,cFroms]
    return imWithFoundROI

def processPair(refImg,im,dummySourceImg,ptsWarpPrev=None):
    im = cv2.resize(im,(refImg.shape[1],refImg.shape[0]),None)
#         print(dummySourceImg.shape,refImg.shape,im.shape)
    gridR, gridC = np.meshgrid(np.arange(im.shape[0]),np.arange(im.shape[1]))
    wpPointsIn = np.vstack([gridR.flatten(),gridC.flatten()]).transpose()
    ptsWarped, imWithFoundROI, inRatio, HMat = MI.matchPair(dummySourceImg,refImg,wpPointsIn,displayImgFlag=False)
    if (inRatio < 0.3) and ptsWarpPrev is not None:
        ptsWarped = ptsWarpPrev
        
    print(wpPointsIn.shape,ptsWarped.shape)
    imWithFoundROI = drawOnTop2(ptsWarped,wpPointsIn,imWithFoundROI,im)
    # imWithFoundROI = drawNewImg(imWithFoundROI,im,ptsWarped,HMat)
    # plt.imshow(imWithFoundROI)
    # plt.show()
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.imshow("img",imWithFoundROI[:,:,::-1])
    cv2.waitKey(10)
    # cv2.imwrite(os.path.join(outPath,"{0:07d}.png".format(i)),imWithFoundROI[:,:,::-1])
    
    ptsWarpPrev = ptsWarped
    return ptsWarpPrev


refImgPath = "./refImg-qut.png"#"./refImg.jpg"
refImg = cv2.imread(refImgPath)
# refImg = cv2.resize(refImg,None,None,0.06,0.06,cv2.INTER_AREA)
refImg = cv2.resize(refImg,None,None,0.16,0.16,cv2.INTER_AREA)

# print(refImg.shape)
# plt.imshow(refImg)
# plt.show()

def processMain():
    # reader = imageio.get_reader("./vid2play.mp4")
    # src_reader = imageio.get_reader("./sourceVideo.mp4")
    reader = imageio.get_reader("./vid2play-qut.mp4")
    src_reader = imageio.get_reader("<video0>")
    ptsWarpPrev = None
    for i, im in enumerate(reader):
        print("\n\n",i)
#         print('Mean of frame %i is %1.1f' % (i, im.mean()))
#         print(im.shape)
#         plt.imshow(im)
#         plt.show()
        
        dummySourceImg = src_reader.get_data(i)#getRandomSourceImg() 
        dummySourceImg = cv2.resize(dummySourceImg,None,None,0.4,0.4,cv2.INTER_AREA)
        ptsWarpPrev = processPair(refImg,im,dummySourceImg,ptsWarpPrev)
        
def processMainImPair():
    # im = cv2.imread("vid2play.png")
    im = cv2.imread("show-qut.png")
    # im = cv2.resize(im,None,None,0.3,0.3,cv2.INTER_AREA)

    # dummySourceImg = cv2.imread("sourceVideo.png")
    dummySourceImg = cv2.imread("source-qut.jpg")
    # dummySourceImg = cv2.VideoCapture(0).read()[1]
    # dummySourceImg = cv2.resize(dummySourceImg,None,None,0.3,0.3,cv2.INTER_AREA)

    ptsWarpPrev = processPair(refImg,im,dummySourceImg)

    
processMain()
# processMainImPair()
