import numpy as np
import cv2, os, sys, time, random
# import matplotlib.pyplot as plt
# %matplotlib inline

#sys.path.append("/home/localuser/workspace/s/other/")
# import otherUtils as ut


def showRoiOnImage(imFull,box,show=True):
    y1,x1,y2,x2= box
    imRoi = imFull[y1:y2,x1:x2].copy()
    imF2 = cv2.rectangle(imFull.copy(),(x1,y1),(x2,y2),(0,255,0),10,cv2.LINE_AA)
#     plt.figure(figsize=(15,10))
    if show:
        cv2.imshow("DisplayROI",imF2)
        cv2.waitKey(3)
    return imF2

def adjustImages_hConcant(imL,imR):
    im2Adjust, imOther = None, None
    revertStack = False
    if imL.shape[0] < imR.shape[0]:
        im2Adjust = imL.copy()
        imOther = imR.copy()
    else:
        im2Adjust = imR.copy()
        imOther = imL.copy()
        revertStack = True
    
    imgTmp = np.zeros([imOther.shape[0],im2Adjust.shape[1],3],im2Adjust.dtype)
    imgTmp[:im2Adjust.shape[0],...] = im2Adjust.copy()
    im2Adjust = imgTmp.copy()
    
    if revertStack:
        return np.hstack([imOther,im2Adjust])
    else:
        return np.hstack([im2Adjust,imOther])
    
def drawMatches(img1,img2,kps1,kps2,ptSize=5,show=False):
    if img1.shape[0] != img2.shape[0]:
#         imgTmp = np.zeros([img2.shape[0],img1.shape[1],3],img1.dtype)
#         imgTmp[:img1.shape[0],...] = img1.copy()
#         img1 = imgTmp.copy()
        imD = adjustImages_hConcant(img1,img2)
    else:
        imD = np.hstack([img1,img2])
    offset = img1.shape[1]
    
    for i1 in range(kps1.shape[0]):
        color = np.random.choice(range(256), size=3)
        
        pt1 = (kps1[i1,1],kps1[i1,0])
        pt2 = (kps2[i1,1]+offset,kps2[i1,0])

        imD = cv2.circle(imD,pt1,ptSize,color.tolist(),-1,cv2.LINE_AA)
        imD = cv2.circle(imD,pt2,ptSize,color.tolist(),-1,cv2.LINE_AA)
        
        imD = cv2.line(imD,pt1,pt2,color.tolist(),int(ptSize/2),cv2.LINE_AA)
        
#     if show:
#         plt.figure(figsize=(15,10))
#         plt.imshow(imD)
#         plt.show()
    return imD

def getFeatureCorresConventional(im1_,im2_,show=False):

    detector = cv2.ORB_create(nfeatures=5000,scaleFactor=1.05,nlevels=12,edgeThreshold=1,patchSize=20)
#     detector = cv2.xfeatures2d.SIFT_create(nfeatures=1600,contrastThreshold=0.02,edgeThreshold=20)
    # detector = cv2.xfeatures2d.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(im1_,None)
    kp2, des2 = detector.detectAndCompute(im2_,None)
    
    
    if des2 is None or des1 is None:
        print("One of the descriptor set is empty")
        return [None for _ in range(4)]
#     print(des1.shape,des2.shape)
#     # FLANN parameters
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#     search_params = dict(checks=50)
    
    flann = cv2.BFMatcher()#index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good = []
    pts1 = []
    pts2 = []
    inds = []
    
#     print("Num descriptor matches: ", len(matches))
    if len(matches[0]) < 2:
        print("No matches found")
        return [None for _ in range(4)]
    
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            inds.append(i)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt) 
            
#     inds1, inds2 = getMatchesLSA(des1[inds],des2[inds])
#     pts1 = cv2.KeyPoint_convert(kp1)[inds1]
#     pts2 = cv2.KeyPoint_convert(kp2)[inds2]
    
    if len(pts1) == 0:
        print("No matches found post Ratio Test")
        return [None for _ in range(4)]
    
    pts1 = np.fliplr(np.array(pts1)).astype(int)
    pts2 = np.fliplr(np.array(pts2)).astype(int)
        
#     imD = drawMatches(im1_,im2_,pts1,pts2,show=show) 
#     plt.imshow(imD)
#     plt.show()
    
    pts1Org = np.fliplr(cv2.KeyPoint_convert(kp1)).astype(int)
    pts2Org = np.fliplr(cv2.KeyPoint_convert(kp2)).astype(int)
    
    return pts1, pts2, pts1Org, pts2Org



def getMatch(refImg,qImg,inROI=None):
#     qImgROI = getExpandedRoiImg(inImg,inROI,4.0)                    
#     plt.imshow(qImgROI)
#     plt.show()  
    
    H = None
    p1Fil, p2Fil = None, None
    p1,p2,p1O,p2O = getFeatureCorresConventional(refImg,qImg,False)
    
    if p1 is not None:

        print("Inital Matches",len(p1))

        H,inliers = cv2.findHomography(p1,p2,cv2.RANSAC,10)

        numInliers = inliers.sum()
        print("Inliers",numInliers)

        inlierRatio = float(numInliers)/float(len(p1))
        print("Inlier ratio",inlierRatio)

        if H is not None:

            p1Fil = p1[inliers[:,0].astype(bool)]
            p2Fil = p2[inliers[:,0].astype(bool)]
            # imD = drawMatches(refImg,qImg,p1Fil,p2Fil,show=False) 
            # cv2.putText(imD,"{0:.3f}".format(inlierRatio),(10,10),1,1,(0,0,255))
            # cv2.imshow("imD",imD)
            # cv2.waitKey(0)
            # cv2.imwrite(outFullName.format(gb1),imD)
        else:
            print("H not found")

        mchInfoDet = [inlierRatio,numInliers,len(p1),len(p1O),len(p2O)]
#         mchInfoPerDet.append(mchInfoDet)
#     else:
#         mchInfoPerDet.append([0,0,0,0,0])    
    return H, p1Fil, p2Fil, inlierRatio

def getPtsFromROI(box):
    y1,x1,y2,x2= box
    p1 = np.array([x1,y1])
    p2 = np.array([x2,y2])
    return np.vstack([p1,p2])

def getROIFromPts(pts_):
    pts = pts_.astype(int)
    return pts[0,1], pts[0,0], pts[1,1], pts[1,0]

def warpPoints(pts,H_in):
    ptsWarped = np.matmul(H_in,np.column_stack([pts,np.ones(pts.shape[0])]).transpose())
    ptsWarped /= ptsWarped[-1,:]
    ptsWarped = ptsWarped[:-1,:].transpose()
    return ptsWarped


def matchPair(qImg,rImg,pts2WarpIn=None,displayImgFlag=False):
#     qImgPath = "/home/localuser/pos_-27_636722-152_987333.png"
#     qImg = cv2.imread(qImgPath)
    print(qImg.shape)
    # plt.imshow(qImg)

#     qROI = 380,460,420,510 #y1,x1,y2,x2 # r1,c1,r2,c2
#     qROI = 280,680,310,740
#     showRoiOnImage(qImg,qROI)


#     refImgPath = "/home/localuser/other/data/Deployment-Positioning/GOPR6959.JPG"
#     # refImg = cv2.imread(refImgPath)
#     refImg = ut.undistortImage(refImgPath)
    #qImg = ut.undistortImage(qImg,False,True)
    #qImg = cv2.resize(qImg,(0,0),None,0.125,0.125,cv2.INTER_AREA)

    # qImgPath = "/home/localuser/other/data/Deployment-Positioning/GOPR6960.JPG"
    # qImg = ut.undistortImage(qImgPath)
#     rImg = cv2.resize(rImg,(0,0),None,0.5,0.5,cv2.INTER_AREA)

    print(qImg.shape,rImg.shape)

    # qROI = 0,0,qImg.shape[0],qImg.shape[1]
    # refROI = 0,0,refImg.shape[0], refImg.shape[1]
    print(time.time())
    H, p1F, p2F, inRatio = getMatch(rImg,qImg)
    print(time.time())
    # H = getMatch(refImg,refROI,qImg)
    # getMatchDeep(qImg,refImg)    
    
    
    if pts2WarpIn is not None:
        ptsWarped = warpPoints(pts2WarpIn,H)
        imWithFoundRoi = qImg.copy()
    else:
        fac = 1
        roi2Track = np.array([0,0,rImg.shape[0],rImg.shape[1]])//fac
    #     roi2Track = np.array([380,460,420,510])//fac
        # roi2Track = np.array([280,680,310,740])//fac
        ptsRoi = getPtsFromROI(roi2Track)

        ptsWarped = warpPoints(np.fliplr(ptsRoi),H)
        
        boxWarped = getROIFromPts(np.fliplr(ptsWarped))
#     showRoiOnImage(rImg,roi2Track)
        imWithFoundRoi = showRoiOnImage(qImg,boxWarped,show=displayImgFlag)
    
    return ptsWarped, imWithFoundRoi, inRatio, H
    