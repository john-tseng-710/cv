import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

img_query = cv.imread('./sample/oreo.png')
img_train = cv.imread('./sample/oreo_train.png')
gray_query = cv.cvtColor(img_query, cv.COLOR_BGR2GRAY)
gray_train = cv.cvtColor(img_train, cv.COLOR_BGR2GRAY)
print(img_query.shape)
print(img_train.shape)

def BF_ORB():
    orb = cv.ORB_create(WTA_K=2)
    kp_q, des_q = orb.detectAndCompute(gray_query, mask=None)
    kp_t, des_t = orb.detectAndCompute(gray_train, mask=None)

    matcher = cv.BFMatcher(normType=cv.NORM_HAMMING, crossCheck=False)
    matches = matcher.match(des_q, des_t, mask=None)
    print(f'Found {len(matches)} matches')
    matches = sorted(matches, key=lambda x: x.distance)
    img_match = cv.drawMatches(
        img1=img_query, keypoints1=kp_q, img2=img_train, keypoints2=kp_t,
        matches1to2=matches[:10], outImg=None, matchColor=(0,0,255), singlePointColor=(255,0,0),
        matchesMask=None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('img_match', img_match)

    cv.waitKey(0)
    cv.destroyAllWindows()

def BF_SIFT():
    sift = cv.xfeatures2d.SIFT_create()
    kp_q, des_q = sift.detectAndCompute(gray_query, mask=None)
    kp_t, des_t = sift.detectAndCompute(gray_train, mask=None)

    matcher = cv.BFMatcher(normType=cv.NORM_L2, crossCheck=False)
    matches = matcher.knnMatch(des_q, des_t, k=3)

    # Apply ratio test proposed by D.Lowe
    good = []
    for i, j, k in matches:
        if i.distance < 0.25*j.distance:
            good.append([i])

    img_match = cv.drawMatchesKnn(
        img1=img_query, keypoints1=kp_q, img2=img_train, keypoints2=kp_t,
        matches1to2=good, outImg=None, matchColor=(0,0,255), singlePointColor=(255,0,0),
        matchesMask=None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('img_match', img_match)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    BF_ORB()
    BF_SIFT()

