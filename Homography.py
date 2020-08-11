import numpy as np
import cv2 as cv

img_query = cv.imread('./sample/monster1.png')
img_train = cv.imread('./sample/monster_train.png')
gray_query = cv.cvtColor(img_query, cv.COLOR_BGR2GRAY)
gray_train = cv.cvtColor(img_train, cv.COLOR_BGR2GRAY)
print(img_query.shape)
print(img_train.shape)

# Apply feature detection
surf = cv.xfeatures2d.SURF_create(extended=True)
kp_q, des_q = surf.detectAndCompute(gray_query, None)
kp_t, des_t = surf.detectAndCompute(gray_train, None)

# Match features
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
search_params = dict(checks=5)
matcher = cv.FlannBasedMatcher(index_params, search_params)
matches = matcher.knnMatch(des_q, des_t, k=2)

# Apply D.Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.5*n.distance:
        good.append(m)

if len(good) > 10:
    # get perspective M and match mask
    src_pts = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape((-1, 1, 2))
    dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape((-1, 1, 2))
    M, matchMask = cv.findHomography(srcPoints=src_pts, dstPoints=dst_pts,
                                     method=cv.RANSAC, ransacReprojThreshold=5)
    # get perspective transformed points for drawing object outlines
    h, w, d = img_query.shape
    query_pts = np.float32([[0,0], [w, 0], [w, h], [0, h]]).reshape((-1, 1, 2))
    train_pts = np.int32(cv.perspectiveTransform(query_pts, M))
    img_train = cv.polylines(img_train, [train_pts], True, (0, 255, 0), 3, cv.LINE_AA)
else:
    print(f"Not enough good matches are found {len(good)} / 10")
    matchMask = None

# Draw matches
img_match = cv.drawMatches(
    img_query, kp_q, img_train, kp_t, good,
    None, (0,0,255), (255,0,0), matchMask.ravel().tolist(), cv.DRAW_MATCHES_FLAGS_DEFAULT
)

cv.imshow('img_match', img_match)
cv.waitKey(0)
cv.destroyAllWindows()