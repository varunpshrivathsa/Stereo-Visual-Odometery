import cv2


def create_orb():
    return cv2.ORB_create(3000)


def detect(img, orb):
    return orb.detectAndCompute(img, None)


def match_knn(des1, des2, ratio=0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    return good