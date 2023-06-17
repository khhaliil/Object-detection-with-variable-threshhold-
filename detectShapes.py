import cv2
import numpy as np


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def empty(a):  # cest la fonction que passera les threshhold

    pass


def getCountours(img, imgcountour):
    countours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # too reduce noise :
    for cntr in countours:
        area = cv2.contourArea(cntr)
        areaMAX = cv2.getTrackbarPos("areaMAX", "param")
        areaMin = cv2.getTrackbarPos("areaMIN", "param")
        if areaMAX > area > areaMin:  # cest une valeur alea
            cv2.drawContours(imgcountour, cntr, -1, (255, 0, 0), 5)
            peri = cv2.arcLength(cntr, True)
            approx = cv2.approxPolyDP(cntr, 0.02*peri, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgcountour, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(imgcountour, "points: " + str(len(approx)),
                        (x+w+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


cap = cv2.VideoCapture(0)
cv2.namedWindow("param")
cv2.resizeWindow("param", 640, 340)
cv2.createTrackbar("th1", "param", 0, 255, empty)
cv2.createTrackbar("th2", "param", 0, 255, empty)
cv2.createTrackbar("areaMIN", "param", 3230, 30000, empty)
cv2.createTrackbar("areaMAX", "param", 29000, 30000, empty)
cv2.createTrackbar("blur", "param", 37, 45, empty)
kernel = np.ones((5, 5))

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    out = img.copy()
    x = cv2.getTrackbarPos("blur", "param")
    a = x % 2
    print("-----", x)
    if (a == 0):
        imgblur = cv2.GaussianBlur(img, (x+1, x+1), 1)
    else:
        imgblur = cv2.GaussianBlur(img, (x, x), 1)
    GrayImg = cv2.cvtColor(imgblur, cv2.COLOR_BGR2GRAY)
    th1 = cv2.getTrackbarPos("th1", "param")
    th2 = cv2.getTrackbarPos("th2", "param")
    imgcanny = cv2.Canny(GrayImg, th1, th2)
    imgdil = cv2.dilate(imgcanny, kernel, iterations=1)
    getCountours(imgdil, out)
    cv2.imshow("out", stackImages(
        0.8, [[img, imgdil, out], [imgblur, imgcanny, GrayImg]]))
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
