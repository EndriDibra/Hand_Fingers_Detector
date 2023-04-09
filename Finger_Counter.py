# importing the required libraries
import cv2
import time
import os
import HandTrackingModule as ht


# camera size
weight_cam, height_cam = 640, 480
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, weight_cam)
cap.set(4, height_cam)

# folder path with the images about fingers
folder_path = "Finger_Images"
myList = os.listdir(folder_path)
print(myList)

imgList = []

for image_path in myList:

    image = cv2.imread(f'{folder_path}/{image_path}')
    imgList.append(image)

print(len(imgList))

pTime = 0

detector = ht.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:

    # detecting hands
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:

            fingers.append(1)

        else:

            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):

            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:

                fingers.append(1)

            else:

                fingers.append(0)

        totalFingers = fingers.count(1)
        print(totalFingers)

        # shaping the images and then printing them into the screen of computer
        h, w, c = imgList[totalFingers - 1].shape
        img[0:h, 0:w] = imgList[totalFingers]

        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)

    # initializing fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # printing into the screen fps
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)