import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui
import math

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

######################
wCam, hCam = 640, 480
frameR = 100     #Frame Reduction
smoothening = 3  #random value
######################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=2)
wScr, hScr = pyautogui.size()

mouse_is_down = False
past_zoom_dist = None

while True:
    # Step1: Find the landmarks
    success, img = cap.read()
    img, numHands = detector.findHands(img)
    if numHands == 2:
        lmList2, bbox = detector.findPosition(img, 1)
    lmList, bbox = detector.findPosition(img)

    # Step2: Get the tip of the index and middle finger
    if numHands != 0:
        x1, y1 = lmList[12][1:]
        if numHands == 2:
            x2, y2 = lmList2[12][1:]

        # Step3: Check which fingers are up
        fingers = detector.fingersUp(lmList)
        if numHands == 2:
            fingers2 = detector.fingersUp(lmList2)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # Step8: Both Index and middle are up: Clicking Mode
        if fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            #print(lmList, "\n\n")
            # Step9: Find distance between fingers
            length, img, lineInfo = detector.findDistance(lmList, 8, 4, img)
            if numHands == 2:
                length2, img2, lineInfo2 = detector.findDistance(lmList2, 8, 4, img)

            # Step10: Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                if numHands == 2:
                    if length2 < 40:
                        cv2.line(img, (lineInfo[4], lineInfo[5]), (lineInfo2[4], lineInfo2[5]), (255, 0, 255), 3)
                        zoom_dist = math.hypot(lineInfo2[4] - lineInfo[4], lineInfo2[5] - lineInfo[5])
                        
                        if past_zoom_dist:
                            scroll_dist = zoom_dist-past_zoom_dist
                            
                            if scroll_dist > 5:
                                pyautogui.scroll(int(zoom_dist-past_zoom_dist))
                            
                            past_zoom_dist = zoom_dist
                        else:
                            past_zoom_dist = zoom_dist
                if not mouse_is_down:
                    pyautogui.mouseDown()
                    mouse_is_down = True
            else:
                if mouse_is_down:
                    pyautogui.mouseUp()
                    mouse_is_down = False

            # Step5: Convert the coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # Step6: Smooth Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Step7: Move Mouse
            #print(wScr - clocX, clocY)
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
                

    # Step11: Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 8), 3)
    img = cv2.resize(img, (920, 540))
    # Step12: Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)