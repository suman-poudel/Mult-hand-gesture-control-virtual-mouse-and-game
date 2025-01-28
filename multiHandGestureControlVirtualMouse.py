import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller
import random

mouse = Controller()




# ---------------------------------------- Function -------------------------------------------------------------- #

def moveMouse(img, x, y):
    pyautogui.moveTo(x*screenWidth/img.shape[1], y*screenHeight/img.shape[0])
    return None


def isMouseMove(lengthIndex, lengthPinky):
    return (lengthIndex >= 100 and lengthPinky >=100)
def isLeftClick(lengthIndex, lengthPinky):
    return (lengthIndex <= 20 and lengthPinky >=50)
def ifRightClick(lengthIndex, lengthPinky):
    return (lengthPinky <=20 and lengthIndex <= 20)


def findDistance(pt1, pt2, img = None):
    if pt1 != None and pt2 != None:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        centerX, centerY = int((x2+x1)/2), int((y2+y1)/2)
        # print(x1, y1, x2, y2)

        length = np.hypot(x2-x1, y2-y1)


        if img is not None:
            cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (centerX, centerY), 10, (0, 255, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

    return length, img
    



def findHands(img, hands, mpDraw, mpHands):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            if handLMS:
                # print(handLMS.landmark) # relative x, y, z
                mpDraw.draw_landmarks(img, handLMS, mpHands.HAND_CONNECTIONS,
                )


    return img, results

def findPosition(img, results, draw = False):
    lmList = []
    if results.multi_hand_landmarks:
        for hand_idx, handLMS in enumerate(results.multi_hand_landmarks):
            for id, lm in enumerate(handLMS.landmark):
                # print(id, lm)
                height, weight, channel = img.shape
                cx, cy = int(lm.x*weight), int(lm.y*height)
                lmList.append([cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
    return lmList

def showHandBox(img, results):
    numberHands = []
    if results.multi_hand_landmarks:
        for hand_idx, handLMS in enumerate(results.multi_hand_landmarks):
            lmList = []

            for id, lm in enumerate(handLMS.landmark):
                # print(id, lm)
                height, weight, channel = img.shape
                cx, cy = int(lm.x*weight), int(lm.y*height)
                lmList.append([cx, cy])
            

            # print(hand_idx) # 0 & 1
            label = results.multi_handedness[hand_idx].classification[0].label
            landmarks = handLMS.landmark
            # print(hand_idx, handLMS)
            middle = landmarks[12]
            middleBot = landmarks[9]
            thumb = landmarks[4]
            pinky = landmarks[20]
            wrist = landmarks[0] # Landmark 0 is the wrist
            Index = landmarks[8]


            img_h, img_w, _ = img.shape
            xM = int(middle.x * img_w)  # Convert normalized x to image coordinate
            yM = int(middle.y * img_h) - 20  # Convert normalized y to image coordinate and offset above the wrist
            xT = int(thumb.x * img_w) 
            yT = int(thumb.y * img_h)
            xP = int(pinky.x * img_w)
            yP= int(pinky.y * img_h)
            xW = int(wrist.x * img_w)
            yW = int(wrist.y * img_h)
            xI = int(Index.x * img_w)
            yI = int(Index.y * img_h)
            xMBot = int(middleBot.x * img_w)
            yMBot = int(middleBot.y * img_h)



            centerPt = int((xMBot+xW)/2), int((yMBot+yW)/2)
            indexPt = xI, yI
            cv2.circle(img, centerPt, 5, (0, 255, 0), cv2.FILLED)

            # Display the label near the middle finger
            cv2.putText(img, label + ' Hand', 
                        (xM, yM), 
                        cv2.FONT_HERSHEY_COMPLEX, 
                        0.5, (0, 255, 0), 2)
            


            if label == 'Left':
                cv2.rectangle(img, (xT+20, yM - 20), (xP-20, yW+20), (0, 0, 255), 2)
                numberHands.append({'lmList': lmList, 'bbox': ((xT+20, yM - 20), (xP-20, yW+20)), 'centerPt': centerPt, 'handType': label})
            if label == 'Right':
                cv2.rectangle(img, (xT-20, yM - 20), (xP+20, yW+20), (0, 0, 255), 2)
                numberHands.append({'lmList': lmList, 'bbox': ((xT-20, yM - 20), (xP+20, yW+20)), 'centerPt': centerPt, 'handType': label})
    return numberHands

            
            
def detectGestures(img, lmList1, lmList2):

    lengthIndex, img = findDistance(lmList1[8], lmList2[8], img)
    lengthPinky, _ = findDistance(lmList1[20], lmList2[20])
    xI1, yI1 = int(lmList1[8][0]), int(lmList1[8][1])
    xI2, yI2 = int(lmList2[8][0]), int(lmList2[8][1])
    xT1, yT1 = int(lmList1[4][0]), int(lmList1[4][1])
    xT2, yT2 = int(lmList2[4][0]), int(lmList2[4][1])

    xmidII, ymidII = (xI1+xI2)//2, (yI1+yI2)//2
    xmidTT, ymidTT = (xT1+xT2)//2, (yT1+yT2)//2

    cv2.circle(img, (int(xmidII), int(ymidII)), 5, (0, 255, 255), cv2.FILLED)




    if isMouseMove(lengthIndex, lengthPinky):
        cv2.line(img, (xI1, yI1), (xI2, yI2), (0, 255, 0), 2)
        cv2.circle(img, (int(xmidII), int(ymidII)), 5, (0, 255, 255), cv2.FILLED)
        pyautogui.moveTo(xmidII*screenWidth/img.shape[1], ymidII*screenHeight/img.shape[0])
        cv2.rectangle(img, (440, 30), (580, 60), (255, 0, 255), -1)
        cv2.putText(img, "Mouse Moving", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    elif isLeftClick(lengthIndex, lengthPinky):
        cv2.circle(img, (int(xmidII), int(ymidII)), 5, (0, 0, 255), cv2.FILLED)
        mouse.press(Button.left)
        mouse.release(Button.left)
        cv2.rectangle(img, (440, 30), (580, 60), (255, 0, 255), -1)
        cv2.putText(img, "Left Click", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    elif ifRightClick(lengthIndex, lengthPinky):
        cv2.circle(img, (int(xmidTT), int(ymidTT)), 5, (0, 0, 255), cv2.FILLED)
        mouse.press(Button.right)
        mouse.release(Button.right)
        cv2.rectangle(img, (440, 30), (580, 60), (255, 0, 255), -1)
        cv2.putText(img, "Right Click", (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def showGamePoints(img, numberHands, counter, pointX, pointY):
    hand1 = numberHands[0]
    lmList1 = hand1['lmList']
    hand2 = numberHands[1]
    lmList2 = hand2['lmList']
    xI1, yI1 = int(lmList1[8][0]), int(lmList1[8][1])
    xI2, yI2 = int(lmList2[8][0]), int(lmList2[8][1])
    lengthIndex, img = findDistance(lmList1[8], lmList2[8], img)
    xmidII, ymidII = (xI1+xI2)//2, (yI1+yI2)//2

    cv2.circle(img, (int(xmidII), int(ymidII)), 5, (0, 255, 255), cv2.FILLED)
    

    if lengthIndex<50:
        cv2.line(img, (xI1, yI1), (xI2, yI2), (0, 255, 255), 2)
        if  xmidII - 30 < pointX < xmidII + 30 and ymidII-30 < pointY < ymidII + 30:
            counter = 1
    

    
    
    # cv2.circle(img, (pointX, pointY), 20, color, cv2.FILLED)
    # cv2.circle(img, (pointX, pointY), 10, (0, 0, 0), cv2.FILLED)
    # cv2.circle(img, (pointX, pointY), 20, (255, 255, 255), 2)
    return counter

        
    
    

# ---------------------------------------- Function ------------------------------------------------------------------------ #




# ----------------------------------------------- Virtual Mouse -------------------------------------------------------------- #

def virtualMouse():
    print("Virtual Mouse Activated!")
    cap = cv2.VideoCapture(0)

    # -----------------------------------
    wcam, hcam = 640, 360
    # --------------------------------------
    cap.set(3, wcam)
    cap.set(4, hcam)

    pTime = 0
    cTime = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('Hand_Gesture_Virtual_Mouse.mp4', fourcc, 24, (640,360))

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        img = cv2.flip(img, 1)

        # Add your virtual mouse code here
        cv2.rectangle(img, (5, 20), (150, 50), (255, 0, 255), -1)
        cv2.putText(img, "Mouse Mode On", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        img, results = findHands(img, hands, mpDraw, mpHands)
        # lmList = findPosition(img, results, draw = False)
        numberHands = showHandBox(img, results)
        # print(numberHands)

        if numberHands:
            # hand 1
            hand1 = numberHands[0]
            lmList1 = hand1['lmList']
            bbox1 = hand1['bbox']
            centerPt1 = hand1['centerPt']
            handType1 = hand1['handType']

            if len(numberHands) ==2:
                hand2 = numberHands[1]
                lmList2 = hand2['lmList']
                bbox2 = hand2['bbox']
                centerPt2 = hand2['centerPt']
                handType2 = hand2['handType']

                lengthIndex, img = findDistance(lmList1[8], lmList2[8], img)
                # print(lengthIndex)
                detectGestures(img, lmList1, lmList2)

        out.write(img)
        cv2.imshow("Webcam Feed", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            print("Exiting Mouse Mode.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ----------------------------------------------- Virtual Mouse -------------------------------------------------------------- #

# ------------------------------------------------- Game Mode ----------------------------------------------------------------- #

def gameMode():
    print("Game Mode Activated!")
    cap = cv2.VideoCapture(0)

    # -----------------------------------
    wcam, hcam = 640, 480
    # --------------------------------------
    cap.set(3, wcam)
    cap.set(4, hcam)

    pointX = 250
    pointY = 250

    color = (255, 0, 255)

    counter = 0
    score = 0
    timeStart = time.time()
    totalTime = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('Hand_Gesture_Game.mp4', fourcc, 24, (640, 480))

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break
        img = cv2.flip(img, 1)

        # Add your game code here
        if time.time()-timeStart < totalTime:

            img, results = findHands(img, hands, mpDraw, mpHands)
            #  lmList = findPosition(img, results, draw = False)
            numberHands = showHandBox(img, results)

            if numberHands:
                # hand 1
                hand1 = numberHands[0]
                lmList1 = hand1['lmList']
                bbox1 = hand1['bbox']
                centerPt1 = hand1['centerPt']
                handType1 = hand1['handType']

                if len(numberHands) ==2:
                    hand2 = numberHands[1]
                    lmList2 = hand2['lmList']
                    bbox2 = hand2['bbox']
                    centerPt2 = hand2['centerPt']
                    handType2 = hand2['handType']

                    # lengthIndex, img = findDistance(lmList1[8], lmList2[8], img)

                    counter = showGamePoints(img, numberHands, counter, pointX, pointY)

                    if counter:
                        counter += 1
                        color = (0, 255, 0)
                        if counter == 3:
                            pointX = random.randint(100, int(img.shape[1]  * 0.8))
                            pointY = random.randint(100, int(img.shape[0]  * 0.8))
                            color = (255, 0, 255)
                            score += 1
                            counter = 0   

            cv2.circle(img, (pointX, pointY), 20, color, cv2.FILLED)
            cv2.circle(img, (pointX, pointY), 10, (0, 0, 0), cv2.FILLED)
            cv2.circle(img, (pointX, pointY), 20, (255, 255, 255), 2)

            cv2.rectangle(img, (5, 20), (270, 60), (255, 0, 255), -1)
            cv2.putText(img, "Game Mode On", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Game HUD
            cv2.rectangle(img, (490, 20), (600, 70), (255, 0, 255), -1)
            cv2.putText(img, f'Time: {int(totalTime - (time.time()-timeStart))}', (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(img, f'Score: {str(score).zfill(2)}', (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        else:
            cv2.rectangle(img, (150, 250), (550, 310), (255, 0, 0), -1)
            cv2.putText(img, 'Game Over', (160, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(img, (190, 315), (450, 380), (255, 0, 0), -1)
            cv2.putText(img, f'Your Score: {str(score).zfill(2)}', (270, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, 'Press "R" to restart', (250, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, 'Press "Q" to restart', (250, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            

        out.write(img)
        cv2.imshow("Webcam Feed", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            print("Exiting Game Mode.")
            break
        elif key== ord('r'):
            print("reseting game")
            timeStart = time.time()
            score = 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ------------------------------------------------- Game Mode ----------------------------------------------------------------- #




# ---------------------------------------------------- Starting Code -----------------------------------------------------------#


cap = cv2.VideoCapture(0)
# -----------------------------------
wcam, hcam = 640, 480
# --------------------------------------
cap.set(3, wcam)
cap.set(4, hcam)


mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,       
    max_num_hands=2,            
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pyautogui.FAILSAFE = False
screenWidth, screenHeight = pyautogui.size()
print(f"screenWidth: {screenWidth}, screenHeight: {screenHeight}")





print("Press 'M' for Mouse Mode or 'G' for Game Mode. Press 'Q' to quit.")


while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    img = cv2.flip(img, 1)

    # Show a neutral screen
    cv2.rectangle(img, (5, 35), (500, 55), (255, 0, 0), -1)
    cv2.putText(img, "Press M for Mouse Mode, G for Game Mode, Q to Quit",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Main Menu", img)

    key = cv2.waitKey(1) & 0xFF  # Check for keypress
    if key == ord('m') or key == ord('M'):
        cap.release()  # Release camera for the mode
        cv2.destroyWindow("Main Menu")
        virtualMouse()
        cap = cv2.VideoCapture(0)  # Reinitialize camera after returning
    elif key == ord('g') or key == ord('G'):
        cap.release()
        cv2.destroyWindow("Main Menu")
        gameMode()
        cap = cv2.VideoCapture(0)
    elif key == ord('q') or key == ord('Q'):
        print("Exiting program.")
        break

cap.release()
cv2.destroyAllWindows()


