import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import collections

# ==========================================
# 1. SETUP & INITIALIZATION
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

# 🚀 Double Smoothing (History Tracker)
smoothening = 6 
plocX, plocY = 0, 0
clocX, clocY = 0, 0
# History of last 3 points for super-smooth moving
history_x = collections.deque(maxlen=3)
history_y = collections.deque(maxlen=3)

frameR = 100 
prev_y = 0   

# Finger tip IDs: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
tipIds = [4, 8, 12, 16, 20]

print("AI Virtual Mouse Started! Press 'q' in camera window to exit.")

# ==========================================
# 2. MAIN LOOP
# ==========================================
while True:
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    cv2.rectangle(frame, (frameR, frameR), (w - frameR, h - frameR), (255, 0, 255), 2)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmark lists
            lmList = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            # 🚀 Logic to Check Which Fingers are UP
            fingers = []
            if len(lmList) != 0:
                # Thumb (Right hand assumption for simplicity, comparing X coordinate)
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                
                # 4 Fingers (Comparing Y coordinate to the joint below it)
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1) # Up
                    else:
                        fingers.append(0) # Down
            
            # Key Coordinates
            ix, iy = lmList[8][1], lmList[8][2]   # Index
            mx, my = lmList[12][1], lmList[12][2] # Middle
            tx, ty = lmList[4][1], lmList[4][2]   # Thumb

            # ==========================================
            # 🚀 GESTURE LOGIC START
            # ==========================================
            
            # 1. MOVE MODE (ONLY Index is UP, rest are DOWN)
            if fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
                cv2.putText(frame, "MOVE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.circle(frame, (ix, iy), 15, (0, 255, 0), cv2.FILLED)
                
                # Mapping
                x3 = np.interp(ix, (frameR, w - frameR), (0, screen_w))
                y3 = np.interp(iy, (frameR, h - frameR), (0, screen_h))
                
                # Super Smooth logic
                history_x.append(x3)
                history_y.append(y3)
                avg_x = sum(history_x) / len(history_x)
                avg_y = sum(history_y) / len(history_y)

                clocX = plocX + (avg_x - plocX) / smoothening
                clocY = plocY + (avg_y - plocY) / smoothening
                
                try: pyautogui.moveTo(clocX, clocY)
                except: pass
                
                plocX, plocY = clocX, clocY
                prev_y = 0

            # 2. LEFT CLICK (Index UP, Thumb UP, and touching)
            elif fingers[1] == 1 and fingers[2] == 0:
                dist_it = math.hypot(tx - ix, ty - iy)
                if dist_it < 40:
                    cv2.putText(frame, "LEFT CLICK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.circle(frame, (ix, iy), 15, (0, 0, 255), cv2.FILLED)
                    pyautogui.click()
                    pyautogui.sleep(0.3)

            # 3. RIGHT CLICK (Index, Middle, Thumb touching)
            elif fingers[1] == 1 and fingers[2] == 1:
                dist_imt = (math.hypot(tx - ix, ty - iy) + math.hypot(tx - mx, ty - my)) / 2
                if dist_imt < 40:
                    cv2.putText(frame, "RIGHT CLICK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    cv2.circle(frame, (tx, ty), 15, (0, 255, 255), cv2.FILLED)
                    pyautogui.rightClick()
                    pyautogui.sleep(0.3)
                    
                # 4. SCROLL (Index & Middle UP, Thumb NOT touching)
                elif dist_imt >= 40:
                    cv2.putText(frame, "SCROLL", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    if prev_y != 0:
                        if iy < prev_y - 10: pyautogui.scroll(200)
                        elif iy > prev_y + 10: pyautogui.scroll(-200)
                    prev_y = iy

    cv2.imshow("AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()