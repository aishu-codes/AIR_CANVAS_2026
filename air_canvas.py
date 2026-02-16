import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# ---------------- State Variables ----------------
canvas = None
prev_point = None
start_point = None

mode = "FREE"
draw_color = (0, 255, 255)
brush_size = 4
eraser = False
dark_mode = False
text_mode = False
typed_text = ""

# ---------------- UI BUTTONS ----------------
mode_buttons = [
    (20,100,140,160,"FREE"), (160,100,300,160,"CIRCLE"),
    (320,100,460,160,"RECT"), (480,100,620,160,"TRIANGLE"),
    (640,100,780,160,"OVAL"), (800,100,940,160,"PENTAGON"),
    (960,100,1100,160,"HEXAGON"), (1120,100,1260,160,"TEXT")
]

color_buttons = [
    (20,180,120,240,(255,0,0),"BLUE"),
    (140,180,240,240,(0,255,0),"GREEN"),
    (260,180,360,240,(0,0,255),"RED"),
    (380,180,480,240,(0,255,255),"YELLOW"),
    (500,180,600,240,(128,0,128),"PURPLE"),
    (620,180,720,240,(0,165,255),"ORANGE"),
    (740,180,840,240,(0,0,0),"BLACK")
]

utility_buttons = [
    (860,180,960,240,"SMALL"),
    (980,180,1080,240,"MEDIUM"),
    (1100,180,1200,240,"LARGE"),
    (20,260,140,320,"ERASER"),
    (160,260,280,320,"SAVE"),
    (300,260,420,320,"DARK")
]

# ---------------- Helper Functions ----------------
def draw_hand_points(frame, hand_landmarks):
    mp_draw.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),  # red dots
        mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2)               # white lines
    )

def neon_line(img, p1, p2, color, size):
    for glow in [size+8, size+4]:
        overlay = img.copy()
        cv2.line(overlay, p1, p2, color, glow)
        cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    cv2.line(img, p1, p2, color, size)

def draw_polygon(img, center, sides, radius, color):
    cx, cy = center
    pts = []
    for i in range(sides):
        ang = 2 * math.pi * i / sides
        pts.append((int(cx + radius * math.cos(ang)),
                    int(cy + radius * math.sin(ang))))
    cv2.polylines(img, [np.array(pts)], True, color, 3)

# ---------------- MAIN LOOP ----------------
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        if canvas is None:
            canvas = np.ones_like(frame) * 255

        # ---------- Draw UI ----------
        for x1,y1,x2,y2,n in mode_buttons:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(60,60,60),-1)
            cv2.putText(frame,n,(x1+5,y2-10),0,0.5,(255,255,255),2)

        for x1,y1,x2,y2,c,n in color_buttons:
            cv2.rectangle(frame,(x1,y1),(x2,y2),c,-1)
            cv2.putText(frame,n,(x1+5,y2-10),0,0.5,(255,255,255),2)

        for x1,y1,x2,y2,n in utility_buttons:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(100,100,100),-1)
            cv2.putText(frame,n,(x1+5,y2-10),0,0.5,(255,255,255),2)

        # ---------- Hand Tracking ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:

                # draw hand skeleton (IMPORTANT)
                draw_hand_points(frame, hand)

                ix = int(hand.landmark[8].x * w)
                iy = int(hand.landmark[8].y * h)

                # gesture logic (RELIABLE)
                index_up = hand.landmark[8].y < hand.landmark[6].y
                middle_down = hand.landmark[12].y > hand.landmark[10].y
                draw = index_up and middle_down

                # ---------- Button detection ----------
                for x1,y1,x2,y2,n in mode_buttons:
                    if x1 < ix < x2 and y1 < iy < y2:
                        mode = n
                        text_mode = (n == "TEXT")
                        eraser = False

                for x1,y1,x2,y2,c,n in color_buttons:
                    if x1 < ix < x2 and y1 < iy < y2:
                        if n == "BLACK":
                            draw_color = (0,0,0)
                            canvas = np.ones_like(frame) * 255
                            dark_mode = False
                        else:
                            draw_color = c
                        eraser = False

                for x1,y1,x2,y2,n in utility_buttons:
                    if x1 < ix < x2 and y1 < iy < y2:
                        if n == "ERASER":
                            eraser = True
                        elif n == "SAVE":
                            cv2.imwrite(f"air_canvas_{int(time.time())}.png", canvas)
                        elif n == "DARK":
                            dark_mode = not dark_mode
                            canvas = np.zeros_like(frame) if dark_mode else np.ones_like(frame)*255
                        elif n == "SMALL":
                            brush_size = 3
                        elif n == "MEDIUM":
                            brush_size = 6
                        elif n == "LARGE":
                            brush_size = 10

                # ---------- Drawing ----------
                if mode == "FREE":
                    if draw:
                        if prev_point:
                            if eraser:
                                cv2.circle(canvas, (ix,iy), 25, (0,0,0), -1)
                            else:
                                neon_line(canvas, prev_point, (ix,iy), draw_color, brush_size)
                        prev_point = (ix,iy)
                    else:
                        prev_point = None

                elif mode == "TEXT" and draw:
                    cv2.putText(canvas, typed_text, (ix,iy),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 3)

                else:
                    if draw:
                        if start_point is None:
                            start_point = (ix,iy)
                    else:
                        if start_point:
                            cx,cy = start_point
                            r = int(math.hypot(ix-cx, iy-cy))
                            if mode == "CIRCLE":
                                cv2.circle(canvas, start_point, r, draw_color, 3)
                            elif mode == "RECT":
                                cv2.rectangle(canvas, start_point, (ix,iy), draw_color, 3)
                            elif mode == "TRIANGLE":
                                pts=[(cx,cy-r),(cx-r,cy+r),(cx+r,cy+r)]
                                cv2.polylines(canvas,[np.array(pts)],True,draw_color,3)
                            elif mode == "OVAL":
                                cv2.ellipse(canvas,start_point,(r,r//2),0,0,360,draw_color,3)
                            elif mode == "PENTAGON":
                                draw_polygon(canvas,start_point,5,r,draw_color)
                            elif mode == "HEXAGON":
                                draw_polygon(canvas,start_point,6,r,draw_color)
                        start_point = None

        # ---------- Final Output ----------
        bg = np.zeros_like(frame) if dark_mode else frame
        output = cv2.addWeighted(bg, 0.7, canvas, 0.3, 0)
        cv2.putText(output, f"Mode: {mode}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
        cv2.imshow("ULTIMATE AIR CANVAS", output)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros_like(frame) if dark_mode else np.ones_like(frame)*255
        elif key != -1 and text_mode:
            if key == 8:
                typed_text = typed_text[:-1]
            elif key == 13:
                typed_text = ""
            else:
                typed_text += chr(key)

cap.release()
cv2.destroyAllWindows()







