import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time


# ============================================================
# AI VIRTUAL MOUSE
# Webcam + OpenCV + MediaPipe + PyAutoGUI
# ============================================================


# -----------------------------
# Screen information
# -----------------------------
screen_width, screen_height = pyautogui.size()

print("Screen Size:", screen_width, "x", screen_height)


# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()


# Set webcam resolution
cam_width = 1280
cam_height = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)


# -----------------------------
# MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# -----------------------------
# Variables
# -----------------------------
previous_x = 0
previous_y = 0

smoothening = 5

click_cooldown = 0.5
last_click_time = 0

right_click_cooldown = 0.7
last_right_click_time = 0

scroll_cooldown = 0.2
last_scroll_time = 0


# ============================================================
# Function: Calculate distance
# ============================================================

def distance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2

    return math.hypot(x2 - x1, y2 - y1)


# ============================================================
# Function: Finger status
# ============================================================

def fingers_up(landmarks):

    fingers = []

    # Thumb
    if landmarks[4][0] > landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index
    if landmarks[8][1] < landmarks[6][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle
    if landmarks[12][1] < landmarks[10][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Ring
    if landmarks[16][1] < landmarks[14][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Pinky
    if landmarks[20][1] < landmarks[18][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers


# ============================================================
# Main Loop
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Cannot read webcam.")
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe processing
    result = hands.process(rgb_frame)


    # --------------------------------------------------------
    # If hand detected
    # --------------------------------------------------------

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]

        # Store landmarks
        landmarks = []

        for landmark in hand_landmarks.landmark:

            x = int(landmark.x * cam_width)
            y = int(landmark.y * cam_height)

            landmarks.append((x, y))


        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # ----------------------------------------------------
        # Get important fingers
        # ----------------------------------------------------

        thumb = landmarks[4]
        index = landmarks[8]
        middle = landmarks[12]
        ring = landmarks[16]
        pinky = landmarks[20]


        # ----------------------------------------------------
        # Finger detection
        # ----------------------------------------------------

        finger_state = fingers_up(landmarks)

        thumb_up = finger_state[0]
        index_up = finger_state[1]
        middle_up = finger_state[2]
        ring_up = finger_state[3]
        pinky_up = finger_state[4]


        # ----------------------------------------------------
        # INDEX FINGER = MOVE MOUSE
        # ----------------------------------------------------

        if index_up and not middle_up and not ring_up and not pinky_up:

            # Convert camera coordinates to screen coordinates
            screen_x = np.interp(
                index[0],
                [100, cam_width - 100],
                [0, screen_width]
            )

            screen_y = np.interp(
                index[1],
                [100, cam_height - 100],
                [0, screen_height]
            )


            # Smooth mouse movement
            current_x = previous_x + (
                screen_x - previous_x
            ) / smoothening

            current_y = previous_y + (
                screen_y - previous_y
            ) / smoothening


            # Move mouse
            pyautogui.moveTo(
                int(current_x),
                int(current_y)
            )


            previous_x = current_x
            previous_y = current_y


            # Show status
            cv2.putText(
                frame,
                "MOVE",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # LEFT CLICK
        # INDEX + MIDDLE
        # ----------------------------------------------------

        if index_up and middle_up and not ring_up and not pinky_up:

            distance_index_middle = distance(
                index,
                middle
            )

            if distance_index_middle < 50:

                current_time = time.time()

                if current_time - last_click_time > click_cooldown:

                    pyautogui.click()

                    last_click_time = current_time

                    cv2.putText(
                        frame,
                        "LEFT CLICK",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2
                    )


        # ----------------------------------------------------
        # RIGHT CLICK
        # THUMB + INDEX PINCH
        # ----------------------------------------------------

        thumb_index_distance = distance(
            thumb,
            index
        )

        if thumb_index_distance < 40:

            current_time = time.time()

            if current_time - last_right_click_time > right_click_cooldown:

                pyautogui.rightClick()

                last_right_click_time = current_time

                cv2.putText(
                    frame,
                    "RIGHT CLICK",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )


        # ----------------------------------------------------
        # SCROLL UP
        # Four fingers up
        # ----------------------------------------------------

        if (
            index_up
            and middle_up
            and ring_up
            and pinky_up
        ):

            current_time = time.time()

            if current_time - last_scroll_time > scroll_cooldown:

                pyautogui.scroll(2)

                last_scroll_time = current_time

                cv2.putText(
                    frame,
                    "SCROLL UP",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )


        # ----------------------------------------------------
        # SCROLL DOWN
        # Fist
        # ----------------------------------------------------

        if (
            not index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):

            current_time = time.time()

            if current_time - last_scroll_time > scroll_cooldown:

                pyautogui.scroll(-2)

                last_scroll_time = current_time

                cv2.putText(
                    frame,
                    "SCROLL DOWN",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    2
                )


        # ----------------------------------------------------
        # Draw finger points
        # ----------------------------------------------------

        cv2.circle(
            frame,
            index,
            10,
            (0, 255, 0),
            cv2.FILLED
        )

        cv2.circle(
            frame,
            thumb,
            10,
            (255, 0, 0),
            cv2.FILLED
        )


    else:

        cv2.putText(
            frame,
            "HAND NOT DETECTED",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # ========================================================
    # Instructions
    # ========================================================

    cv2.rectangle(
        frame,
        (10, cam_height - 150),
        (500, cam_height - 10),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "INDEX  = Move Mouse",
        (25, cam_height - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "INDEX + MIDDLE = Left Click",
        (25, cam_height - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "THUMB + INDEX = Right Click",
        (25, cam_height - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "FIST = Scroll Down | 4 Fingers = Scroll Up",
        (25, cam_height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # Display
    # ========================================================

    cv2.imshow(
        "AI Virtual Mouse",
        frame
    )


    # ESC to exit
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()
hands.close()

print("AI Virtual Mouse stopped.")