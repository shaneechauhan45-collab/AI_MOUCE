import cv2
import mediapipe as mp
import pyautogui
import time
import os

# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = "face_landmarker.task"

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

# Mouse sensitivity
MOUSE_SPEED = 18

# Thresholds
GAZE_THRESHOLD = 0.35
BLINK_THRESHOLD = 0.55
BROW_UP_THRESHOLD = 0.45
BROW_DOWN_THRESHOLD = 0.15

# Cooldowns
CLICK_COOLDOWN = 0.8
SCROLL_COOLDOWN = 0.25


# =========================================================
# CHECK MODEL
# =========================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: face_landmarker.task not found.")
    print("Download it and place it in the same folder as main.py")
    exit()


# =========================================================
# MEDIAPIPE FACE LANDMARKER
# =========================================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


base_options = BaseOptions(
    model_asset_path=MODEL_PATH
)

options = FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True
)

landmarker = FaceLandmarker.create_from_options(options)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()


# =========================================================
# SCREEN SIZE
# =========================================================

screen_width, screen_height = pyautogui.size()

print("Screen:", screen_width, "x", screen_height)
print()
print("FACE COMPUTER CONTROLLER STARTED")
print()
print("Eye movement       -> Mouse movement")
print("Both eyes blink    -> Left click")
print("Raise eyebrows     -> Scroll UP")
print("Lower eyebrows     -> Scroll DOWN")
print("Press Q             -> Exit")


# =========================================================
# VARIABLES
# =========================================================

previous_time = 0
last_click_time = 0
last_scroll_time = 0

# To prevent repeated click
blink_active = False


# =========================================================
# HELPER FUNCTION
# =========================================================

def get_blendshape(blendshapes, name):

    for item in blendshapes:
        if item.category_name == name:
            return item.score

    return 0.0


# =========================================================
# MAIN LOOP
# =========================================================

timestamp_ms = 0

while True:

    success, frame = cap.read()

    if not success:
        print("Camera frame not received.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms += 33

    # Detect face
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # =====================================================
    # IF FACE FOUND
    # =====================================================

    if result.face_blendshapes:

        blendshapes = result.face_blendshapes[0]

        # -------------------------------------------------
        # EYE BLINK
        # -------------------------------------------------

        left_blink = get_blendshape(
            blendshapes,
            "eyeBlinkLeft"
        )

        right_blink = get_blendshape(
            blendshapes,
            "eyeBlinkRight"
        )

        both_eyes_closed = (
            left_blink > BLINK_THRESHOLD
            and
            right_blink > BLINK_THRESHOLD
        )


        # -------------------------------------------------
        # DOUBLE/BOTH EYE BLINK = LEFT CLICK
        # -------------------------------------------------

        current_time = time.time()

        if both_eyes_closed:

            if not blink_active:

                if current_time - last_click_time > CLICK_COOLDOWN:

                    pyautogui.click()

                    last_click_time = current_time

                    print("LEFT CLICK")

                blink_active = True

        else:

            blink_active = False


        # -------------------------------------------------
        # EYE GAZE
        # -------------------------------------------------

        look_left = (
            get_blendshape(
                blendshapes,
                "eyeLookOutLeft"
            )
        )

        look_right = (
            get_blendshape(
                blendshapes,
                "eyeLookInLeft"
            )
        )

        look_up = (
            get_blendshape(
                blendshapes,
                "eyeLookUpLeft"
            )
        )

        look_down = (
            get_blendshape(
                blendshapes,
                "eyeLookDownLeft"
            )
        )


        # -------------------------------------------------
        # MOVE MOUSE
        # -------------------------------------------------

        move_x = 0
        move_y = 0

        if look_left > GAZE_THRESHOLD:
            move_x = -MOUSE_SPEED

        elif look_right > GAZE_THRESHOLD:
            move_x = MOUSE_SPEED

        if look_up > GAZE_THRESHOLD:
            move_y = -MOUSE_SPEED

        elif look_down > GAZE_THRESHOLD:
            move_y = MOUSE_SPEED


        if move_x != 0 or move_y != 0:

            pyautogui.moveRel(
                move_x,
                move_y,
                duration=0.01
            )


        # -------------------------------------------------
        # EYEBROWS
        # -------------------------------------------------

        left_brow = get_blendshape(
            blendshapes,
            "browOuterUpLeft"
        )

        right_brow = get_blendshape(
            blendshapes,
            "browOuterUpRight"
        )

        inner_brow = get_blendshape(
            blendshapes,
            "browInnerUp"
        )

        brow_average = (
            left_brow +
            right_brow +
            inner_brow
        ) / 3


        # -------------------------------------------------
        # EYEBROW UP = SCROLL UP
        # -------------------------------------------------

        current_time = time.time()

        if brow_average > BROW_UP_THRESHOLD:

            if current_time - last_scroll_time > SCROLL_COOLDOWN:

                pyautogui.scroll(2)

                last_scroll_time = current_time


        # -------------------------------------------------
        # EYEBROW DOWN = SCROLL DOWN
        # -------------------------------------------------

        elif brow_average < BROW_DOWN_THRESHOLD:

            if current_time - last_scroll_time > SCROLL_COOLDOWN:

                pyautogui.scroll(-2)

                last_scroll_time = current_time


        # =================================================
        # DISPLAY INFORMATION
        # =================================================

        cv2.putText(
            frame,
            "FACE COMPUTER CONTROLLER",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Blink L: {left_blink:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blink R: {right_blink:.2f}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Brow: {brow_average:.2f}",
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Q = EXIT",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # =====================================================
    # SHOW CAMERA
    # =====================================================

    cv2.imshow(
        "Face Computer Controller",
        frame
    )


    # =====================================================
    # EXIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Program closed.")