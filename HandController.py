import cv2
import mediapipe as mp
import pyautogui
import math
import time

pyautogui.FAILSAFE = False

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Initialize Video Capture
cap = cv2.VideoCapture(1)

# Movement speed multiplier
speed = 2.0

# Click variables
last_click_time = 0
click_threshold = 0.5  # Double-click threshold in seconds
click_cooldown = 0.3  # Prevent rapid misclicks

# Exponential moving average (EMA) parameters for smoother movement
alpha = 0.3  # Smoothing factor
smoothed_x, smoothed_y = None, None

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for natural interaction and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Extract frame dimensions
    frame_height, frame_width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index and middle finger tip coordinates
            index_finger_tip = hand_landmarks.landmark[8]
            middle_finger_tip = hand_landmarks.landmark[12]

            # Convert normalized coordinates to screen coordinates
            index_x = int(index_finger_tip.x * screen_width * speed)
            index_y = int(index_finger_tip.y * screen_height * speed)
            middle_x = int(middle_finger_tip.x * screen_width * speed)
            middle_y = int(middle_finger_tip.y * screen_height * speed)

            # Smooth mouse movement using EMA
            if smoothed_x is None or smoothed_y is None:
                smoothed_x, smoothed_y = index_x, index_y
            else:
                smoothed_x = alpha * index_x + (1 - alpha) * smoothed_x
                smoothed_y = alpha * index_y + (1 - alpha) * smoothed_y

            # Move the mouse smoothly
            pyautogui.moveTo(int(smoothed_x), int(smoothed_y), duration=0.05)

            # Calculate distance between index and middle finger
            distance = calculate_distance((index_x, index_y), (middle_x, middle_y))

            # Click detection with cooldown
            current_time = time.time()
            if distance < 50 and (current_time - last_click_time) > click_cooldown:
                if (current_time - last_click_time) < click_threshold:
                    pyautogui.doubleClick()
                    print("Double click detected")
                else:
                    pyautogui.click()
                    print("Single click detected")

                last_click_time = current_time

    # Display the webcam feed
    cv2.imshow("Hand Tracking - Index and Middle Finger Click", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
