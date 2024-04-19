import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

def are_fingers_close(hand_landmarks, finger_one, finger_two):
    x1, y1 = hand_landmarks.landmark[finger_one].x, hand_landmarks.landmark[finger_one].y
    x2, y2 = hand_landmarks.landmark[finger_two].x, hand_landmarks.landmark[finger_two].y
    distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    return distance < 0.05 

control_flag = False

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            if are_fingers_close(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP):
                if not control_flag:
                    windows = gw.getWindowsWithTitle('YouTube')
                    if windows:
                        windows[0].activate()
                        pyautogui.press('k')
                        control_flag = True
                    cv2.putText(image, 'Paused', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            else:
                control_flag = False
    cv2.imshow('MediaPipe Hands', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
