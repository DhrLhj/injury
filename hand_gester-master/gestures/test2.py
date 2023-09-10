import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
            handness = results.multi_handedness[hand_idx].classification[0].label  # This will give either 'Right' or 'Left'
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f'Hand: {handness}, ID: {id}, x: {cx}, y: {cy}')

    cv2.imshow('Hand Keypoints', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
