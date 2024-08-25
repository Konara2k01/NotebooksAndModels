import cv2
import mediapipe
import pyautogui

capture_hands = mediapipe.solutions.hands.Hands()
drawing_option = mediapipe.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(0)
x1 = y1 = x2 = y2 = 0
mouse_x = mouse_y = 0

# Smoothing factor (between 0 and 1)
alpha = 0.2

while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    if all_hands:
        for hand in all_hands: 
            drawing_option.draw_landmarks(image, hand)
            one_hand_landmarks = hand.landmark
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                if id == 8:  # Index finger tip
                    new_mouse_x = int(screen_width / image_width * x)
                    new_mouse_y = int(2*screen_height / image_height * y)
                    
                    # Apply exponential smoothing
                    mouse_x = alpha * new_mouse_x + (1 - alpha) * mouse_x
                    mouse_y = alpha * new_mouse_y + (1 - alpha) * mouse_y
                    
                    cv2.circle(image, (x, y), 10, (0, 255, 255))
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1 = x
                    y1 = y

                if id == 4:  # Thumb tip
                    x2 = x
                    y2 = y
                    cv2.circle(image, (x, y), 10, (0, 255, 255))
                    
            dist = y2 - y1
            if abs(dist) < 20:
                pyautogui.click()

    cv2.imshow("Hand movement video capture", image)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

camera.release()
cv2.destroyAllWindows()
