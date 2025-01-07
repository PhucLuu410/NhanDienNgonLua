import cv2
import numpy as np
import serial
import time

ser = serial.Serial('COM3', 115200)

def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_fire = np.array([15, 90, 255])   # Hue, Saturation, Value thấp
    upper_fire = np.array([20, 168, 255]) # Hue, Saturation, Value cao

    mask = cv2.inRange(hsv, lower_fire, upper_fire)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(frame, frame, mask=mask)


    fire_detected = False
    if cv2.countNonZero(mask) > 500:
        fire_detected = True

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return result, mask, fire_detected

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result, mask, fire_detected = detect_fire(frame)
    cv2.imshow("Fire Detection", result)
    cv2.imshow("Original", frame)
    if fire_detected:
        ser.write(b'1')
    else:
        ser.write(b'0')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
ser.close()
