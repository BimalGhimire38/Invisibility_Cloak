import cv2
import numpy as np
import time

def initialize_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Error: Could not open webcam")
    time.sleep(2)
    return cap

def capture_background(cap, num_frames=30):
    for _ in range(num_frames):
        ret, bg = cap.read()
        if not ret:
            raise Exception("Error: Could not capture background")
    return np.flip(bg, axis=1)

def create_white_mask(hsv_frame):
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask

def main():
    try:
        cap = initialize_webcam()
        background = capture_background(cap)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = np.flip(frame, axis=1)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = create_white_mask(hsv)
            inv_mask = cv2.bitwise_not(mask)
            
            fg = cv2.bitwise_and(frame, frame, mask=inv_mask)
            bg = cv2.bitwise_and(background, background, mask=mask)
            result = cv2.add(fg, bg)
            
            cv2.imshow("Invisibility Cloak", result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()