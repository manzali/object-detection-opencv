import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
ret = True

while(ret):

    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()

    elapsed_time = time.time() - start_time
    print("Frame acquired in: " + str(round(elapsed_time * 1000)) + " ms")

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()