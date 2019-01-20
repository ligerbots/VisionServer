#!/usr/bin/env python3

import sys
import cv2

dev = int(sys.argv[1])
cap = cv2.VideoCapture(dev)

#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
#print('fourcc=', cap.get(cv2.CAP_PROP_FOURCC))

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
print('auto exposure=', cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
# cap.set(cv2.CAP_PROP_BACKLIGHT, 1)

print('size', cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('backlight', cap.get(cv2.CAP_PROP_BACKLIGHT))
print('fps=', cap.get(cv2.CAP_PROP_FPS))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print('error capturing image ret=', ret)
        sys.exit(1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
