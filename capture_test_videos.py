import os
from imutils.video import VideoStream
import imutils
import cv2

basePath = os.path.dirname(os.path.realpath(__file__))

print("[INFO] starting camera...")
vs = VideoStream(src=0).start()
# setup output
i = 0
while os.path.exists(f"out{i}.mp4"):
    i += 1
name = f'out{i}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps = 60
out = None

print("[INFO] starting capture...")
while True:
    frame = vs.read()
    # check if the writer is None
    if out is None:
		# store the image dimensions, initialize the video writer,
		# and construct the zeros array
        (h, w) = frame.shape[:2]
        out = cv2.VideoWriter(name, fourcc, fps,
			(w, h), True)
    out.write(frame)
    # show the frames
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
out.release()
