import cv2
import numpy as np

# Load video
video_path = 'drone_footage.mp4'  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# ORB feature detector and descriptor
orb = cv2.ORB_create(nfeatures=1000)

# FAST detector
fast = cv2.FastFeatureDetector_create()

# Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, prev_frame = cap.read()
if not ret:
    print("Failed to load video.")
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_keypoints = fast.detect(prev_gray, None)
prev_keypoints, prev_descriptors = orb.compute(prev_gray, prev_keypoints)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect FAST keypoints
    keypoints = fast.detect(gray, None)

    # Compute ORB descriptors
    keypoints, descriptors = orb.compute(gray, keypoints)

    if descriptors is not None and prev_descriptors is not None:
        # Match descriptors
        matches = bf.match(prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 50 matches
        matched_img = cv2.drawMatches(prev_frame, prev_keypoints, frame, keypoints, matches[:50], None, flags=2)

        cv2.imshow("Motion Tracking (FAST + ORB)", matched_img)

    prev_gray = gray
    prev_keypoints = keypoints
    prev_descriptors = descriptors
    prev_frame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
