pip install opencv-python imutils
import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

# Initialize HOG descriptor and set SVM detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people(frame):
    """
    Detect people in the given frame using HOG descriptor.
    """
    frame_resized = imutils.resize(frame, width=min(400, frame.shape[1]))
    rects, _ = hog.detectMultiScale(frame_resized, winStride=(8, 8), padding=(16, 16), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    return non_max_suppression(rects, overlapThresh=0.65)

def main():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or replace with video file path
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        detections = detect_people(frame)

        # Draw bounding boxes and count
        for (xA, yA, xB, yB) in detections:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        # Display count
        count = len(detections)
        cv2.putText(frame, f"People Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Crowd Counting", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
