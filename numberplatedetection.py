import cv2
import pytesseract

# Path to Tesseract executable (modify if not in PATH)
# For Windows users:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'car.jpg'  # Replace with your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade for license plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

# Detect plates
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))

for (x, y, w, h) in plates:
    # Draw rectangle around plate
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Crop the plate region
    plate_region = gray[y:y + h, x:x + w]
    
    # Optional: thresholding to improve OCR
    _, thresh = cv2.threshold(plate_region, 127, 255, cv2.THRESH_BINARY)

    # OCR with Tesseract
    plate_text = pytesseract.image_to_string(thresh, config='--psm 8')  # PSM 8 treats image as a single word
    print("Detected Plate Text:", plate_text.strip())

# Show result
cv2.imshow("Detected Plates", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
