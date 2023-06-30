import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('D:/Kanishka/face_detection/haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('D:/Kanishka/face_detection/image.jpg')

# Check if image was read successfully
if img is None:
    print('Error: Image not found or cannot be read.')
    exit()

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Check if any faces were detected
if len(faces) == 0:
    print('No Image found.')
    exit()

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Add a text label to the image indicating the object detected
    cv2.putText(img, "Image Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
