import cv2

# Load the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier('D:/Kanishka/face_detection/haarcascade_frontalface_default.xml')

# Open a video capture object to capture video from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Loop until the user exits
while True:
    # Read a frame from the video capture object
    _, img = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame using the cascade classifier
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangles around the detected faces in the original frame, and display a "face detected" message
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, 'Face detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Display the modified frame with detected faces
    cv2.imshow('img', img)
    
    # If the user presses the 'q' key, break out of the loop and exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources used by the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
