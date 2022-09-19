import cv2

# obtain trained facial data from OpenCV dataset
open_cv_trained_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# use webcam as input
webcam_input = cv2.VideoCapture(0)

#infinite loop
while True:
    read_frame_success, frame = webcam_input.read()
    greyscale_conversion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to greyscaled
    data_points_face = open_cv_trained_model.detectMultiScale(greyscale_conversion)
    for (x, y, w, h) in data_points_face: #loop through data points to create rectangle for facial recognition
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,85,204), 10)
    cv2.imshow('Face Detection Using OpenCV', frame) #displays face detection
    key_input = cv2.waitKey(1) #store key input

    if key_input == 81 or key_input == 113: #if q is pressed, exit loop
        break

webcam_input.release() #terminate program