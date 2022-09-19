import cv2

open_cv_trained_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam_input = cv2.VideoCapture(0)

while True:
    read_frame_success, frame = webcam_input.read()
    greyscale_conversion = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    data_points_face = open_cv_trained_model.detectMultiScale(greyscale_conversion)
    for (x, y, w, h) in data_points_face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,85,204), 10)
    cv2.imshow('Face Detection Using OpenCV', frame)
    key_input = cv2.waitKey(1)

    if key_input == 81 or key_input == 113: 
        break

webcam_input.release()