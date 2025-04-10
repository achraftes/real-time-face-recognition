import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
while True:
    _,img=webcam.read()
    cv2.imshow("Face detection" ,img)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllwindows()    