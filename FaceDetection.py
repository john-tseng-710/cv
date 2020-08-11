import numpy as np
import cv2 as cv


class FaceDetector(object):

    __haar_face = './data/haarcascades/haarcascade_frontalface_alt.xml'
    __haar_eyes = './data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
    __haar_mouth = './data/haarcascades/Mouth.xml'

    def __init__(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Can't capture camera")
            exit(0)
        self.face_classifier = cv.CascadeClassifier(self.__haar_face)
        self.eyes_classifier = cv.CascadeClassifier(self.__haar_eyes)
        self.mouth_classifier = cv.CascadeClassifier(self.__haar_mouth)

    def Run(self):
        n = 0
        while True:
            n += 1
            k = cv.waitKey(1)
            if k == 27:
                break
            ret, frame = self.cap.read()
            if frame is None:
                print("Can't read frame {n}")
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # enhance contrast
            clahe = cv.createCLAHE(clipLimit=40, tileGridSize=(16, 16))
            frame_gray = clahe.apply(frame_gray)

            # detect and draw face roi
            faces = self.face_classifier.detectMultiScale(frame_gray)
            for face in faces:
                f_x, f_y, f_w, f_h = face
                face_center = (f_x + f_w // 2, f_y + f_h // 2)
                frame = cv.ellipse(frame, face_center, (f_w // 2, f_h // 2), 0, 0, 360, (0, 255, 0), 3)
                face_roi = frame_gray[f_y:f_y+f_h, f_x:f_x+f_w]

                # detect and draw eyes roi in face roi
                eyes = self.eyes_classifier.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(30,30))
                for eye in eyes:
                    e_x, e_y, e_w, e_h = eye
                    eye_center = (f_x + e_x + e_w//2, f_y + e_y + e_h//2)
                    eye_radius = int(round(np.sqrt((e_w)**2 + (e_h)**2)))
                    frame = cv.circle(frame, eye_center, eye_radius, (255, 0, 0), 3)

                # detect and draw mouth roi in face roi
                mouths = self.mouth_classifier.detectMultiScale(face_roi, scaleFactor=1.1, minSize=(50,50))
                for mouth in mouths:
                    m_x, m_y, m_w, m_h = mouth
                    mouth_upperlet = f_x + m_x, f_y + m_y
                    mouth_bottomright = f_x + m_x + m_w, f_y + m_y + m_h
                    frame = cv.rectangle(frame, mouth_upperlet, mouth_bottomright, (0, 255, 255), 3)

            cv.imshow('Face', frame)

        self.cap.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    F = FaceDetector()
    F.Run()