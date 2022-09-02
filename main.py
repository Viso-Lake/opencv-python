import cv2


def face_capture():
    cascade_path = 'filters/face_default.xml'
    face_detector = cv2.CascadeClassifier(cascade_path)
    
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        cvt_gray = cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(frame, cvt_gray)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE,
            
        )

        for (x, y, width, height) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (155, 255, 155),
                2,
            )

        cv2.imshow('Faces', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    face_capture()


if __name__ == '__main__':
    main()
