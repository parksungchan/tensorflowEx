# neccessary imports
import cv2, os
import imutils
import numpy as np
import dlib
import matplotlib.pyplot as plt
import wget, bz2

def face_predictor_download(down_pred_url, bz_pred_file):
    down_pred_path = '/home/dev/tensormsa/third_party/facedetect/'
    dt_pred_file = bz_pred_file.replace('.bz2','')

    bz_pred_path = down_pred_path + bz_pred_file
    dt_pred_path = down_pred_path + dt_pred_file

    if not os.path.exists(down_pred_path):
        os.makedirs(down_pred_path)

    if os.path.isfile(bz_pred_path) == False:
        wget.download(down_pred_url, down_pred_path)
        zipfile = bz2.BZ2File(bz_pred_path)  # open the file
        data = zipfile.read()  # get the decompressed data
        newfilepath = bz_pred_path[:-4]  # assuming the filepath ends with .bz2
        open(newfilepath, 'wb').write(data)  # wr

    predictor = dlib.shape_predictor(dt_pred_path)

    return predictor

def land2coords(landmarks, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    return coords

def objdetectland(img):
    down_pred_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    bz_pred_file = 'shape_predictor_68_face_landmarks.dat.bz2'
    predictor = face_predictor_download(down_pred_url, bz_pred_file)
    face_detector = dlib.get_frontal_face_detector()
    frame = cv2.imread(img)
    frame = imutils.resize(frame, width=400)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_boundaries = face_detector(frame_gray, 0)

    for (enum, face) in enumerate(face_boundaries):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 160, 230), 2)

        landmarks = predictor(frame_gray, face)
        landmarks = land2coords(landmarks)
        for (a, b) in landmarks:
            cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)

        cv2.putText(frame, "Face :{}".format(enum + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

    return frame

if __name__ == "__main__":
    img = ['/home/dev/face/face1.jpg', '/home/dev/face/face2.jpg', '/home/dev/face/face3.jpg']
    img = ['/home/dev/face/face3.jpg']

    for i in img:
        frame = objdetectland(i)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()

























