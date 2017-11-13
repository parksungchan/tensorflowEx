# neccessary imports
import cv2, os
import imutils
import numpy as np
import dlib
import matplotlib.pyplot as plt
import wget, bz2

def land2coords(landmarks, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)

    return coords

def objdetectland(img):
    down_pred_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    down_pred_path = '/home/dev/tensormsa/third_party/objdetectland/'

    bz_pred_file = 'shape_predictor_68_face_landmarks.dat.bz2'
    dt_pred_file = 'shape_predictor_68_face_landmarks.dat'

    bz_pred_path = down_pred_path + bz_pred_file
    dt_pred_path = down_pred_path + dt_pred_file

    if not os.path.exists(down_pred_path):
        os.makedirs(down_pred_path)

    # if os.path.isfile(bz_pred_path) == False:
    #     wget.download(down_pred_url,down_pred_path)
    with open(bz_pred_file, 'rb') as source, open(dt_pred_file, 'wb') as dest:
        dest.write(bz2.decompress(source.read()))

    face_detector = dlib.get_frontal_face_detector()

    landmark_predictor = dlib.shape_predictor(dt_pred_path)

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

        landmarks = landmark_predictor(frame_gray, face)
        landmarks = land2coords(landmarks)
        for (a, b) in landmarks:
            cv2.circle(frame, (a, b), 2, (255, 0, 0), -1)

        cv2.putText(frame, "Face :{}".format(enum + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 128), 2)

    return frame

if __name__ == "__main__":
    img = '/home/dev/face/011799.jpg'
    # img = '/home/dev/face/010164.jpg'
    frame = objdetectland(img)

    plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    plt.show()





















