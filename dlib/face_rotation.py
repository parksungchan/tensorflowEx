# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2, os, bz2, wget
import matplotlib.pyplot as plt
import numpy as np

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
    detector = dlib.get_frontal_face_detector()
    return predictor, detector

def img_read(img):
    image = cv2.imread(img)
    return image

def face_lotation(image, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_boundaries = detector(gray, 2)

    # loop over the face detections
    fa = FaceAligner(predictor, desiredFaceWidth=512)
    for rect in face_boundaries:
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(image, gray, rect)

    return faceAligned

def objdetectland(image, predictor, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_boundaries = detector(gray, 0)

    for (enum, face) in enumerate(face_boundaries):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(image, (x, y), (x + w, y + h), (120, 160, 230), 2)

        image = image[face.top():face.bottom(), face.left():face.right()]

    return image

if __name__ == "__main__":
    img = ['/home/dev/face/face1.jpg','/home/dev/face/face2.jpg','/home/dev/face/face3.jpg']
    # img = ['/home/dev/face/face3.jpg']

    down_pred_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    bz_pred_file = 'shape_predictor_68_face_landmarks.dat.bz2'
    predictor, detector = face_predictor_download(down_pred_url, bz_pred_file)

    for i in img:
        ir = img_read(i)
        frame = face_lotation(ir, predictor, detector)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.show()

        frame = objdetectland(frame, predictor, detector)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()