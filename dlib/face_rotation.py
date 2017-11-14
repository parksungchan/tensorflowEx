# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2, os, bz2, wget
import matplotlib.pyplot as plt

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

def face_lotation(img):
    detector = dlib.get_frontal_face_detector()
    down_pred_url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    bz_pred_file = 'shape_predictor_68_face_landmarks.dat.bz2'
    predictor = face_predictor_download(down_pred_url, bz_pred_file)
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(img)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # show the original input image and detect faces in the grayscale
    # image
    # cv2.imshow("Input", image)
    rects = detector(gray, 2)

    # loop over the face detections
    for rect in rects:
        # extract the ROI of the *original* face, then align the face
        # using facial landmarks
        (x, y, w, h) = rect_to_bb(rect)
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        faceAligned = fa.align(image, gray, rect)

    return faceAligned

if __name__ == "__main__":
    img = ['/home/dev/face/face1.jpg','/home/dev/face/face2.jpg','/home/dev/face/face3.jpg']

    for i in img:
        frame = face_lotation(i)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()