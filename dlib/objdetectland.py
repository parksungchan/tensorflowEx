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

def objalign(img):
    down_pred_url = 'http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2'
    bz_pred_file = 'shape_predictor_5_face_landmarks.dat.bz2'
    predictor = face_predictor_download(down_pred_url, bz_pred_file)
    face_detector = dlib.get_frontal_face_detector()
    frame = cv2.imread(img)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = face_detector(frame_gray, 1)

    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(img))
        exit()

    faces = dlib.full_object_detections()
    for detection in dets:
        faces.append(predictor(frame_gray, detection))

    # Get the aligned face images
    # Optionally:
    # images = dlib.get_face_chips(img, faces, size=160, padding=0.25)
    images = dlib.get_face_chips(img, faces, size=320)
    for image in images:
        cv_rgb_image = np.array(image).astype(np.uint8)
        cv_bgr_img = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('image', cv_bgr_img)
        cv2.waitKey(0)

    # It is also possible to get a single chip
    image = dlib.get_face_chip(img, faces[0])
    cv_rgb_image = np.array(image).astype(np.uint8)
    cv_bgr_img = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    img = '/home/dev/face/011799.jpg'
    # img = '/home/dev/face/010164.jpg'
    img = '/home/dev/face/face1.jpg'
    img = '/home/dev/face/face2.jpg'
    img = '/home/dev/face/face3.jpg'
    # frame = objdetectland(img)
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.show()

    frame = objalign(img)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()

























