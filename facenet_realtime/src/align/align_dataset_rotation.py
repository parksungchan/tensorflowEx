from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from scipy import misc

import facenet_realtime.src.align.detect_face as detect_face
from facenet_realtime import init_value
from facenet_realtime.src.common import facenet
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
# import matplotlib.pyplot as plt
import wget
import dlib, bz2, cv2

class AlignDatasetRotation():
    def rotation_dataset(self, input_path, output_path):
        init_value.init_value.init(self)

        predictor, detector = self.face_rotation_predictor_download()

        dir_list = os.listdir(input_path)
        for dirList in dir_list:
            if dirList.find(self.bounding_boxes) == 0:
                continue
            file_list = os.listdir(input_path+dirList)
            if not os.path.exists(output_path + dirList):
                os.makedirs(output_path + dirList)

            for img in file_list:
                image = cv2.imread(input_path+'/'+dirList+'/'+img)
                try:
                    image = self.face_lotation(image, predictor, detector)
                except:
                    print('Lotation Error:'+input_path+'/'+dirList+'/'+img)
                cv2.imwrite(output_path + dirList + '/' + img, image)

    def face_lotation(self, image, predictor, detector):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_boundaries = detector(gray, 2)

        # loop over the face detections
        fa = FaceAligner(predictor, desiredFaceWidth=512)
        for rect in face_boundaries:
            (x, y, w, h) = rect_to_bb(rect)
            faceAligned = fa.align(image, gray, rect)

        return faceAligned

    def face_rotation_predictor_download(self):
        init_value.init_value.init(self)
        down_pred_url = self.down_land68_url
        bz_pred_file = self.land68_file
        down_pred_path = self.model_path
        dt_pred_file = bz_pred_file.replace('.bz2', '')

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



