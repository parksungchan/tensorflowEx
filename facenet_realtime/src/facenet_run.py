from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc

from facenet_realtime.src import facenet
import facenet_realtime.src.align.detect_face as detect_face
import matplotlib.pyplot as plt

class DataNodeImage():
    def realtime_run(self, output_dir_path, modeldir, model_name, image_size, batch_size):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, './align/')

                minsize = 20  # minimum size of face
                threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                factor = 0.709  # scale factor
                frame_interval = 3
                out_image_size = 182

                HumanNames = ['L001','L002','L003','L004','L005']    #train human name

                print('Loading feature extraction model')
                facenet.load_model(model_name)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                classifier_filename = modeldir+'my_classifier.pkl'
                classifier_filename_exp = os.path.expanduser(classifier_filename)
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)
                    print('load classifier file-> %s' % classifier_filename_exp)

                # video_capture = cv2.VideoCapture(0)
                c = 0

                print('Start Recognition!')
                prevTime = 0

                # ret, frame = video_capture.read()
                frame =  misc.imread('/home/dev/faceeval/L004/001728.jpg')

                frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)

                curTime = time.time()    # calc fps
                timeF = frame_interval

                if (c % timeF == 0):
                    find_results = []

                    if frame.ndim == 2:
                        frame = facenet.to_rgb(frame)
                    frame = frame[:, :, 0:3]
                    bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    print('Detected_FaceNum: %d' % nrof_faces)

                    if nrof_faces > 0:
                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(frame.shape)[0:2]

                        cropped = []
                        scaled = []
                        scaled_reshape = []
                        bb = np.zeros((nrof_faces,4), dtype=np.int32)

                        for i in range(nrof_faces):
                            emb_array = np.zeros((1, embedding_size))

                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            # inner exception
                            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                print('face is inner of range!')
                                continue

                            cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                            cropped[0] = facenet.flip(cropped[0], False)
                            scaled.append(misc.imresize(cropped[0], (out_image_size, out_image_size), interp='bilinear'))
                            scaled[0] = cv2.resize(scaled[0], (image_size,image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[0] = facenet.prewhiten(scaled[0])
                            scaled_reshape.append(scaled[0].reshape(-1,image_size, image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                            #plot result idx under box
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20
                            # print('result: ', best_class_indices[0])
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                    else:
                        print('Unable to align')

                sec = curTime - prevTime
                prevTime = curTime
                fps = 1 / (sec)
                str = 'FPS: %2.3f' % fps
                text_fps_x = len(frame[0]) - 150
                text_fps_y = 20
                cv2.putText(frame, str, (text_fps_x, text_fps_y),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)

                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.show()
            #     cv2.imshow('Video', frame)
            #
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            #
            # video_capture.release()
            # cv2.destroyAllWindows()


if __name__ == '__main__':
    output_dir_path = '/hoya_src_root/org/'
    modeldir = '/home/dev/tensormsa/third_party/facedetect/'
    # modeldir = '/..Path to Pre-trained model../20170512-110547/20170512-110547.pb'
    model_name = facenet.get_pre_model_path(modeldir)
    image_size = 160
    batch_size = 1000

    # object detect
    DataNodeImage().realtime_run(output_dir_path, modeldir, model_name, image_size, batch_size)