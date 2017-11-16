from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC

from facenet_realtime import init_value
from facenet_realtime.src.align.align_dataset_mtcnn import AlignDataset
from facenet_realtime.src.common import facenet


class DataNodeImage():
    def classifier_dataset(self):
        init_value.init_value.init(self)
        with tf.Graph().as_default():
            with tf.Session() as sess:
                dataset = facenet.get_dataset(self.detect_data_path)
                paths, labels = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))

                print('Loading feature extraction model')
                # get Model Path
                model_name = facenet.get_pre_model_path(self.model_path)
                facenet.load_model(model_name)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * self.batch_size
                    end_index = min((i + 1) * self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename = self.model_path+'my_classifier.pkl'
                classifier_filename_exp = os.path.expanduser(classifier_filename)

                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

if __name__ == '__main__':
    # object detect
    AlignDataset().align_dataset()

    # classifier Train
    DataNodeImage().classifier_dataset()





