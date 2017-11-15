from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from facenet_realtime.src import facenet
import os
import math
import pickle
from sklearn.svm import SVC

class DataNodeImage():
    def classifier_dataset(self, datadir, modeldir, model_name, image_size, batch_size, type):
        with tf.Graph().as_default():
            with tf.Session() as sess:
                dataset = facenet.get_dataset(datadir)
                paths, labels = facenet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))

                print('Loading feature extraction model')
                facenet.load_model(model_name)

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * batch_size
                    end_index = min((i + 1) * batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, image_size)
                    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

                classifier_filename = modeldir+'my_classifier.pkl'
                classifier_filename_exp = os.path.expanduser(classifier_filename)

                if (type == 'TRAIN'):
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
                elif (type == 'CLASSIFY'):
                    # Classify images
                    print('Testing classifier')
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)

                    print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                    for i in range(len(best_class_indices)):
                        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                    accuracy = np.mean(np.equal(best_class_indices, labels))
                    print('Accuracy: %.3f' % accuracy)

if __name__ == '__main__':
    output_dir_path = '/hoya_src_root/org/'
    modeldir = '/home/dev/tensormsa/third_party/facedetect/'
    # modeldir = '/..Path to Pre-trained model../20170512-110547/20170512-110547.pb'
    model_name = facenet.get_pre_model_path(modeldir)
    image_size = 160
    batch_size = 1000
    type = ["TRAIN", "CLASSIFY"]
    # object detect
    DataNodeImage().classifier_dataset(output_dir_path, modeldir, model_name, image_size, batch_size, type[0])

