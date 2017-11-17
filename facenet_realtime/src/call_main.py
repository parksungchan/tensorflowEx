from facenet_realtime.src.align.align_dataset_mtcnn import AlignDatasetMtcnn
from facenet_realtime.src.align.align_dataset_rotation import AlignDatasetRotation
from facenet_realtime.src.classifier import ClassifierImage

if __name__ == '__main__':
    # # object detect
    # AlignDatasetMtcnn().align_dataset()

    # object rotation
    AlignDatasetRotation().rotation_dataset()

    # classifier Train
    # ClassifierImage().classifier_dataset()

