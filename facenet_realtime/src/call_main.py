from facenet_realtime.src.align.align_dataset_mtcnn import AlignDatasetMtcnn
from facenet_realtime.src.align.align_dataset_rotation import AlignDatasetRotation
from facenet_realtime.src.classifier import ClassifierImage
from facenet_realtime import init_value

class Facenet_run():
    def run(self):
        init_value.init_value.init(self)

        # # object detect
        AlignDatasetMtcnn().align_dataset(self.train_data_path)

        # object rotation
        AlignDatasetRotation().rotation_dataset(self.train_data_path)

        # classifier Train
        ClassifierImage().classifier_dataset()

if __name__ == '__main__':
    Facenet_run().run()



