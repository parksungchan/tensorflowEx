import os

class init_value():
    def init(self):
        self.project_path = os.path.dirname(os.path.abspath(__file__))+'/'
        self.train_data_path = self.project_path+'data/train_data/'
        self.eval_data_path = self.project_path+'data/eval_data/'
        self.detect_data_path = self.project_path+'data/detect_data/'
        if not os.path.exists(self.detect_data_path):
            os.makedirs(self.detect_data_path)

        self.model_path = self.project_path + 'pre_model/'
        self.dets_path = self.model_path + 'dets/'

        self.test_data_files = [self.eval_data_path+'L00002_아무개2/000618.jpg'
                                # ,self.eval_data_path + 'L00004_아무개4/001536.jpg'
                                ]
        self.font_location = self.project_path+'font/ttf/NanumBarunGothic.ttf'

        self.image_size = 160
        self.batch_size = 1000
        self.minsize = 20  # minimum size of face
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.frame_interval = 3
        self.out_image_size = 182

        self.pnet = None
        self.rnet = None
        self.onet = None
        self.images_placeholder = None
        self.embeddings = None
        self.embedding_size = None
        self.phase_train_placeholder = None