import tensorflow as tf
import numpy as np
import json

class gVal:
    log = "Y"

class JsonObject:

    def __init__(self, d):
        self.__dict__ = d

    def __getitem__(self, item):
        return self.__dict__[item]

    def keys(self):
        return self.__dict__.keys()

    def get_dict(self):
        return self.__dict__

    def dumps(self):
        # only for the simple
        return self.__dict__

# Preprocessing function
def preprocess(detailData, to_ignore, to_outtag, x_type, y_type):
    tmpData = []
    headerData = []
    # print(range(len(detailData)))
    for i in range(len(detailData)):
        for j in range(len(to_outtag)):
            tmpData.append(detailData[i][to_outtag[j]])
        headerData.append(tmpData)
        tmpData = []

    to_ignore = to_ignore+to_outtag

    for id in sorted(to_ignore, reverse=True):
        [r.pop(id) for r in detailData]
    # print("detailData=", detailData)
    # print("headerData=", headerData)
    train_x = np.array(detailData, x_type)
    train_y = np.array(headerData, y_type)

    print("train_x=", train_x)
    print("train_y=", train_y)

    return train_x, train_y


def main(case):
    gValue = gVal()
    errMsg = "S"
    directory = "/home/dev/tensorflowEx/cnnTest/"

    net_id = "test"
    confFile = directory + net_id+"_conf.json"
    with open(confFile) as dataFile:
        conf = json.loads(dataFile.read(), object_hook=JsonObject)

    confFile = directory + net_id + "_input.json"
    with open(confFile) as dataFile:
        detailData = json.loads(dataFile.read(), object_hook=JsonObject)[1:]

    to_ignore = ""
    to_outtag = ""

    to_ignore = [1, 2, 3, 4, 6, 7, 11]
    to_outtag = [0]
    x_type = np.float
    y_type = np.int # Target Type is Int
    train_x, train_y = preprocess(detailData, to_ignore, to_outtag, x_type, y_type)

    test_x = train_x
    test_y = train_y

    # # 데이터셋을 불러옵니다.
    # IRIS_TRAINING = "iris_training.csv"
    # IRIS_TEST = "iris_test.csv"
    # training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
    # test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)
    #
    #
    # train_x = training_set.data
    # train_y = training_set.target
    #
    # test_x = test_set.data
    # test_y = test_set.target

    print(train_x)
    print(train_y)
    if gValue.log == "Y":
        print("====================================================================")
        print(errMsg)
        print("====================================================================")

    # # Define model
    if errMsg == "S" or errMsg[0:1] == "W":

        # 10-20-10의 구조를 갖는 3층 DNN를 만듭니다
        classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

        # 모델을 피팅합니다.
        classifier.fit(x=train_x, y=train_y, steps=200)

        # 정확도를 평가합니다.
        accuracy_score = classifier.evaluate(x=test_x, y=test_y)["accuracy"]
        print('Accuracy: {0:f}'.format(accuracy_score))


        # 새로운 두 꽃의 표본을 분류합니다.
        # new_samples = np.array(
        #     [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
        # y = classifier.predict(new_samples)
        # print('Predictions: {}'.format(str(y)))

if __name__ == '__main__':
    tf.app.run()

