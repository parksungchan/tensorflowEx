import json
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def println(str):
    print(str)

def get_model1(netconf, dataconf, type):
    x_size = dataconf["preprocess"]["x_size"]
    y_size = dataconf["preprocess"]["y_size"]
    channel = dataconf["preprocess"]["channel"]
    num_classes = netconf["config"]["num_classes"]
    learnrate = netconf["config"]["learnrate"]
    global_step = tf.Variable(initial_value=10, name='global_step', trainable=False)
    ################################################################
    X = tf.placeholder(tf.float32, shape=[None, x_size, y_size, channel], name='x')
    Y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
    ################################################################
    net_check = "S"
    stopper = 1
    model = X
    try:
        while True:
            try:
                layer = netconf["layer" + str(stopper)]
            except Exception as e:
                if stopper == 1:
                    net_check = "Error[100] .............................................."
                    model = "layer is None."
                    return net_check
                break
            println(layer)

            try:
                if str(layer["active"]) == 'relu':
                    activitaion = tf.nn.relu
                else:
                    activitaion = tf.nn.relu

                model = tf.contrib.layers.conv2d(inputs=model
                                              , num_outputs=int(layer["node_out"])
                                              , kernel_size=[int((layer["cnnfilter"][0])), int((layer["cnnfilter"][1]))]
                                              , activation_fn=activitaion
                                              , weights_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                              , padding=str((layer["padding"])))

                model = tf.contrib.layers.max_pool2d(inputs=model
                                                  , kernel_size=[int((layer["maxpoolmatrix"][0])),
                                                                 int((layer["maxpoolmatrix"][1]))]
                                                  , stride=[int((layer["maxpoolstride"][0])),
                                                            int((layer["maxpoolstride"][1]))]
                                                  , padding=str((layer["padding"])))

                if str(layer["droprate"]) is not "":
                    droprate = float((layer["droprate"]))
                else:
                    droprate = 0.0

                if droprate > 0.0 and type == "T":
                    model = tf.nn.dropout(model, droprate)

                println(model)
            except Exception as e:
                net_check = "Error[200] .............................................."
                model = e
                return net_check

            stopper += 1
            if (stopper >= 1000):
                break

        fclayer = netconf["out"]

        # 1. softmax
        reout = int(model.shape[1])*int(model.shape[2])*int(model.shape[3])
        model = tf.reshape(model, [-1, reout])
        println(model)
        W1 = tf.Variable(tf.truncated_normal([reout, fclayer["node_out"]], stddev=0.1))
        model = tf.nn.relu(tf.matmul(model, W1))
        println(model)
        W5 = tf.Variable(tf.truncated_normal([fclayer["node_out"], num_classes], stddev=0.1))
        model = tf.matmul(model, W5)
        println(model)
        #     # # 2. tf.contrib.layers.fully_connected
        #     model = tf.contrib.layers.flatten(model)
        #     model = tf.contrib.layers.fully_connected(model, fclayer["node_out"],
        #                                            normalizer_fn=tf.contrib.layers.batch_norm)
        #     model = tf.contrib.layers.fully_connected(model, num_classes)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(cost, global_step=global_step)
        y_pred_cls = tf.argmax(model, 1)
        check_prediction = tf.equal(y_pred_cls, tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))

    except Exception as e:
        net_check = "Error[300] .............................................."
        println(net_check)
        println(e)
        model = e

    return net_check, model, X, Y, optimizer, y_pred_cls, accuracy, global_step, cost

def get_model(netconf, dataconf, type):
    x_size = dataconf["preprocess"]["x_size"]
    y_size = dataconf["preprocess"]["y_size"]
    channel = dataconf["preprocess"]["channel"]
    num_classes = netconf["config"]["num_classes"]
    learnrate = netconf["config"]["learnrate"]
    global_step = tf.Variable(initial_value=10, name='global_step', trainable=False)
    ################################################################
    X = tf.placeholder(tf.float32, shape=[None, x_size, y_size, channel], name='x')
    Y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
    ################################################################
    stopper = 1
    prenumoutputs = 1
    numoutputs = 1
    model = X
    net_check = 'S'

    while True:
        try:
            layer = netconf["layer" + str(stopper)]
        except Exception as e:
            if stopper == 1:
                net_check = "Error[100] layer is None ..............................."
                return net_check
            break
        stopper += 1

        try:
            layercnt = layer["layercnt"]
            for i in range(layercnt):
                # println(layer)
                if numoutputs == 1:
                    numoutputs = int(layer["numoutputs"])
                    prenumoutputs = numoutputs
                else:
                    numoutputs = prenumoutputs*2
                    prenumoutputs = numoutputs
                active          = str(layer["active"])
                convkernelsize  = [int((layer["cnnfilter"][0])), int((layer["cnnfilter"][1]))]
                maxpkernelsize  = [int((layer["maxpoolmatrix"][0])), int((layer["maxpoolmatrix"][1]))]
                stride          = [int((layer["maxpoolstride"][0])), int((layer["maxpoolstride"][1]))]
                padding         = str((layer["padding"]))

                if active == 'relu':
                    activitaion = tf.nn.relu
                else:
                    activitaion = tf.nn.relu

                if str(layer["droprate"]) is not "":
                    droprate = float((layer["droprate"]))
                else:
                    droprate = 0.0

                model = tf.contrib.layers.conv2d(inputs=model
                                                 , num_outputs=numoutputs
                                                 , kernel_size=convkernelsize
                                                 , activation_fn=activitaion
                                                 , weights_initializer=tf.contrib.layers.xavier_initializer_conv2d()
                                                 , padding=padding)

                model = tf.contrib.layers.max_pool2d(inputs=model
                                                     , kernel_size=maxpkernelsize
                                                     , stride=stride
                                                     , padding=padding)

                if droprate > 0.0 and type == "T":
                    model = tf.nn.dropout(model, droprate)

                println(model)
        except Exception as e:
            net_check = "Error[200] Model Create Fail."
            println(net_check)
            println(e)

    fclayer = netconf["out"]
    reout = int(model.shape[1]) * int(model.shape[2]) * int(model.shape[3])
    model = tf.reshape(model, [-1, reout])
    println(model)
    W1 = tf.Variable(tf.truncated_normal([reout, fclayer["node_out"]], stddev=0.1))
    model = tf.nn.relu(tf.matmul(model, W1))
    println(model)
    W5 = tf.Variable(tf.truncated_normal([fclayer["node_out"], num_classes], stddev=0.1))
    model = tf.matmul(model, W5)
    println(model)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(cost, global_step=global_step)
    y_pred_cls = tf.argmax(model, 1)
    check_prediction = tf.equal(y_pred_cls, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))

    return net_check, model, X, Y, optimizer, y_pred_cls, accuracy, global_step, cost
########################################################################################################################
directory = "/home/dev/tensorflowEx/cnnMnist/"
filename = "netconf.json"
confFile = directory + filename
with open(confFile) as dataFile:
    netconf = json.loads(dataFile.read())
filename = "dataconf.json"
confFile = directory + filename
with open(confFile) as dataFile:
    dataconf = json.loads(dataFile.read())

# print(netconf)
# print(dataconf)
#
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

net_check, model, X, Y, optimizer, y_pred_cls, accuracy, global_step, cost = get_model(netconf, dataconf, "T")
# println(model)

batch_size = 1000
test_size = 256
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
train_op = optimizer
predict_op = y_pred_cls



#######################################################################################################################
# 신경망 모델 학습
#####
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(2):
    total_cost = 0

    for i in range(10):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print( 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print( '최적화 완료!')


#########
# 결과 확인
######
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print ('정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                  Y: mnist.test.labels}))
