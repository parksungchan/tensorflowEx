import tensorflow as tf
import time
import numpy as np
from datetime import timedelta
import os
import operator
import json
import datetime
import matplotlib.pyplot as plt
import h5py
import shutil
from time import gmtime, strftime
import zipfile

DataNode = "str"
NeuralNetNode = ""
########################################################################
def plot_image(image, cls_true):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 1)

    if cls_true != None:
        axes.set_xlabel(cls_true)
    axes.imshow(image)

    plt.show()
########################################################################
# nm_classes = label cnt or max label cnt
def one_hot_encoded(num_classes):
    one = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                one[i][j] = 1
    return one
##################################################################################################################
def get_model_path(p1,p2,p3):
    model_path = "/hoya_model_root/nn00004/30/netconf_node"
    return model_path
##################################################################################################################
def get_source_path(p1,p2,p3):
    model_path = "/hoya_src_root/nn00004/30/datasrc"
    return model_path
def get_store_path(p1,p2,p3):
    model_path = "/hoya_str_root/nn00004/30/datasrc"
    return model_path
########################################################################
def get_filepaths(directory):
    """
    utils return file paths under directory
    :param directory:
    :return:
    """
    import os
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)
    return file_paths

def model_file_delete(model_path, save_name):
    existcnt = 10
    filelist = os.listdir(model_path)

    flist = []
    i = 0
    for filename in filelist:
        filetime = datetime.datetime.fromtimestamp(os.path.getctime(model_path + '/' +filename)).strftime('%Y%m%d%H%M%S')
        tmp = [filename, filetime]
        if filename.find(save_name) > -1:
            flist.append(tmp)
        i += 1
        flistsort = sorted(flist, key=operator.itemgetter(1), reverse=True)
    # print(flistsort)

    for i in range(len(flistsort)):
        if i > existcnt * 3:
            os.remove(model_path + "/" + flistsort[i][0])

def println(str):
    print(str)

class DataNodeImage(): #Modify
    """

    """

    def run(self, conf_data):
        println(conf_data)
        dataconf = WorkFlowNetConfCNN().get_view_obj(str(conf_data["node_list"][0]))
        directory = get_source_path(dataconf["key"]["nn_id"], dataconf["key"]["wf_ver_id"],
                                    dataconf["key"]["node"])
        output_directory = get_store_path(dataconf["key"]["nn_id"], dataconf["key"]["wf_ver_id"],
                                          dataconf["key"]["node"])
        x_size = dataconf["preprocess"]["x_size"]
        y_size = dataconf["preprocess"]["y_size"]
        channel = dataconf["preprocess"]["channel"]
        output_filename = strftime("%Y-%m-%d-%H:%M:%S", gmtime())

        # unzip & remove zip
        ziplist = os.listdir(directory)
        for zipname in ziplist:
            if zipname.find(".zip") > -1:
                print("Zip=" + zipname)
                fantasy_zip = zipfile.ZipFile(directory + '/' + zipname)
                fantasy_zip.extractall(directory)
                fantasy_zip.close()
                os.remove(directory + "/" + zipname)

        forderlist = os.listdir(directory)

        filecnt = 0
        for forder in forderlist:
            filelist = os.listdir(directory + '/' + forder)
            filecnt += len(filelist)
        try:
            if len(forderlist) > 0 and filecnt > 0:
                output_path = os.path.join(output_directory, output_filename)
                h5file = h5py.File(output_path, mode='w')
                dtype = h5py.special_dtype(vlen=np.dtype('uint8'))
                hdf_features = h5file.create_dataset('image_features', (filecnt,), dtype=dtype)
                hdf_shapes = h5file.create_dataset('image_features_shapes', (filecnt, channel), dtype='int32')
                hdf_labels = h5file.create_dataset('targets', (filecnt,), dtype='S10')

                # Attach shape annotations and scales
                hdf_features.dims.create_scale(hdf_shapes, 'shapes')
                hdf_features.dims[0].attach_scale(hdf_shapes)

                hdf_shapes_labels = h5file.create_dataset('image_features_shapes_labels', (channel,), dtype='S7')
                hdf_shapes_labels[...] = ['channel'.encode('utf8'),
                                          'height'.encode('utf8'),
                                          'width'.encode('utf8')]
                hdf_features.dims.create_scale(hdf_shapes_labels, 'shape_labels')
                hdf_features.dims[0].attach_scale(hdf_shapes_labels)

                # Add axis annotations
                hdf_features.dims[0].label = 'batch'
                i = 0

                labels = dataconf['labels']
                # println(labels)
                for forder in forderlist:
                    filelist = os.listdir(directory + '/' + forder)
                    # println(forder)
                    for filename in filelist:

                        value = tf.read_file(directory + '/' + forder + '/' + filename)

                        try:
                            decoded_image = tf.image.decode_image(contents=value, channels=channel, name="img")
                            resized_image = tf.image.resize_image_with_crop_or_pad(decoded_image, x_size, y_size)

                            with tf.Session() as sess:
                                image = sess.run(resized_image)
                                image = image.reshape([-1, x_size, y_size, channel])

                                println(image)
                                image = image.flatten()
                                hdf_features[i] = image
                                hdf_shapes[i] = image.shape
                                hdf_labels[i] = forder.encode('utf8')
                                i += 1
                        except:
                            println("ErrorFile="+directory + " forder=" + forder + "  name=" + filename)

                    # shutil.rmtree(directory + "/" + forder)
                    try:
                        idx = labels.index(forder)
                    except:
                        labels.append(forder)

                dataconf["labels"] = labels
                # println(labels)

                # obj = models.NN_WF_NODE_INFO.objects.get(
                #     wf_state_id=dataconf["key"]["nn_id"] + "_" + dataconf["key"]["wf_ver_id"],
                #     nn_wf_node_name=dataconf["key"]["node"])
                # setattr(obj, 'node_config_data', dataconf)
                # obj.save()

                h5file.flush()
                h5file.close()

            return None

        except Exception as e:
            println(e)
            raise Exception(e)

    def _init_node_parm(self, node_id):
        return None

    def _set_progress_state(self):
        return None

    def load_data(self, node_id, parm='all'):
        dataconf = WorkFlowDataImage().get_step_source(node_id)
        output_directory = get_store_path(dataconf["key"]["nn_id"], dataconf["key"]["wf_ver_id"],
                                          dataconf["key"]["node"])
        # fp_list = utils.get_filepaths(output_directory) # modify
        fp_list = get_filepaths(output_directory)
        return_arr = []
        for file_path in fp_list:
            h5file = h5py.File(file_path, mode='r')
            return_arr.append(h5file)
        return return_arr

########################################################################
def get_model(self, netconf, dataconf, type):
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
                    return net_check, model
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
                return net_check, model

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

    return net_check, model, X, Y, optimizer, y_pred_cls, accuracy, global_step
########################################################################
def train_cnn(input_data, netconf, dataconf, X, Y, optimizer, accuracy, global_step):
    x_size = dataconf["preprocess"]["x_size"]
    y_size = dataconf["preprocess"]["y_size"]
    channel = dataconf["preprocess"]["channel"]
    num_classes = netconf["config"]["num_classes"]
    batchsize = netconf["config"]["batch_size"]
    labels = dataconf["labels"]
    labelsHot = one_hot_encoded(num_classes)
    start_time = time.time()

    try:
        for data in input_data:
            println(data)
            labels_data = data['targets']
            img_data = data['image_features']
            # println(labels)
            for i in range(0, img_data.len(), batchsize):
                label_data_batch = labels_data[i:i + batchsize]
                img_data_batch = img_data[i:i + batchsize]

                y_batch = np.zeros((len(label_data_batch), num_classes))
                r = 0
                for j in label_data_batch:
                    j = j.decode('UTF-8')
                    k = labels.index(j)
                    y_batch[r] = labelsHot[k]
                    r += 1
                # println(labels)
                x_batch = np.zeros((len(img_data_batch), len(img_data_batch[0])))
                r = 0
                for j in img_data_batch:
                    j = j.tolist()
                    x_batch[r] = j
                    r += 1

                x_batch = np.reshape(x_batch, (-1, x_size, y_size, channel))
                println("Image Label ////////////////////////////////////////////////")
                println(label_data_batch)
                println(y_batch)
                # println("Image /////////////////////////////////////////////////")
                # println(x_batch)
                train_run(x_batch, y_batch, netconf, X, Y, optimizer, accuracy, global_step)

    except Exception as e:
        net_check = "Error[400] .............................................."
        println(net_check)
        println(e)
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    println("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
########################################################################
def train_run(x_batch, y_batch, netconf, X, Y, optimizer, accuracy, global_step):
    modelname = netconf["key"]["modelname"]
    train_cnt = netconf["config"]["traincnt"]
    model_path = get_model_path(netconf["key"]["nn_id"], netconf["key"]["wf_ver_id"], netconf["key"]["node"])
    save_path = model_path + "/" + modelname

    with tf.Session() as sess:
        try:
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=model_path)
            saver = tf.train.Saver()
            saver.restore(sess, save_path=last_chk_path)
            println("Restored checkpoint from:" + last_chk_path)
        except:
            println("None to restore checkpoint. Initializing variables instead.")
            sess.run(tf.initialize_all_variables())

        for i in range(train_cnt):
            feed_dict_train = {X: x_batch, Y: y_batch}

            i_global, _ = sess.run([global_step, optimizer], feed_dict=feed_dict_train)

            # Print status to screen every 10 iterations (and last).
            if (i_global % 10 == 0) or (i == train_cnt - 1):
                # Calculate the accuracy on the training-batch.
                batch_acc = sess.run(accuracy, feed_dict=feed_dict_train)
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                println(msg.format(i_global, batch_acc))

            # Save a checkpoint to disk every 100 iterations (and last).
            if (i_global % 100 == 0) or (i == train_cnt - 1):
                println("Save model_path=" + model_path)
                saver.save(sess, save_path=save_path, global_step=global_step)
                model_file_delete(model_path, modelname)

    println("Saved checkpoint.")

class NeuralNetNodeCnn():
    """
    """
    def run(self, conf_data):
        println("run NeuralNetNodeCnn")
        # println(conf_data)
        ################################################################
        dataconf = WorkFlowNetConfCNN().get_view_obj(str(conf_data["node_list"][0]))
        netconf = WorkFlowNetConfCNN().get_view_obj(str(conf_data["node_list"][1]))

        net_check, model, X, Y, optimizer, y_pred_cls, accuracy, global_step = get_model(self, netconf, dataconf, "T")
        println(model)
        if net_check == "S":
            input_data = DataNodeImage().load_data(str(conf_data["node_list"][0]))
            train_cnn(input_data, netconf, dataconf, X, Y, optimizer, accuracy, global_step)
        else:
            println("net_check=" + net_check)

        println("train end......")

        return None

    def _init_node_parm(self, node_id):
        return None

    def _set_progress_state(self):
        return None

    def predict1(self, conf_data, ver, filelist):
        """
        predict service method
        1. type (vector) : return vector
        2. type (sim) : positive list & negative list
        :param node_id:
        :param parm:
        :return:
        """
        println("run NeuralNetNodeCnn Predict")
        println(conf_data)

        # search nn_node_info
        dataconf = WorkFlowNetConfCNN().get_view_obj(str(conf_data["node_list"][0]))
        netconf = WorkFlowNetConfCNN().get_view_obj(str(conf_data["node_list"][1]))
        x_size = dataconf["preprocess"]["x_size"]
        y_size = dataconf["preprocess"]["y_size"]
        channel = dataconf["preprocess"]["channel"]
        pred_cnt = netconf["config"]["predictcnt"]
        model_path = get_model_path(netconf["key"]["nn_id"], netconf["key"]["wf_ver_id"], netconf["key"]["node"])

        net_check, model, X, Y, optimizer, y_pred_cls, accuracy, global_step = get_model(self, netconf, dataconf, "P")

        filelist = sorted(filelist.items(), key=operator.itemgetter(0))
        # println(filelist)
        data = {}
        data_sub = {}
        labels = dataconf["labels"]
        println(labels)
        # labelsDictHot = one_hot_encoded(num_classes)
        with tf.Session() as sess:
            try:
                last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=model_path)
                saverss = tf.train.Saver()
                saverss.restore(sess, save_path=last_chk_path)
                println("Restored checkpoint from:" + last_chk_path)

                for file in filelist:
                    # println(file)
                    value = file[1]
                    filename = file[1].name
                    for chunk in value.chunks():
                        decoded_image = tf.image.decode_jpeg(chunk, channels=channel)
                        resized_image = tf.image.resize_images(decoded_image, [x_size, y_size])
                        resized_image = tf.cast(resized_image, tf.uint8)

                        image = sess.run(resized_image)
                        image = image.reshape([-1, x_size, y_size, channel])

                        # println(image)

                        logits, y_pred_true = sess.run([model, y_pred_cls], feed_dict={X: image})
                        # println(logits)
                        # println(y_pred_true)
                        # cls_name = labels[y_pred_true[0]]
                        # println(cls_name)

                        one = np.zeros((len(labels), 2))

                        for i in range(len(labels)):
                            one[i][0] = i
                            one[i][1] = logits[0][i]

                        onesort = sorted(one, key=operator.itemgetter(1,0), reverse=True)

                        println("filename=" + filename+ " predict=" + labels[int(onesort[0][0])])
                        println(onesort)

                        for i in range(pred_cnt):
                            key = str(i)+"key"
                            val = str(i)+"val"
                            data_sub[key] = labels[int(onesort[i][0])]
                            data_sub[val] = onesort[i][1]
                        data[filename] = data_sub

            except Exception as e:
                println("None to restore checkpoint. Initializing variables instead.")
                println(e)

        println("run predict end........")
        # NeuralNetNodeCnn().eval_cnn(conf_data)
        return data

    # def eval_cnn(self, conf_data):
    def predict(self, conf_data, ver, filelist):
        println("run eval .........")
        # search nn_node_info
        dataconf = WorkFlowNetConfCNN().get_view_obj(str(conf_data["node_list"][0]))
        netconf = WorkFlowNetConfCNN().get_view_obj(str(conf_data["node_list"][1]))
        x_size = dataconf["preprocess"]["x_size"]
        y_size = dataconf["preprocess"]["y_size"]
        channel = dataconf["preprocess"]["channel"]
        num_classes = netconf["config"]["num_classes"]
        batchsize = netconf["config"]["batch_size"]
        modelname = netconf["key"]["modelname"]
        labels = dataconf["labels"]
        labelsHot = one_hot_encoded(num_classes)
        model_path = get_model_path(netconf["key"]["nn_id"], netconf["key"]["wf_ver_id"], netconf["key"]["node"])
        save_path = model_path + "/" + modelname

        net_check, model, X, Y, optimizer, y_pred_cls, accuracy, global_step = get_model(self, netconf, dataconf, "P")
        input_data = DataNodeImage().load_data(str(conf_data["node_list"][0]))
        start_time = time.time()

        println(labels)
        try:
            totalcnt = 0
            t_cnt = 0
            f_cnt = 0
            for data in input_data:
                labels_data = data['targets']
                img_data = data['image_features']

                for i in range(0, img_data.len(), batchsize):
                    label_data_batch = labels_data[i:i + batchsize]
                    img_data_batch = img_data[i:i + batchsize]

                    y_batch = np.zeros((len(label_data_batch), num_classes))
                    r = 0
                    for j in label_data_batch:
                        j = j.decode('UTF-8')
                        k = labels.index(j)
                        y_batch[r] = labelsHot[k]
                        r += 1

                    x_batch = np.zeros((len(img_data_batch), len(img_data_batch[0])))
                    r = 0
                    for j in img_data_batch:
                        j = j.tolist()
                        x_batch[r] = j
                        r += 1

                    x_batch = np.reshape(x_batch, (-1, x_size, y_size, channel))
                    # println("Image Label ////////////////////////////////////////////////")
                    # println(label_data_batch)
                    # println(y_batch)
                    # println("Image /////////////////////////////////////////////////")
                    # println(x_batch)

                    with tf.Session() as sess:
                        try:
                            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=model_path)
                            saver = tf.train.Saver()
                            saver.restore(sess, save_path=last_chk_path)
                            println("Restored checkpoint from:" + last_chk_path)

                            logits, y_pred_true = sess.run([model, y_pred_cls], feed_dict={X: x_batch})
                            # # println(logits)
                            for i in range(len(logits)):
                                true_name = label_data_batch[i].decode('UTF-8')
                                pred_name = labels[y_pred_true[i]]
                                println("True Category=" + true_name +" Predict Category="+pred_name)
                                println(logits[i])
                                totalcnt += 1
                                if true_name == pred_name:
                                    t_cnt += 1
                                else:
                                    f_cnt += 1
                        except Exception as e:
                            println("None to restore checkpoint. Initializing variables instead.")
                            println(net_check)
                            println(e)

            println("TotalCnt=" + str(totalcnt) + " TrueCnt=" + str(t_cnt) + " FalseCnt=" + str(f_cnt))
            percent = t_cnt/totalcnt*100
            println("Percent="+str(percent)+"%")
        except Exception as e:
            net_check = "Error[500] .............................................."
            println(net_check)
            println(e)
        # Ending time.
        end_time = time.time()

        # Difference between start and end-times.
        time_dif = end_time - start_time

        # Print the time-usage.
        println("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        return None
##################################################################################################################
class WorkFlowDataImage():
    def get_step_source(self, node_id):
        # print(confstr)
        data = {}
        data_sub1 = {}
        data_sub2 = {}
        data_sub1["node"] = "datasrc"
        data_sub1["nn_id"] = "nn00004"
        data_sub1["wf_ver_id"] = "30"
        data["key"] = data_sub1

        data_sub2["x_size"] = 32
        data_sub2["y_size"] = 32
        data_sub2["channel"] = 3

        data["preprocess"] = data_sub2

        data["labels"] = ["dog","airplane","cat"]
        return data

class WorkFlowNetConfCNN():
    def get_view_obj(self, confstr):
        # print(confstr)

        if confstr == "nn00004_30_datasrc":
            data = {}
            data_sub1 = {}
            data_sub2 = {}
            data_sub1["node"] = "datasrc"
            data_sub1["nn_id"] = "nn00004"
            data_sub1["wf_ver_id"] = "30"
            data["key"] = data_sub1

            data_sub2["x_size"] = 32
            data_sub2["y_size"] = 32
            data_sub2["channel"] = 3

            data["preprocess"] = data_sub2

            data["labels"] = ["dog","airplane","cat"]

        if confstr == "nn00004_30_netconf_node":
            data = {}
            data_sub1 = {}
            data_sub2 = {}
            data_sub3 = {}
            data_sub4 = {}
            data_sub5 = {}
            data_sub6 = {}

            data_sub1["node"] = "datasrc"
            data_sub1["nn_id"] = "nn00004"
            data_sub1["wf_ver_id"] = "30"
            data_sub1["modelname"] = "model"
            data["key"] = data_sub1

            data_sub2["learnrate"] = 0.001
            data_sub2["traincnt"] = 1
            data_sub2["batch_size"] = 10000
            data_sub2["num_classes"] = 5
            data_sub2["predictcnt"] = 3
            data["config"] = data_sub2

            data_sub3["type"] = "cnn"
            data_sub3["active"] = "relu"
            data_sub3["cnnfilter"] = [3, 3]
            data_sub3["cnnstride"] = [1, 1]
            data_sub3["maxpoolmatrix"] = [2, 2]
            data_sub3["maxpoolstride"] = [2, 2]
            data_sub3["node_in"] = 1
            data_sub3["node_out"] = 32
            data_sub3["regualizer"] = ""
            data_sub3["padding"] = "SAME"
            data_sub3["droprate"] = "0.1"
            data["layer1"] = data_sub3

            data_sub4["type"] = "cnn"
            data_sub4["active"] = "relu"
            data_sub4["cnnfilter"] = [3, 3]
            data_sub4["cnnstride"] = [1, 1]
            data_sub4["maxpoolmatrix"] = [2, 2]
            data_sub4["maxpoolstride"] = [2, 2]
            data_sub4["node_in"] = 32
            data_sub4["node_out"] = 64
            data_sub4["regualizer"] = ""
            data_sub4["padding"] = "SAME"
            data_sub4["droprate"] = "0.1"
            data["layer2"] = data_sub4

            data_sub5["type"] = "cnn"
            data_sub5["active"] = "relu"
            data_sub5["cnnfilter"] = [3, 3]
            data_sub5["cnnstride"] = [1, 1]
            data_sub5["maxpoolmatrix"] = [2, 2]
            data_sub5["maxpoolstride"] = [2, 2]
            data_sub5["node_in"] = 64
            data_sub5["node_out"] = 128
            data_sub5["regualizer"] = ""
            data_sub5["padding"] = "SAME"
            data_sub5["droprate"] = "0.1"
            data["layer3"] = data_sub5

            data_sub6["active"] = "softmax"
            data_sub6["cnnfilter"] = ""
            data_sub6["cnnstride"] = ""
            data_sub6["maxpoolmatrix"] = ""
            data_sub6["maxpoolstride"] = ""
            data_sub6["node_in"] = 128
            data_sub6["node_out"] = 1024
            data_sub6["regualizer"] = ""
            data_sub6["padding"] = "SAME"
            data_sub6["droprate"] = ""
            data["out"] = data_sub5

        # print(data)
        return data

##################################################################################################################


# filelist = {
# 'files000001': open('/hoya_src_root/dataTest/car/1car.jpg','rb')
# ,'files000002': open('/hoya_src_root/dataTest/airplane/1air.jpg','rb')
# ,'files000003': open('/hoya_src_root/dataTest/car/3car.jpg','rb')
# ,'files000004': open('/hoya_src_root/dataTest/airplane/86_pic1.jpg','rb')
# ,'files000005': open('/hoya_src_root/dataTest/airplane/1014879013-2.jpg','rb')
# ,'files000006': open('/hoya_src_root/dataTest/airplane/744395362_9a3a25ad84.jpg','rb')
#         }
ver = 30
conf_data = {}
conf_data["node_list"] = ["nn00004_30_datasrc", "nn00004_30_netconf_node", "nn00004_30_eval_node"]
conf_data["node_id"] = "nn00004"
# conf_data = {node_list: [nn00004_30_datasrc, nn00004_30_netconf_node, nn00004_30_eval_node], node_id: nn00004}


# NeuralNetNodeCnn.predict("", conf_data, ver, filelist)
# NeuralNetNodeCnn.eval("", conf_data)


input_data = DataNodeImage().run(conf_data)

