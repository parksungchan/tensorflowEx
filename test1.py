import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import datetime
import operator
def println(str):
    print(str)
print("10 ====================================================================")

data = {}

filenames = ["a", "b"]
for filename in filenames:
    data_sub = {}
    if filename == "a":
        label = "111"
        onesort = "222"
    if filename == "b":
        label = "aaa"
        onesort = "bbb"
    for i in range(1):  # pred_cnt
        key = str(i) + "key"
        val = str(i) + "val"
        data_sub[key] = label
        data_sub[val] = onesort
    data[filename] = data_sub
    println(filename)
    println(data_sub)
    println(data)


    println(filename)
    println(data_sub)
    println(data)

# filname = "1air.jpg"
# data_sub = {"0val": 0.080396443605422974, "0key": "car"}
# data[filname] = data_sub
# println(data_sub)
# println(data)
# filname = "2motor.jpg"
# data_sub = {"0val": 0.078210592269897461, "0key": "bolt"}
# data[filname] = data_sub
# println(data_sub)
# println(data)


# labels = ["car", "airplane", "motor", "glove", "bolt"]
# t_cnt_arr = [16, 12, 5, 13, 6]
# f_cnt_arr = [34, 38, 44, 34, 43]
# # print(labels.index("car"))
# # print(labels.find("car"))
#
#
# # view = []
# # for i in range(len(labels)):
# #     view.append(0)
# # print(view)
# #
# # pred_name = "motor"
# #
# # # print(view.index(pred_name))
#
#
# def spaceprint(str, cnt):
#     leng = len(str)
#     cnt = cnt - leng
#     restr = ""
#     for i in range(cnt):
#         restr += " "
#     return restr
#
# for i in range(len(labels)):
#     # println("Category : " + labels[i] + "       TrueCnt=" + str(t_cnt_arr[i]) + "       FalseCnt=" + str(f_cnt_arr[i]) + "      True Percent=" +
#     #         str(t_cnt_arr[i] / (t_cnt_arr[i] + f_cnt_arr[i]) * 100))
#
#     println(labels[i]+spaceprint(labels[i],15)+"###")


# println("TotalCnt=" + str(totalcnt) + " TrueCnt=" + str(t_cnt) + " FalseCnt=" + str(f_cnt))

print("end ====================================================================")
def plot_image(image, cls_true):
    fig, axes = plt.subplots(1, 1)
    axes.imshow(image)

    plt.show()

def load_train_data(output_directory):
    # println("load_train_data =")
    # println(node_id)
    # output_directory = "/hoya_str_root/nn00004/30/datasrc"
    # print(output_directory)
    fp_list = get_filepaths(output_directory)
    return_arr = []
    for file_path in fp_list:
        h5file = h5py.File(file_path, mode='r')
        return_arr.append(h5file)
        #img_data = h5file['image_features']
        #targets = h5file['targets']
        #labels = config_data['labels']
    return return_arr#img_data, targets, labels

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
########################################################################
# nm_classes = label cnt or max label cnt
def one_hot_encoded(num_classes):
    one = np.zeros((num_classes, num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                one[i][j] = 1
    return one
########################################################################
# print("1 ====================================================================")
# batchsize = 5
# num_classes = 5
# labelsDict = ["dog", "cat", "car", "airplane"]
# input_data = load_train_data()
# ################################################################ Label
# print("Label Dict //////////////////////////////////////////////////")
# labelsDictHot = one_hot_encoded(num_classes)
# print(labelsDict)
# print(labelsDictHot)
# ################################################################ Image Label
# for data in input_data:
#     print("File //////////////////////////////////////////////////////")
#     print(data)
#     labels_data = data['targets']
#     img_data = data['image_features']
#
#     for i in range(0, img_data.len(), batchsize):
#         label_data_batch = labels_data[i:i + batchsize]
#         img_data_batch = img_data[i:i + batchsize]
#
#         y_batch = np.zeros((len(label_data_batch), num_classes))
#         r = 0
#         for j in label_data_batch:
#             j = j.decode('UTF-8')
#             k = labelsDict.index(j)
#             y_batch[r] = labelsDictHot[k]
#             r += 1
#
#         x_batch = np.zeros((len(img_data_batch), len(img_data_batch[0])))
#         r = 0
#         for j in img_data_batch:
#             j = j.tolist()
#             x_batch[r] = j
#             r += 1
#             # print(type(j))
#             # print(j)
#         print("Image Label ////////////////////////////////////////////////")
#         print(label_data_batch)
#         print(y_batch)
#         print("Image /////////////////////////////////////////////////")
#         print(x_batch)
#         # if img_data[i].decode('UTF-8') is not None:
#         #     print(img_data[i].decode('UTF-8'))
# # img_data[i].decode('UTF-8')
# print("2 ====================================================================")
#
# model_path = "/hoya_model_root/nn00004/30/cnnmodel"
# save_name = "model"
#
# existcnt = 10
# filelist = os.listdir(model_path)
# print(filelist)
# flist = []
# for i in filelist:
#     i = i.replace(save_name+"-","")
#     dotidx = i.find(".")
#     if dotidx > -1:
#         i = i[:dotidx]
#         try:
#             flist.append(int(i))
#         except:
#             None
# flist.sort()
#
# j =len(flist)-1
# for i in range(len(flist)):
#     if i>existcnt*3:
#         fname = save_name+"-"+str(flist[j])
#         rfname = model_path+"/"+fname+"*.*"
#
#         for file in filelist:
#             print(fname,file)
#             fidx = file.find(fname)
#             if fidx >-1:
#                 if os.path.isfile(model_path + "/" + file):
#                     os.remove(model_path + "/" + file)
#     j -= 1
#
#
#
# # os.remove("/tmp/foo.txt")
#
# print("3 ====================================================================")
# for key, value in filelist.items():
#     println(key)
#     println(value)
#     println(type(value))
#     fp = open("/hoya_str_root/nn00004/30/datasrc/"+value.name, 'wb')
#
#     for chunk in value.chunks():
#         fp.write(chunk)
#     fp.close()
# print("4 ====================================================================")
# filenames = ["/hoya_src_root/dataTest/car/1car.jpg","/hoya_src_root/dataTest/airplane/1air.jpg"]
# img_size = 100
# x_size = 100
# y_size = 100
# num_channels = 10
# channel = 3
# num_classes = 2
#
# images = []
# image = np.zeros((x_size, y_size, channel))
# # print(image)
# images.append(image)
# print(images)
#
# with tf.Session() as sess:
#     for filename in filenames:
#         value = tf.read_file(filename)
#         decoded_image = tf.image.decode_jpeg(value, channels=channel)
#         # print(decoded_image)
#         resized_image = tf.image.resize_images(decoded_image, [img_size, img_size])
#         resized_image = tf.cast(resized_image, tf.uint8)
#         # print(resized_image)
#
#         image = sess.run(resized_image)
#         # print(image)

# with tf.Session() as sess:
#
#     #
#     # image = image.reshape([-1, x_size, y_size, channel])
#     print(image)
#     print(image.shape)
#     x_batch = np.zeros((x_size, y_size, channel))
#     x_batch = image


# print(x_batch)
#
# images = []
#
#
# with tf.Session() as sess:
#     image = sess.run(resized_image)
#     image = np.reshape(image.data, [img_size, img_size, 3])
#     # image = image.reshape([-1, img_size, img_size, num_channels])
#
#     images.append(image)
#
#     print(images)
#     # image = sess.run(decoded_image)
#     # plot_image(image, None)
#     resized_image = tf.image.resize_images(decoded_image, [img_size, img_size], method=0, align_corners=False)
#     print(resized_image)
#     resized_image = tf.cast(resized_image, tf.uint8)
#     # lambda x, method: tf.image.resize_images(x, [img_size, img_size], method=method)
#     print(resized_image)
#     # plot_image(resized_image, None)
#     image = sess.run(resized_image)
#     image = np.reshape(resized_image.data, [img_size, img_size, 3])
#     image = image.reshape([-1, img_size, img_size, num_channels])
#     # resized_image = tf.cast(resized_image, tf.uint8)
# image = np.array(decoded_image)
# image = image.transpose(2, 0, 1)
# print(image)
# print("5 ====================================================================")
# last_chk_path = "/hoya_model_root/nn00004/30/cnnmodel/model-50"
# with tf.Session() as sess:
#     try:
#         print("111")
#         saver = tf.train.Saver()
#         # print("222")
#         # saver.restore(sess, save_path=last_chk_path)
#         # tf.train.Saver.restore(sess=sess, save_path=last_chk_path)
#         print("333")
#     except Exception as e:
#         print(e)


# print("6 ====================================================================")
# modelname = "model"
#
# model_path = "/hoya_model_root/nn00004/30/cnnmodel"
# save_path = model_path + "/" + modelname
# # println(model_path)
# tf.initialize_all_variables()
# with tf.Session() as sess:
#     try:
#         # last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=model_path)
#         saver = tf.train.Saver()
#         # saver.restore(sess, save_path=last_chk_path)
#         # print("Restored checkpoint from:" + last_chk_path)
#     except Exception as e:
#         print(e)


# print("7 ====================================================================")

# X = tf.Variable(tf.random_normal([784, 200], stddev=0.35))
# Y = tf.Variable(X.initialized_value() + 3.)

# X = tf.placeholder(tf.float32, shape=[None, 100, 100, 3], name='x')
# Y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

# init_op = tf.initialize_all_variables()

# saver = tf.train.Saver()

# sess = tf.Session()
# sess.run(init_op)
# save_path = saver.save(sess, "/tmp/model.ckpt")
# print("8 ====================================================================")

# 모델을
# saver
# 를
# 사용하여
# 복구합니다.
# sess.run(init_op)
# saver.restore(sess, "model2.ckpt")
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.restore(sess, "model2.ckpt")
#     prediction = tf.argmax(y_conv, 1)
#     return prediction.eval(feed_dict={x: [imvalue], keep_prob: 1.0},
# print("9 ====================================================================")
# def model_file_delete(model_path, save_name):
#     existcnt = 7
#     filelist = os.listdir(model_path)
#
#     flist = []
#     i = 0
#     for filename in filelist:
#         filetime = datetime.datetime.fromtimestamp(os.path.getctime(model_path + '/' +filename)).strftime('%Y%m%d%H%M%S')
#         tmp = [filename, filetime]
#         if filename.find(save_name) > -1:
#             flist.append(tmp)
#         i += 1
#         flistsort = sorted(flist, key=operator.itemgetter(1), reverse=True)
#     # print(flistsort)
#
#     for i in range(len(flistsort)):
#         if i > existcnt * 3:
#             os.remove(model_path + "/" + flistsort[i][0])
#
#
#
#
#
#
# model_path = "/hoya_model_root/nn00004/30/cnnmodel"
# modelname = "model"
#
# model_file_delete(model_path, modelname)
# import json
#
# data = {}
# data_sub = {}
#
# key = "1"+"key"
# value = "1"+"value"
# data_sub[key] = 1
# data_sub[value] = 3
#
#
# data["filename"] = data_sub
# print(json.dumps(data, sort_keys=True, indent=4))
#
#
#
#
#
#
# evaluation result :\
#     {'3car.jpg': {'0key': 'dog', '1val': 68.96944427490234, '0val': 96.5360107421875,
#                   '3val': -76.7177734375, '2val': 46.39153289794922, '2key': 'airplane'
#         , '1key': 'car', '3key': 'cat'}
#         , '1air.jpg': {'0key': 'dog', '1val': 68.96944427490234, '0val': 96.5360107421875,
#                        '3val': -76.7177734375, '2val': 46.39153289794922
#         , '2key': 'airplane', '1key': 'car', '3key': 'cat'}, '1car.jpg':
#          {'0key': 'dog', '1val': 68.96944427490234, '0val': 96.5360107421875, '3val': -76.7177734375
#              , '2val': 46.39153289794922, '2key': 'airplane', '1key': 'car', '3key': 'cat'}}




#
# Restored checkpoint from:/hoya_model_root/nn00004/30/cnnmodel/model-8191
# filename=1car.jpg predict=dog
# [array([ 0.        ,  1.56972718]), array([ 1.        ,  1.05635142]), array([ 2.        , -0.63770062]), array([ 3.        , -0.91153467])]
# filename=3car.jpg predict=dog
# [array([ 0.        ,  1.38816488]), array([ 1.        ,  0.91998273]), array([ 2.        , -0.49559471]), array([ 3.        , -0.76980686])]
# filename=1air.jpg predict=dog
# [array([ 0.        ,  1.55365396]), array([ 1.        ,  1.06619859]), array([ 2.        , -0.66061038]), array([ 3.        , -0.94450051])]
# filename=86_pic1.jpg predict=dog
# [array([ 0.        ,  1.55365396]), array([ 1.        ,  1.06619859]), array([ 2.        , -0.66061038]), array([ 3.        , -0.94450051])]
# filename=1014879013-2.jpg predict=dog
# [array([ 0.        ,  1.39309752]), array([ 1.        ,  0.92448026]), array([ 2.        , -0.51843065]), array([ 3.        , -0.82029074])]
# filename=744395362_9a3a25ad84.jpg predict=dog
# [array([ 0.        ,  1.55365396]), array([ 1.        ,  1.06619859]), array([ 2.        , -0.66061038]), array([ 3.        , -0.94450051])]
# run predict end........







































