# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:35:19 2017

@author: LiDongNing
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.interpolate import interp1d

import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]

  '''
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  

    '''
    
  pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 32])
  dense = tf.layers.dense(inputs=pool1_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)




# read data
def read_file_list(inputFile, encoding):
    file_list = []
    fin = open(inputFile, 'r', encoding=encoding)
    for eachLiine in fin.readlines():
        file_list.append(eachLiine.strip('\n'))
    fin.close()
    return file_list

# change absulute coordinates into  relative coordinates
def getPosition(DataX,DataY,size):
    X = np.array(DataX).astype(np.float)
    Y = np.array(DataY).astype(np.float)
    array = np.zeros([size, size])

    min_x = np.min(X)
    min_y = np.min(Y)

    value_x = np.max(X)- min_x
    value_y = np.max(Y)- min_y

    dx = value_x/size
    dy = value_y/size

    for i in range(0, len(X)):
        row = -1
        col = -2
        if dx == 0 and dy == 0:
            dx_new = size/len(X)
            dy_new = size/len(Y)
            row = int(i*dx_new)
            col = int(i*dy_new)
        if dx == 0 and dy != 0:
            dx_new = size/len(X)
            row = int(i*dx_new)
            col = int((Y[i] - min_y)/dy)
            #print(row,col)
        if dx != 0 and dy == 0:
            dy_new = size/len(Y)
            col = int(i*dy_new)
            row = int((X[i]-min_x)/dx)
        if dx != 0 and dy != 0:
            #print(i,Y[i],min_y)
            row = int((X[i]-min_x)/dx)
            col = int((Y[i] - min_y)/dy)

        if row == size and col == size:
            
            array[0][size-1] = 1
        if row == size and col != size:
            
            array[0][col] = 1
        if row != size and col == size:
            
            array[size-1-row][size-1] = 1
        if row != size and col != size:
            #print(row,col,dx,dy)
            
            array[size-1-row][col] = 1
           
    return array

def getPostion_with_interp1d(DataX,DataY,size):
    X = np.array(DataX).astype(np.float)
    Y = np.array(DataY).astype(np.float)
    #样条曲线差值，得到样条函数
    
    f = interp1d(X, Y)

    min_x = np.min(X)
    max_x = np.max(X)
    dx = (max_x - min_x)/size
    aix = []
    for ele in range(0, size):
        aix.append(min_x + ele * dx)

    aiy = f(aix)

    dy = np.ptp(aiy)/size

    array = np.zeros([size, size])
    row_array = []
    for col in range(0, len(aiy)):
        if dy != 0:
            row = -1
            if (aiy[col] - np.min(aiy)) != 0:
                row = int((aiy[col] - np.min(aiy))/dy)

            else:
                if col == 0:
                    row = 0
                else:
                    row = row_array[len(row_array)-1]
            if row != size:
                array[size-1-row][col] = 255
            else:
                array[0][col] = 255

            row_array.append(row)

        else:
            array[int(size/2)][col] = 255
            
    return array
# copy data to array
def getData(file_list):
    Data = []
    Data.append([])
    for i in range(0, len(file_list)):
        temp = file_list[i].split(' ')
        #del temp[len(temp)-1]
        Data[i].extend(temp)
        if i != len(file_list)-1:
            Data.append([])
    return Data

def clearErrorData(T,X):
    new_t, new_x = [], [] 
    new_t.append(T[0])
    new_x.append(X[0])
    for index,ele in enumerate(T):
        if index > 0:
            if float(ele) > float(T[index-1]):
                if float(ele) > float(new_t[len(new_t)-1]):
                    new_t.append(ele)
                    new_x.append(X[index])
                 
    return new_t,new_x

def getNewMtrix(path_x, path_t, size, str):

    encoding = 'UTF-8'
    Datax, Datat = [], []
    if str == 'train':
        Datax = read_file_list(path_x, encoding)[:5192]
        Datat = read_file_list(path_t, encoding)[:5192]
    elif str == 'predict':
        Datax = read_file_list(path_x, encoding)
        Datat = read_file_list(path_t, encoding)

    x = getData(Datax)
    t = getData(Datat)
    print(len(x), len(t))

    # 获取新矩阵，描述V-t
    new_Data_matrix = []
    new_Data_matrix.append([])
    id_new = []
    
    for i in range(0, len(x)):
        #print(x[i],len(x),i+1)
        id_new.append(i)
        #print('i:',i)
        temp_array = []
        if str == 'train':
            new_t,new_x = clearErrorData(t[i], x[i])
            temp_array = getPostion_with_interp1d(new_t, new_x, size)
            #temp_array = getPosition(t[i], x[i], size)
        elif str == 'predict':            
            temp_array = getPostion_with_interp1d(t[i], x[i], size)            
            #temp_array = getPosition(t[i], x[i], size)
        new_Data_matrix[i].append(temp_array)
        #print("提取完{0}组".format(i))
        if i != len(x)-1:
            new_Data_matrix.append([])

    return new_Data_matrix, id_new


def getTrain_Test_data(path_x_true_label, new_Data_matrix_training, id_new):
    encoding = 'UTF-8'
    Label = read_file_list(path_x_true_label, encoding)

    train_ratio = 0.6
    train_id_training = random.sample(id_new, int(len(new_Data_matrix_training)*train_ratio))
    test_id_training = []
    for id_ in id_new:
        if id_ not in train_id_training:
            test_id_training.append(id_)
   
            
    train_labels = []
    training_data = []

    for id_ in train_id_training:
        train_labels.append(Label[id_])
        training_data.append(new_Data_matrix_training[id_])

    test_labels = []
    test_data = []
    for id_ in test_id_training:
        test_labels.append(Label[id_])
        test_data.append(new_Data_matrix_training[id_])
    
    
    return training_data, train_labels, test_data, test_labels


def get_10w(size):
    path_t_pre = "C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/10w/length(5-300)/T(X).txt"
    path_x_pre = "C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/10w/length(5-300)/X.txt"
    path_x_true_id_pre = "C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/10w/length(5-300)/true_id(No,0-5,error_time).txt"
    
    new_Data_matrix_test, id_new_test = getNewMtrix(path_x_pre, path_t_pre, size, 'predict')
    
    #获取真实10w测试数据的id
    x_true_id_pre = []
    f = open(path_x_true_id_pre)
    for eachline in f.readlines():
        x_true_id_pre.append(int(eachline.strip('\n')))
    
    pre_data_ = []
    for j in range(len(new_Data_matrix_test)):
        pre_data_.append(new_Data_matrix_test[j])
    
    
    return x_true_id_pre,pre_data_

def get_newTrainData(size):
    path_new_id_pre = 'C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/10w/24585+0.6634+0.8155+71.69+16310+caffenet.txt'
    path_x_pre = "C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/10w/X.txt"
    path_t_pre = "C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/10w/T(X).txt"
    
    new_Data_matrix_test, id_new_test, del_id = getNewMtrix(path_x_pre, path_t_pre, size, 'predict')
    
    new_id_pre = np.array(read_file_list(path_new_id_pre,'utf-8')).astype(int)
    res_id_ = []
    for ele in new_id_pre:
        if ele not in del_id:
            res_id_.append(ele)
            
    pre_data_ = []
    for j in res_id_:
        pre_data_.append(new_Data_matrix_test[j-1])
        
    return new_id_pre, pre_data_    
def main(unused_argv):
    # Load training and eval data

    path_t_training="C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/black+train+10604(0.92)/len(5-300)/T(X).txt"
    path_x_training="C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/black+train+10604(0.92)/len(5-300)/X.txt"
    path_x_true_label = "C:/Users/LiDongNing/Desktop/mouse_trips/NewTrainingData/black+train+10604(0.92)/len(5-300)/True_label.txt"
    

    #将所有样本x-t绝对位置，转化为相对位置矩阵
    size = 28
    new_Data_matrix_training, id_new = getNewMtrix(path_x_training, path_t_training, size, 'train')

    training_data, train_labels, test_data, test_labels = getTrain_Test_data(path_x_true_label, new_Data_matrix_training, id_new)
    
    
    #mnist = learn.datasets.load_dataset("mnist")
    
    train_data = np.array(training_data, dtype=np.float32)  # Returns np.array
    train_labels = np.asarray(train_labels, dtype=np.int32)
    eval_data = np.array(test_data, dtype=np.float32)  # Returns np.array
    eval_labels = np.asarray(test_labels, dtype=np.int32)


    # Create the Estimator, model_dir="C:/Users/LiDongNing/Desktop/鼠标轨迹/codes/MODELS/model_.ckpt"
    mnist_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="C:/Users/LiDongNing/Desktop/mouse_trips/codes/MODELS/model(w1)(5192,Inter,0.001,32_ft,28,[5,5]).ckpt")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    mnist_classifier.fit(
          x=train_data,
          y=train_labels,
          batch_size=100,
          steps=20000,
          monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
          "accuracy":
              learn.MetricSpec(
                  metric_fn=tf.metrics.accuracy, prediction_key="classes"),
      }


    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
          x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)


    x_true_id_pre, pre_data_ = get_10w(size)
    pre_data = np.array(pre_data_, dtype=np.float32)
    print("pre_data:", len(pre_data))

    #10w条预测样本

    output = mnist_classifier.predict(pre_data, batch_size=10000)
    exportClasses = []
    for element in output:
        exportClasses.append(element.get('classes'))

    print("exportClasses", len(exportClasses), "x_true_id_pre", len(x_true_id_pre))

    f = open("C:/Users/LiDongNing/Desktop/mouse_trips/test_(10w)"+str(len(x_true_id_pre))+".txt", 'w+')
    count = 0
    for i in range(0, len(exportClasses)):
        if exportClasses[i] == 0:
            count = count + 1
            f.write(str(x_true_id_pre[i])+'\n')
    print('预测的黑样本个数为：', count)
    f.flush
    f.close



if __name__ == "__main__":
  tf.app.run()
  
