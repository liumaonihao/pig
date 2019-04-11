# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:40:24 2017

@author: lenovo
"""


#%%
#导入特征向量（知名网络结构）
from keras.models import *
from keras.layers import *
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.mobilenet import Mobilenet
from keras.applications import *
from keras.applications.resnet50 import ResNet50
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import MobileNet
from keras.applications import imagenet_utils
from keras.applications.xception import Xception
from keras.preprocessing.image import *
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
import h5py

def write_gap(MODEL, image_size, lambda_func=None):
#    global test_generator
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    print(x)
    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    print(len(base_model.layers[:]))
    #进行微调，解锁部分网络
    model_finetune = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    
#    model_finetune.summary() 网络模型可视化
    print(len(model_finetune.layers[:]))
#    for layer in model_finetune.layers[-34:]:
#        layer.trainable = True
    #使用预训练模型,加载预训练权重
#    model_finetune = Model(inputs=base_model.input, outputs=y, name='Fine-tuning')
#    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
#    model_finetune.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("F:/python_workspace/pig/train20/", image_size, shuffle=False, batch_size=16)
    test_generator = gen.flow_from_directory("F:/python_workspace/pig/test", image_size, shuffle=False, batch_size=16, class_mode=None)
    #加载训练数据
    train = model_finetune.predict_generator(train_generator,train_generator.samples//16+1,verbose=1)
    test = model_finetune.predict_generator(test_generator, test_generator.samples//16+1,verbose=1)
    #预测得到特征向量
    with h5py.File("F:/python_workspace/bingli/model/%s-1.h5"%MODEL.name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)


#write_gap(ResNet50, (224, 224))
#write_gap(MobileNet,(224,224,3))
#write_gap(InceptionResNetV2,(299,299))
write_gap(InceptionV3, (299, 299))
#write_gap(Xception, (299, 299))
#write_gap(VGG16, (224, 224))
#write_gap(VGG19, (224, 224))
#h=h5py.File('./keras/gap/InceptionV3.h5', 'r')
#print(h['train'].shape)


#%%
# =============================================================================
# #读取模型特征向量训练以及预测
# import keras
# from keras.models import Sequential
# from keras.layers import Activation,Dropout,Dense
# from keras.regularizers import l2
# import numpy as np
# from sklearn.utils import shuffle
# import matplotlib.pyplot as plt
# X_train = []
# X_test = []
# # "model/ResNet50.h5" , "model/Xception.h5","model/VGG19.h5","model/inceptionV3.h5", "model/VGG16.h5"
# # "model/ResNet50_20.h5" , "model/Xception_20.h5","model/VGG19_20.h5","model/inceptionV3_20.h5", "model/VGG16_20.h5"
# filePath="F:/python_workspace/pig/"
# model_list=[ "model/VGG19_20.h5","model/VGG16_20.h5","model/VGG19_20.h5"]
# for filename in model_list:
#     filename=filePath+filename
#     
#     with h5py.File(filename, 'r') as h:
#         X_train.append(np.array(h['train']))
#         X_test.append(np.array(h['test']))
#         y_train = np.array(h['label'])
# 
# X_train = np.concatenate(X_train, axis=1)#拼接特征向量
# X_test = np.concatenate(X_test, axis=1)
# X_train, y_train = shuffle(X_train, y_train) #洗牌
# #y_train= keras.utils.to_categorical(y_train) #单分类
# y_train= keras.utils.to_categorical(y_train , num_classes=30) #多分类
# #print(set(y_train)) #显示类别
# 
# #定制模型
# 
# input_tensor = Input(X_train.shape[1:])
# x = Dropout(0.5)(input_tensor)
# x=Dense(1024,activation='relu',W_regularizer=l2(0.01))(x)
# x=Dropout(0.5)(x)
# x=Dense(30,activation='softmax')(x)
# model=Model(input_tensor,x)
# 
# 
# 
# #model=Sequential()  #通过input_dim和input_shape来指定输入的维数
# #model.add(Dense(1024,activation='relu',W_regularizer=l2(0.01),input_dim=X_train.shape[1]))
# #model.add(Dropout(0.5,input_shape=X_train.shape[1:]))
# #model.add(Dense(1024,activation='relu',W_regularizer=l2(0.01)))
# #model.add(Dropout(0.5))
# #model.add(Dense(1024,activation='relu',W_regularizer=l2(0.01)))
# #model.add(Dropout(0.5,input_shape=X_train.shape[1:]))
# #model.add(Dense(1,activation='sigmoid',W_regularizer=l2(0.01)))
# #model.add(Dense(2,activation='sigmoid',W_regularizer=l2(0.01),init='normal'))
# #model.add(Dense(30,activation='softmax',W_regularizer=l2(0.01),init='normal',input_shape=X_train.shape[1:]))
# 
# #编译模型
# model.compile(optimizer='adadelta',
#               loss='categorical_crossentropy',
# #             loss='binary_crossentropy',
#              metrics=['accuracy'])
# #训练模型
# history=model.fit(X_train, y_train, batch_size=256, epochs=10000, validation_split=0.2,verbose=2)
# =============================================================================
#%%
#画图loss和acc

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('train val loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['loss', 'val_loss'], loc='upper right')


#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('train val acc')
#plt.ylabel('acc')
#plt.xlabel('epoch')
#plt.legend(['acc', 'val_acc'], loc='upper left')
#%%
#预测
#gen = keras.preprocessing.image.ImageDataGenerator() 
#test_generator = gen.flow_from_directory("test2", (224, 224), shuffle=False, 
#                                         batch_size=16, class_mode=None)


#%%
#y_=model.predict(X_test)
#y_ = y_.clip(min=0.005, max=0.995)  #将结果控制在0.005和0.995之间


#%%
#将预测结果进行输出
#import pandas as pd
#from keras.preprocessing.image import *
#
#df =pd.DataFrame()
##data生成器
#gen = ImageDataGenerator()
#test_generator = gen.flow_from_directory(filePath+"/test", (224, 224), shuffle=False, 
#                                         batch_size=16, class_mode=None)
##for i, fname in enumerate(test_generator.filenames):
##    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
##    df.set_value(index-1, 'label', y_pred[i])
##输入到文件
#imglabels=[]
#for i, fname in enumerate(test_generator.filenames):
#    index = str(fname[fname.rfind('/')+1:fname.rfind('.')]).split("\\")[1]
##    imglabels.append(index)
##for i in range(len(X_test)):
##    imglabel=imglabels[i]
#    imglabel=index
#    if (i+1)%100==0:print("图片处理张数：第%d张"%(i+1))
#    max_pred=max(y_[i])
#    for j in range(len(y_[i])):
#        df.set_value(i*30+j,"picture_id",imglabel)
#        df.set_value(i*30+j,"label_id",j+1)
#        if(y_[i][j]<max_pred):      #如果比率最大为
#            df.set_value(i*30+j,"y_predict",0.005)
#        else:
#            df.set_value(i*30+j,"y_predict",0.995)
##        df.set_value(i*30+j,"y_predict",y_[i][j])
##    print(y_.shape)
##    df.set_value()
##    df.set_value(i, 'label0', y_[i][0])
##    df.set_value(i,'label1',y_[i][1])
##df.head(10)
#df.to_csv(filePath+'pred.csv', index=None)

#%%
#import pandas as pd
#from keras.preprocessing.image import *
#gen = ImageDataGenerator()
#test_generator = gen.flow_from_directory("./test", (224, 224), shuffle=False, 
#                                         batch_size=16, class_mode=None)
#train_generator = gen.flow_from_directory("./train20/", (224, 224), shuffle=False, 
#                                         batch_size=16, class_mode=None)
#lablelist=train_generator.class_indices
#
#idlist=[]
#biaojilist=[]
#resultlist=[]
#b=0.0
#c=0.0
#for i, fname in enumerate(test_generator.filenames):
#    id = fname[fname.rfind('\\')+1:fname.rfind('.')]
#    for j in range(30):
#        idlist.append(id)
#        a=list(lablelist.keys())[list(lablelist.values()).index(j)]
#        biaojilist.append(a)
#
#        resultlist.append(y_[i][j])
#    max_index=np.argmax(y_[i])
#           
#submission = pd.DataFrame({'1a':idlist,'1b':biaojilist,"1c":resultlist})
#submission.to_csv("submission2.csv", encoding = "UTF-8",index=False,header=False)

#%%

#模型可视化

#from keras.utils import plot_model
#plot_model(model.summary(), to_file=filePath+'model.png', show_shapes=True)




