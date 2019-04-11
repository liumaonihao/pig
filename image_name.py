# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:51:52 2017

@author: lenovo
"""
#将图片的格式进行替换
#使用PIL的话可以通过参数设定几种默认的图片格式，
#也可以使用路径保存，不使用参数，可以保存为想要保存的图片格式
from glob import glob
from PIL import Image
import numpy as np
imglist=glob("F:/python_workspace/pig/test/test0/*")#源文件路径
for i in imglist:
    im=Image.open(i)
#    im_array=np.asarray(im)
    src=i.split("\\")[0]+"//"+"test"+"//"+i.split("\\")[1]
    src=i.split("test0")[0]+"test"+i.split("test0")[1]   #保存路径
    print(src)
    src=src.split('.')[0]+"."+'jpeg' 
    print(src)
#    im.save(src)
