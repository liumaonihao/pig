# -*- coding: utf-8 -*-

#将视频的帧中提取图片，可以通过控制帧数来获取某一帧的图片

import cv2 
from glob import glob
import os
#定义视频地址
src='F:/python_workspace/pig/train0/*'
src0="E:/pig_data/train20/"  #软连接目录
pig_mp4=glob(src)
for i in pig_mp4:
    img_sort=i.split("train3")[1].split(".")[0].split('\\')[1]  #视频的类别
    
    videoCapture = cv2.VideoCapture() #创建视频对象
    videoCapture.open(i)                 #打开视频
    fps = videoCapture.get(cv2.CAP_PROP_FPS)  #获取帧率，意思是每一秒刷新图片的数量，
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT) #获取视频中总的图片数量
    print("fps=",fps,"frames=",frames)
    #视频中图片的长和宽
#    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),   
#        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))  
    #指定写视频的格式, I420-avi, MJPG-mp4  
#    videoWriter = cv2.VideoWriter('oto_other.mp4', cv2.CV_FOURCC('M', 'J', 'P', 'G'), fps, size)  
    #保存图片
    print(img_sort)
#    dtc=src.split("train3")[0]+"train"+"/"+img_sort+"/" #设定保存路径
    dtc=src0+img_sort+"/"
    print(dtc)
    if os.path.exists(dtc)==0:  #没有创建，自动创建文件目录
        os.makedirs(dtc)
    for j in range(int(frames)):
        ret,frame = videoCapture.read()   #获取下一帧 ,调整
        if((j+1)%20==1):             #20一帧读取
            s="%s-%d.jpg"%(img_sort,j+1)  #图片格式
            print(s)
            cv2.imwrite(dtc+s,frame)
