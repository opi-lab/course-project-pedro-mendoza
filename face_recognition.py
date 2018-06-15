# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:24:28 2018

@author: PEDRO NEL MENDOZA
"""

from PIL import Image
import os, os.path
from skimage import io
import matplotlib.pyplot as plt
import facerecog
#%% face recognition by Kalyan Sourav Dash %%%

#%%%%%%  provide the data path where the training images are present  %%%%%%%
#%% if your matlab environment doesn't support 'uigetdir' function
#%% change those lines in code for datapath and testpath as :
# datapath = 'give here the path of your training images';
# testpath = 'similarly give the path for test images';
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datapathR = ("E:\VARIOS PEDRO MIS DOCUMENTOS\MAESTRIA EN INGENIERIA\Vision Artificial\Final Project of Artificial Vision\Training Images")
testpath = ("E:\VARIOS PEDRO MIS DOCUMENTOS\MAESTRIA EN INGENIERIA\Vision Artificial\Final Project of Artificial Vision\Test Images")

TestImage = raw_input("Enter test image name (a number between 1 to 17): ")
TestImage = testpath + "\\" + str(TestImage) + ".jpg"


sizefolder = len([name for name in os.listdir(datapathR) if os.path.isfile(os.path.join(datapathR, name))])
print("Archivos: " + str(sizefolder))
print("La direccion de la imagen de test es: " + TestImage)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%  calling the functions  %%%%%%%%%%%%%%%%%%%%%%%%

recog_img = facerecog.facerecog(datapathR, TestImage)
selected_img = datapathR + "\\" + recog_img
select_img = io.imread(selected_img)
plt.imshow(select_img)
plt.title("Recognized Image")
test_img = io.imread(TestImage)
plt.figure
plt.imshow(test_img)

plt.title("Test Image")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
result = "the recognized image is : " + recog_img
plt.disp(result)