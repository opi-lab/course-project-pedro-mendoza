# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:25:28 2018

@author: PEDRO NEL MENDOZA
"""
from skimage import io
import os, os.path
from PIL import Image
from numpy import matrix
import numpy as geek
import numpy as np
import matplotlib.pyplot as plt

def facerecog(datapath, testimg):

    # In this part of function, we align a set of face images (the training set x1, x2, ... , xM )
    #
    # This means we reshape all 2D images of the training database
    # into 1D column vectors. Then, it puts these 1D column vectors in a row to 
    # construct 2D matrix 'X'.
    #  
    #
    #          datapath   -    path of the data images used for training
    #               X     -    A 2D matrix, containing all 1D image vectors.
    #                                        Suppose all P images in the training database 
    #                                        have the same size of MxN. So the length of 1D 
    #                                        column vectors is MxN and 'X' will be a (MxN)xP 2D matrix.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #%%%%%%%%  finding number of training images in the data path specified as argument  %%%%%%%%%%
    
    D =  os.listdir(datapath)# D is a Lx1 structure with 4 fields as: name,date,byte,isdir of all L files present in the directory 'datapath'
    print(D)
    imgcount = 0
    sizefolder = len([name for name in os.listdir(datapath) if os.path.isfile(os.path.join(datapath, name))])
    for i in range(sizefolder):
        print "El archivo " + str(i+1) + " es: " + D[i] + "\n"
        if not(D[i] == "." or D[i] == ".." or D[i] == "Thumbs.db"):
            imgcount = imgcount + 1        # Number of all images in the training database
            print "El total de imagenes .jpg es: " + str(imgcount) + " Imagenes."

            #%%%%%%%%%%%%%%%%%%%%%%%%%  creating the image matrix X  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    X = []    
    for i in range(imgcount):
        archivo = datapath + "\\" + str(i+1) + ".jpg"
        img = io.imread(archivo)
        img = Image.open(archivo).convert('L')
        #plt.imshow(img)
        #img.save('greyscale.png')
        #img = io.rgb2gray(img)
        width,height = img.size
        #imgT = img
        img = geek.array(img)
        #imgT = geek.array(img)
        print "Las dimensiones de la imagen " + str(i+1) + " son: " + str(width) + " x " +  str(height)
        print "El tamaño de la imagen " + str(i+1) + " es: " + str(width * height) + " pixeles."
        # for i in range(width):
           # for j in range(height):
                #print "Pixel " + str(i) + str(j) + ": " + str(img[i][j]) + "\n"
                #imgT[j][i] = img[i][j]
       # print("matriz transpuesta de la imagen\n", imgT)
       #matriz = geek.arange(10).reshape(5,2)
       #print("matriz de prueba funcion geek.arange.reshape", matriz)
       #temp = img.flatten()
        temp = img.T.reshape(width*height,1)
        
        #temp = geek.arange(imgT).reshape(width * height, 1)
        #print("Matriz resultante temporal", temp)
        #temp = reshape(img.cT, width * height, 1)            #% Reshaping 2D images into 1D image vectors
        #% here img' is used because reshape(A,M,N) function reads the matrix A columnwise
        #% where as an image matrix is constructed with first N pixels as first row,next N in second row so on
        X.append(temp[i])
    
        
        #X = [X, temp]          #% X,the image matrix with columnsgetting added for each image
        print("Fila" + str(i) + "Matriz X " , X)
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #   Now we calculate m, A and eigenfaces.The descriptions are below :
                #
                #          m           -    (MxN)x1  Mean of the training images
                #          A           -    (MxN)xP  Matrix of image vectors after each vector getting subtracted from the mean vector m
                #     eigenfaces       -    (MxN)xP' P' Eigenvectors of Covariance matrix (C) of training database X
                #                                    where P' is the number of eigenvalues of C that best represent the feature set
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                #%%%% calculating mean image vector %%%%%
    print("Tamaño de temp: " + str(len(temp)))
    m = np.mean(X, axis = 0)      # Computing the average face image m = (1/P)*sum(Xj's)    (j = 1 : P)
    imgcount = len(X)
    

                #%%%%%%%  calculating A matrix, i.e. after subtraction of all image vectors from the mean image vector %%%%%%

    A = []
    n = 0
    for i in range(imgcount):
        n = X[i]
        print ("el dato " + str(i+1) + " es: " + str(n))
        temp[i] = float(n) - m
        A.append(temp[i])

                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATION OF EIGENFACES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    #%%  we know that for a MxN matrix, the maximum number of non-zero eigenvalues that its covariance matrix can have
                    #%%  is min[M-1,N-1]. As the number of dimensions (pixels) of each image vector is very high compared to number of
                    #%%  test images here, so number of non-zero eigenvalues of C will be maximum P-1 (P being the number of test images)
                    #%%  if we calculate eigenvalues & eigenvectors of C = A*A' , then it will be very time consuming as well as memory.
                    #%%  so we calculate eigenvalues & eigenvectors of L = A'*A , whose eigenvectors will be linearly related to eigenvectors of C.
                    #%%  these eigenvectors being calculated from non-zero eigenvalues of C, will represent the best feature sets.
                    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    A = geek.array(A)
    L = A.T * A
    #L = A.T * A
    L = geek.array(L)
    for i in range(len(L)):
        print ("el elemento " + str(i) + " es: " + str(L[i]))
    D, V = geek.linalg.eig(L)                #% V : eigenvector matrix  D : eigenvalue matrix

                    #%%% again we use Kaiser's rule here to find how many Principal Components (eigenvectors) to be taken
                    #%%% if corresponding eigenvalue is greater than 1, then the eigenvector will be chosen for creating eigenface

    L_eig_vec = []
    D = geek.array(D)
    #print ("Filas y Columnas de V: " + str(fil) + " y " + str(col)) 
    print ("El tamaño de V es " + str(len(V)))
    for i in range(len(V)):
        print ("elemento " + str(i) + "," + str(i) + str(D[i]))
        if ( D[i] > 1):
            L_eig_vec.append(D[i])
            #L_eig_vec = [L_eig_vec, V[:,i]]

                            #%% finally the eigenfaces %%%
    eigenfaces = []
    print (len(A))
    print (len(L_eig_vec))
    #eigenfaces = geek.dot(A, L_eig_vec)
    eigenfaces = A * L_eig_vec
        
                            #In this part of recognition, we compare two faces by projecting the images into facespace and 
                            # measuring the Euclidean distance between them.
                            #
                            #            recogimg           -   the recognized image name
                            #             testimg           -   the path of test image
                            #                m              -   mean image vector
                            #                A              -   mean subtracted image vector matrix
                            #           eigenfaces          -   eigenfaces that are calculated from eigenface function
                            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                            #%%%%%% finding the projection of each image vector on the facespace (where the eigenfaces are the co-ordinates or dimensions) %%%%%

    projectimg = []                        # projected image vector matrix
    eigenfaces = geek.array(eigenfaces)
    print ("El tamaño de eigenfaces: " + str(len(eigenfaces)))
    for i in range(len(L_eig_vec)):
    #for i in range(len(eigenfaces)):
        temp = geek.dot(eigenfaces.T, A)
        #temp = eigenfaces.T * A
        print("elemento " + str(i) + "de temp: " + str(temp[i]))
        projectimg.append(temp[i])
        #projectimg = [projectimg, temp]
    print(len(temp))
    print(len(projectimg))
                                #%%%% extractiing PCA features of the test image %%%%%                   
    test_image = io.imread(testimg)
    test_image = test_image[:,:,1]
    test_image = geek.array(test_image)
    #width, height = test_image.size
    print(test_image.size)
    temp = test_image.T.reshape(test_image.size, 1) # creating (MxN)x1 image vector from the 2D image
    for i in range(len(temp)):
        temp[i] = float(temp[i]) - m                            # mean subtracted vector
    #temp = matrix(temp)
    #eigenfaces = matrix(eigenfaces)
    temp = np.asarray(temp)
    eigenfaces = np.asarray(eigenfaces)
    
    
    
    print("Tamaño de X: " + str(len(X)))
    print("Tamaño de m: " + str(len(m)))
    print("Tamaño de A: " + str(len(A)))
    print("Tamaño de L: " + str(len(L)))
    print("Tamaño de V: " + str(len(V)))
    print("Tamaño de D: " + str(len(D)))
    print("Tamaño de L_eig_vec: " + str(len(L_eig_vec)))
    print("Tamaño de eigenfaces: " + str(len(eigenfaces)))
    print("Tamaño de projectimg: " + str(len(projectimg)))
    
    
    
    projtestimg = eigenfaces.T * temp   
    #projtestimg = geek.dot(eigenfaces.T, temp)                         # projection of test image onto the facespace

                                #%%%% calculating & comparing the euclidian distance of all projected trained images from the projected test image %%%%%
    print("Tamaños de PJTT y PJT: " + str(len(projtestimg)) + " y " + str(len(projectimg)))
    euclide_dist = []
    for i in range(len(eigenfaces)):
        temp[i] = (geek.linalg.norm(projtestimg - projectimg)) ** 2
        euclide_dist.append(temp[i])
        #euclide_dist = [euclide_dist, temp]
        print ("Elemento " + str(i) + " Euclide_dist: " + str(euclide_dist[i]))
    euclide_dist_min, recognized_index = min(euclide_dist)
    recognized_img = str(recognized_index) + ".jpg"
    print ("La imagen reconocida es la numero " + str(recognized_img))
    
    
    
    