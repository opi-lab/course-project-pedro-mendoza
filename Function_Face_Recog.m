function [Recognized_Img] = Function_Face_Recog(Datapath,Testimg)

% Initially, different images of the selected training set will be aligned (X1, X2, X3, X4, ..., XN)

% For this, all the images of the training set will be taken in 2D format and
% transformed into 1D columns in order to facilitate a displacement in the 
% form of rows of all these columns 1D and achieve a 2D Matrix called 'X'.

% Datapath --> Path of the data images used for training.
% X --> A 2D matrix containing all 1D image vectors.
% All P images in the training database have the same size of MxN. So the
% length of 1D column vectors is MxN and 'X' will be a (MxN)xP 2D matrix.
% For our project, images of 300 x 300 pixels will be used.

% The total number of images stored in the route of our training set is determined. 
% Folder is the folder of Px1 where 4 fields are found: name, byte, date
% and isdir of all files present in the directory Datapath.

Folder = dir(Datapath) 
Image_Count = 0;
for i = 1 : size(Folder, 1)
    if not(strcmp(Folder(i).name, '.') | strcmp(Folder(i).name, '..') | strcmp(Folder(i).name, 'Thumbs.db'))
        Image_Count = Image_Count + 1; % Number of all images in the training database.
    end
end

% In this step, the 2D matrix that will contain the images of the training set will be created. 'X'
X = [];
for i = 1 : Image_Count
    File = strcat(Datapath,'\',int2str(i),'.jpg');
    Photo = imread(File);
    Photo = rgb2gray(Photo); % We convert the image from RGB format to grayscale to process it more easily.
    [Width, Height] = size(Photo); % We store the dimensions of the training image (width and height).
    Temporary = reshape(Photo', Width * Height, 1);  % Reshaping 2D images into 1D image vectors.
    % Here Photo' (transpose) is used because reshape(A,M,N) Function reads the matrix A columnwise.
    % Where as an image matrix is constructed with first N pixels as first row, and so on.
    X = [X Temporary];  % X, is the image matrix with ColumnsGetting added for each image.
end

% Now we calculate Average, A and EigenFaces. The descriptions are below :
% Average --> (MxN)x 1.  Mean of the training images.
% A --> (MxN)x P.  Matrix of image vectors after each vector getting subtracted from the mean vector Average.
% EigenFaces --> (MxN)x P'. P' Eigenvectors of covariance matrix (C) of
% training database X,
% where P' is the number of eigenvalues of C that best represent the feature set.

% Calculating 'Average' as the mean image vector of 'X' %

Average = mean(X,2); % Computing the Average. Face Image Average = (1/P) * Sum(Xj's) (j = 1 : P)
Image_Count2 = size(X,2);

% Calculating A matrix, after subtraction of all image vectors from the mean image vector %
A = [];
for i = 1 : Image_Count2
    % The image matrix is rewritten, considering the subtraction of each component with the average vector.
    Temporary = double(X(:, i)) - Average; 
    A = [A Temporary];
end

                       %%%%%%%%%%%%%% CALCULATION OF EIGENFACES %%%%%%%%%%%%%%%%%%
%%%  We know that for a MxN matrix, the maximum number of non-zero eigenvalues that its covariance matrix can have
%%%  is min[M-1,N-1]. As the number of dimensions (pixels) of each image vector is very high compared to number of
%%%  test images here, so number of non-zero eigenvalues of C will be maximum P-1 (P being the number of test images)
%%%  if we calculate eigenvalues & eigenvectors of C = A*A' , then it will be very time consuming as well as memory.
%%%  so we calculate eigenvalues & eigenvectors of L = A'*A , whose eigenvectors will be linearly related to eigenvectors of C.
%%%  these eigenvectors being calculated from non-zero eigenvalues of C, will represent the best feature sets.

L = A' * A;
[V, D] = eig(L); % V : Eigenvector Matrix. D : Eigenvalue Matrix. (Squares)

%%%% Again we use Kaiser's rule here to find how many principal components (Eigenvectors) to be taken
%%%% If corresponding eigenvalue is greater than 1, then the eigenvector will be chosen for creating eigenface

L_Eigen_Vectors = [];
for i = 1 : size(V,2) 
    if( D(i,i) > 1 )
        L_Eigen_Vectors = [L_Eigen_Vectors V(:, i)];
    end
end

%%% Finally the EigenFaces %%%
EigenFaces = A * L_Eigen_Vectors;

% In this part of recognition, we compare two faces by projecting the images into facespace and 
% measuring the euclidean distance between them.
% Recognized_Image --> The recognized image name.
% Testimg --> The path of test image.
% Average --> Mean image vector X.
% A --> Mean subtracted image vector matrix. 
% EigenFaces --> Eigenfaces that are calculated from eigenface function.

% Finding the projection of each image vector on the Facespace (where the
% eigenfaces are the coordinates or dimensions).

Project_Image = [ ];  % Projected image vector matrix.
for i = 1 : size(EigenFaces,2)
    Temporary = EigenFaces' * A(:,i);
    Project_Image = [Project_Image Temporary];
end

% Extracting PCA features of the test image %

Test_ImageF = imread(Testimg);
Test_ImageF = Test_ImageF(:,:,1);
[Rows, Cols] = size(Test_ImageF);
Temporary = reshape(Test_ImageF', Rows * Cols, 1); % Creating (MxN) x 1 image vector from the 2D image.
Temporary = double(Temporary) - Average; % Mean subtracted vector.
Project_TestImg = EigenFaces' * Temporary; % Projection of test image into the facespace.

% Calculating and comparing the euclidian distance of all projected trained images from the projected test image %

Euclide_Distance = [ ];
for i=1 : size(EigenFaces,2)
    temp = (norm(Project_TestImg - Project_Image(:,i)))^2;
    Euclide_Distance = [Euclide_Distance temp];
end
[Euclide_Distance_Min, Recognized_Index] = min(Euclide_Distance);
Recognized_Img = strcat(int2str(Recognized_Index),'.jpg');

[a, b] = size(X);
[c, d] = size(Average);
[e, f] = size(A);
[g, h] = size(A');
[i, j] = size(L);
[k, l] = size(V);
[m, n] = size(D);
[o, p] = size(L_Eigen_Vectors);
[q, r] = size(EigenFaces);
[s, t] = size(Project_Image);
[u, v] = size(Project_TestImg);
[w, x] = size(Temporary);
[y, z] = size(Euclide_Distance);

disp("El tamaño de la matriz X es:"); disp(a*b);
disp("El tamaño de la matriz Average es:"); disp(c*d); 
disp("El tamaño de la matriz A es:"); disp(e*f); 
disp("El tamaño de la matriz A' es:"); disp(g*h);
disp("El tamaño de la matriz L es:"); disp(i*j);
disp("El tamaño de la matriz V es:"); disp(k*l);
disp("El tamaño de la matriz D es:"); disp(m*n);
disp("El tamaño de la matriz L_Eigen_Vectors es:"); disp(o*p);
disp("El tamaño de la matriz EigenFaces es:"); disp(q*r);
disp("El tamaño de la matriz Project_Image es:"); disp(s*t);
disp("El tamaño de la matriz Project_TestImg es:"); disp(u*v);
disp("El tamaño de la matriz Temporary es:"); disp(w*x);
disp("El tamaño de la matriz Euclide_Distance es:"); disp(y*z);

