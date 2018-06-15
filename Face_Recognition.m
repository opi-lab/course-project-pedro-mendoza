%%% Face Recognition by PCA (Principal Component Analysis)
clear all % Clear all the function.
close all % Close all the function.
clc      

% In the following lines, the user must enter the access routes to the stored dataset.
Train_Path = uigetdir('C:\Documents and Settings\PedroMendoza\My Documents\MATLAB','Select Path Of Training Images');
Test_Path = uigetdir('C:\Documents and Settings\PedroMendoza\My Documents\MATLAB','Select Path Of Test Images');

% In the next block, the user will be asked to enter the test image with 
% which he desires to validate the correct operation of the training set.
Input = {'Enter Test Image Name (A number between 1 - 9):'};
Title_Input = 'Input Of PCA-Based Face Recognition System';
Number_Lines = 1; % Define the number of cells to enter the necessary response.
Space = {''}; % Space of the rectangle for answer

%The inputdlg function receives as arguments the question, the title, the lines and the start spacing.
Test_Image = inputdlg(Input,Title_Input,Number_Lines,Space); 
disp("Selected Image: "); disp(Test_Image);

% Concatenate several strings in order to write the path of the test image.
Test_Image = strcat(Test_Path, '\', char(Test_Image), '.jpg');

%%%%  The following block of comments highlights the call to the main %%%   
%%%%   function and the implementation of some innate Matlab methods %%% 
%%%%   in order to visualize the results obtained from the training. %%%

% Call of the main function that receives as parameters the path of the training set and the selected test image.
Recognition_Image = Function_Face_Recog(Train_Path,Test_Image); 

%The result obtained from the main function is read and the best image route (similar to the test one) given by the training is obtained.

SelectedP_Image = strcat(Train_Path,'\',Recognition_Image);
Selected_Image = imread(SelectedP_Image); % We get the image using the imread function of Matlab.
imshow(Selected_Image); % We visualize the image using the imshow function of Matlab.
title('Recognized Image');
Test_Face = imread(Test_Image); % We get the image using the imread function of Matlab.
figure,imshow(Test_Face);
title('Test Image');

result = strcat("The Recognized Image is : ", Recognition_Image);
disp(result);