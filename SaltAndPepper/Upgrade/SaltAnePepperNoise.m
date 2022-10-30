inputImage = imread('8k_lake.jpg');
outputImage = imnoise(inputImage,'salt & pepper',0.02);
imwrite(outputImage, '8k_input_02.jpg');