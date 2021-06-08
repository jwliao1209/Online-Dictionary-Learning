function I = imgData(filename)
   I = imread(filename);
   I = double(I) / 255;
end