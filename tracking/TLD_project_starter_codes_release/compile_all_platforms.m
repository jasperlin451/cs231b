% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.

% Compiles mex files
clc; clear all; cd mex;

if ismac
    disp('Mac');
    
    include = ' -I/Users/Jasper/anaconda/pkgs/opencv-2.4.8-np17py27_2/include/opencv/ -I/Users/Jasper/anaconda/pkgs/opencv-2.4.8-np17py27_2/include/'; 
    libpath = '/Users/Jasper/anaconda/pkgs/opencv-2.4.8-np17py27_2/lib/'; 
    
    files = dir([libpath 'libopencv*.2.4.dylib'])
    
    lib = [];
    for i = 1:length(files),
        lib = [lib ' ' libpath files(i).name];
    end
 
    eval(['mex -largeArrayDims lk.cpp -O' include lib]);
    mex -O -c -largeArrayDims tld.cpp
    mex -O -largeArrayDims linkagemex.cpp
    mex -O -largeArrayDims bb_overlap.cpp
    mex -O -largeArrayDims warp.cpp
    mex -O -largeArrayDims distance.cpp
    
end

cd ..
disp('Compilation finished.');

