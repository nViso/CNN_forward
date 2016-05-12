# CNN_forward
#### A [Convolution Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) forward code for [caffe](http://caffe.berkeleyvision.org/) implemented in C++.

The initial version is forked from github, for details about the very initial one, click [here](https://github.com/ihciah/CNN_forward)

#### Dependence
This small CNN_forward just need two dependence, one is opencv and the other is Intel's TBB which makes convolution faster on multicore CPU

The Opencv version I use is 2.4.11, and I cmake it to support vc 14 for x86 structure.  Here is the guidance show how to cmake it

http://amin-ahmadi.com/2015/12/04/how-to-build-opencv-from-source-for-vc14/

You need first download the opencv from official webside, and then cmake it according to this guidance. After doing this, you need to first add it to your envirment variables and then set it in your visual studio. Here is the guidance show how to set environment variables and add it to visual studio.

http://opencv-srf.blogspot.ch/2013/05/installing-configuring-opencv-with-vs.html

Note: instead of vc11 you need vc14 for visual studio 2015, modify this in all the pathes. As the bin folder contains both debug and release, you should add 2 paths to the PATH environment variable, they are
`%OPENCV_DIR%\x86\vc14\bin\Debug and %OPENCV_DIR%\x86\vc14\bin\Release`,
 rather than the `%OPENCV_DIR%\x86\vc11\bin` in the above guidance. What's more, in the visual studio, in debug mode, add `$(OPENCV_DIR)\x86\vc14\lib\Debug and list the *.libs in the Debug folder`. In release mode, add `$(OPENCV_DIR)\x86\vc14\lib\Release` and list the *.libs in the Release folder(the linker general and input part). You can add a new filder called "opencv2" and copy all the files in opencv\include\opencv2 to this folder.

 For the TBB, it is the same step as opencv, you can download the official version from [here](https://www.threadingbuildingblocks.org).

Now you have already set all the dependence, if visual studio can not open head files in opencv2, close visual studio and open it again, remember to save the changes before close it.


#### Build on Visual Studio 2015

First clone this project from the git，then create an empty project with Visual Studio 2015. In the Solution Explorer, right click Source Files -> Add Existing Item, then choose the main.cpp from the project you forked from git.

Before you can run it, you must do these things:

    Right click on your project's entry in solution explorer,

    Select Properties,

	Choose C/C++, then click the General,

	In the "Additional Include Directories", add the whole project you forked from git, it is \path\to\CNN_forward\CNN_forward

	Still in C/C++, Select Preprocessor,

    add these flags to "Preprocessor Definitions":

        _CRT_SECURE_NO_DEPRECATE

    one flag per line.

After this, choose Source Files in Solution Explorer, right click and add New Filter, the Filter name should be 'CNN_forward', and then copy all the files in the folder 'CNN_forward' from the git project, to this New Filter

Now, you can build your own solution

#### nViso Example
The nViso example just do a feed forward to get the landmarks and headposes, the output of 'ip0_2' is the corresponding facial landmarks, which are 32 pairs of float numbers
