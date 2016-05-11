/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __CnnLayer__
#define __CnnLayer__
#define HAVE_TBB
#include<vector>
#include<opencv2\core\core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp> 
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

class CnnLayer{
public:
	std::vector<int> parents;//index of last layer
	std::vector<cv::Mat> result;// result of this layer
	virtual void forward(const std::vector<CnnLayer*>& structure) = 0;//implement forward and store the result to result
};
#endif