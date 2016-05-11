/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __Conv__
#define __Conv__
#include"CnnLayer.h"

class Conv :public CnnLayer{
public:
	cv::Mat bias;
	std::vector<std::vector<cv::Mat>> weight;
	int dim[4];
	// add the pad to support padding
	int pad;
	void forward(const std::vector<CnnLayer*>& structure);
};

#endif