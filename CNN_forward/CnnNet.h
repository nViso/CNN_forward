/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __CnnNet__
#define __CnnNet__

#include<vector>
#include<string.h>
#include<opencv2\core\core.hpp>

#include "CnnAllLayers.h"
#include "ModelReader.h"
#include "LayerConfig.h"
using namespace std;

#define COLOR 1
#define GRAY 0

class CnnNet {
public:
	std::vector<CnnLayer*> structure;//store CnnLayer subclass pointer
	Model* model;
	void forward(const cv::Mat&);
	void forward(const std::string path, int mode);
	void init(std::string FilePath, std::string Key);//make layer and load parameters to layers
	std::vector<int> argmax(const std::vector<int>& layer_nums);//read value from corresponding layer index, return argmax
	std::vector<int> argmax();//read result from this result layer and return argmax
	std::vector<vector<float> > face_info(const std::vector<int>& layer_nums);// return result of facial information(nViso model)
	std::vector<vector<float> > face_info();
private:
	void proc_layers(std::vector<LayerConfig*>);//initialize structure by layer_config
	std::vector<int> result_layer;//store the layer index of the result layer
};

#endif
