/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#pragma once
#ifndef __ModelReader__
#define __ModelReader__

#include<string.h>
#include<vector>
#include<opencv2\core\core.hpp>

class DataBlock{
public:
	std::string name;
	char type;
	int dim[4];
	std::vector<char*> data_bin;
	std::vector<std::vector<char>> data_vec;
};

class Model{
public:
	Model(std::string path);//directly load path to Model
	Model(std::string path,std::string key);//decrypt and then load to Model
	Model();
	void load(std::string model_path);
	void load(std::string model_path,std::string key);
	std::vector<std::vector<cv::Mat>> get_weight(std::string layer_name);//return Weight of corresponding convolutional layer
	cv::Mat get_bias(std::string layer_name);//�return Bias of corresponding convolutional layer
	std::vector<std::vector<cv::Mat>> Model::get_fc_weight(std::string layer_name);//return Weight of corresponding fully connect layer
	std::vector<int> Model::get_dim(std::string layer_name);//�return dimension of corresponding layer
private:
	std::vector<DataBlock*> data;
};

#endif