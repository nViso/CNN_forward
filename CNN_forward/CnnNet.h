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
	std::vector<CnnLayer*> structure;//����CnnLayer����ָ��
	Model* model;
	void forward(const cv::Mat&);
	void forward(const std::string path, int mode);
	void init(std::string FilePath, std::string Key);//����Layer������ģ�Ͳ�����Layer��
	std::vector<int> argmax(const std::vector<int>& layer_nums);//��ָ�����Ŷ�ȡ����������argmax
	std::vector<int> argmax();//��net�ж����Ľ�������ȡ����������argmax
	std::vector<vector<float> > face_info(const std::vector<int>& layer_nums);
	std::vector<vector<float> > face_info();
private:
	void proc_layers(std::vector<LayerConfig*>);//��layer_config����structure�ĳ�ʼ������
	std::vector<int> result_layer;//���Ų��������Ĳ�������
};

#endif
