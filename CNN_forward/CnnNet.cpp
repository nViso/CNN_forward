/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include"CnnNet.h"
#include<iostream>
#include"LayerConfig.h"
#include"CnnAllLayers.h"
#include<vector>

using namespace std;
using namespace cv;

void LayerConfig::init_layer(std::string name, char type, vector<std::string> parent_names, bool is_output, int pad) {
	this->name = name;
	this->parent_names = parent_names;
	this->type = type;
	this->poolsize = 2;
	this->method = MAX_POOLING;
	this->is_output = is_output;
	this->pad = pad;
}
LayerConfig::LayerConfig(std::string name, char type) {
	this->w = 0;
	this->h = 0;
	this->init_layer(name, type, vector<string>(), false, 0);
}

LayerConfig::LayerConfig(std::string name, char type, int pad) {
	this->w = 0;
	this->h = 0;
	this->init_layer(name, type, vector<string>(), false, pad);
}

LayerConfig::LayerConfig(std::string name, char type, int pad, bool is_output) {
	this->w = 0;
	this->h = 0;
	this->init_layer(name, type, vector<string>(), is_output, pad);
}

LayerConfig::LayerConfig(std::string name, char type, bool is_output) {
	this->init_layer(name, type, vector<string>(), is_output, 0);
}
LayerConfig::LayerConfig(std::string name, char type, vector<std::string> parent_names) {
	this->init_layer(name, type, parent_names, false, 0);
}
LayerConfig::LayerConfig(std::string name, char type, vector<std::string> parent_names, bool is_output) {
	this->init_layer(name, type, parent_names, is_output, 0);
}
LayerConfig::LayerConfig(std::string name, char type, std::string parent_name) {
	vector<string> parent_names;
	parent_names.push_back(parent_name);
	this->init_layer(name, type, parent_names, false,0);
}
LayerConfig::LayerConfig(std::string name, char type, std::string parent_name, bool is_output) {
	vector<string> parent_names;
	parent_names.push_back(parent_name);
	this->init_layer(name, type, parent_names, is_output, 0);
}
LayerConfig::LayerConfig(std::string name, char type, int w, int h) {
	this->w = w;
	this->h = h;
	this->init_layer(name, type, vector<string>(), false, 0);
}

LayerConfig::LayerConfig(std::string name, char type, int w, int h, bool is_output) {
	this->w = w;
	this->h = h;
	this->init_layer(name, type, vector<string>(), is_output, 0);
}









void CnnNet::init(string FilePath, string Key = "") {
	vector<LayerConfig*> layer_config;

	LayerConfig* input = new LayerConfig("input", INPUT, 50, 50);
	layer_config.push_back(input);

	// 1 means that the padding parameter is 1
	LayerConfig* conv0_1 = new LayerConfig("conv0_1", CONV, 1);
	layer_config.push_back(conv0_1);


	LayerConfig* relu0_1 = new LayerConfig("relu0_1", RELU);
	layer_config.push_back(relu0_1);


	LayerConfig* conv0_2 = new LayerConfig("conv0_2", CONV, 1);
	layer_config.push_back(conv0_2);


	LayerConfig* relu0_2 = new LayerConfig("relu0_2", RELU);
	layer_config.push_back(relu0_2);

	LayerConfig* pool0 = new LayerConfig("pool0", POOLING);
	layer_config.push_back(pool0);//default 2x2 Max Pooling

	LayerConfig* conv1_1 = new LayerConfig("conv1_1", CONV, 1);
	layer_config.push_back(conv1_1);

	LayerConfig* relu1_1 = new LayerConfig("relu1_1", RELU);
	layer_config.push_back(relu1_1);

	LayerConfig* conv1_2 = new LayerConfig("conv1_2", CONV, 1);
	layer_config.push_back(conv1_2);

	LayerConfig* relu1_2 = new LayerConfig("relu1_2", RELU);
	layer_config.push_back(relu1_2);

	LayerConfig* pool1 = new LayerConfig("pool1", POOLING);
	layer_config.push_back(pool1);



	LayerConfig* conv2_1 = new LayerConfig("conv2_1", CONV, 1);
	layer_config.push_back(conv2_1);

	LayerConfig* relu2_1 = new LayerConfig("relu2_1", RELU);
	layer_config.push_back(relu2_1);

	LayerConfig* conv2_2 = new LayerConfig("conv2_2", CONV, 1);
	layer_config.push_back(conv2_2);

	LayerConfig* relu2_2 = new LayerConfig("relu2_2", RELU);
	layer_config.push_back(relu2_2);

	LayerConfig* pool2 = new LayerConfig("pool2", POOLING);
	layer_config.push_back(pool2);


	LayerConfig* conv3_1 = new LayerConfig("conv3_1", CONV, 1);
	layer_config.push_back(conv3_1);

	LayerConfig* relu3_1 = new LayerConfig("relu3_1", RELU);
	layer_config.push_back(relu3_1);

	LayerConfig* conv3_2 = new LayerConfig("conv3_2", CONV, 1);
	layer_config.push_back(conv3_2);

	LayerConfig* relu3_2 = new LayerConfig("relu3_2", RELU);
	layer_config.push_back(relu3_2);

	LayerConfig* pool3 = new LayerConfig("pool3", POOLING);
	layer_config.push_back(pool3);

	// point out the last layer is 'pool3', add string to prevent being converted to bool
	LayerConfig* ip0_1 = new LayerConfig("ip0_1", DENSE, (string)"pool3");
	layer_config.push_back(ip0_1);

	LayerConfig* relu0_1_ip = new LayerConfig("relu0_1_ip", RELU);
	layer_config.push_back(relu0_1_ip);

	LayerConfig* ip0_2 = new LayerConfig("ip0_2", DENSE, true);
	layer_config.push_back(ip0_2);


	LayerConfig* ip1_1 = new LayerConfig("ip1_1", DENSE, (string)"pool3");
	layer_config.push_back(ip1_1);

	LayerConfig* relu1_1_ip = new LayerConfig("relu1_1_ip", RELU);
	layer_config.push_back(relu1_1_ip);

	LayerConfig* ip1_2 = new LayerConfig("ip1_2", DENSE, true);
	layer_config.push_back(ip1_2);


	LayerConfig* ip2_1 = new LayerConfig("ip2_1", DENSE, (string)"pool3");
	layer_config.push_back(ip2_1);

	LayerConfig* relu2_1_ip = new LayerConfig("relu2_1_ip", RELU);
	layer_config.push_back(relu2_1_ip);

	LayerConfig* ip2_2 = new LayerConfig("ip2_2", DENSE, true);
	layer_config.push_back(ip2_2);



	LayerConfig* ip3_1 = new LayerConfig("ip3_1", DENSE, (string)"pool3");
	layer_config.push_back(ip3_1);

	LayerConfig* relu3_1_ip = new LayerConfig("relu3_1_ip", RELU);
	layer_config.push_back(relu3_1_ip);

	LayerConfig* ip3_2 = new LayerConfig("ip3_2", DENSE, true);
	layer_config.push_back(ip3_2);


	LayerConfig* ip4_1 = new LayerConfig("ip4_1", DENSE, (string)"pool3");
	layer_config.push_back(ip4_1);

	LayerConfig* relu4_1_ip = new LayerConfig("relu4_1_ip", RELU);
	layer_config.push_back(relu4_1_ip);

	LayerConfig* ip4_2 = new LayerConfig("ip4_2", DENSE, true);
	layer_config.push_back(ip4_2);


	LayerConfig* ip5_1 = new LayerConfig("ip5_1", DENSE, (string)"pool3");
	layer_config.push_back(ip5_1);

	LayerConfig* relu5_1_ip = new LayerConfig("relu5_1_ip", RELU);
	layer_config.push_back(relu5_1_ip);

	LayerConfig* ip5_2 = new LayerConfig("ip5_2", DENSE, true);
	layer_config.push_back(ip5_2);


	LayerConfig* ip6_1 = new LayerConfig("ip6_1", DENSE, (string)"pool3");
	layer_config.push_back(ip6_1);

	LayerConfig* relu6_1_ip = new LayerConfig("relu6_1_ip", RELU);
	layer_config.push_back(relu6_1_ip);

	LayerConfig* ip6_2 = new LayerConfig("ip6_2", DENSE, true);
	layer_config.push_back(ip6_2);



	LayerConfig* ip7_1 = new LayerConfig("ip7_1", DENSE, (string)"pool3");
	layer_config.push_back(ip7_1);

	LayerConfig* relu7_1_ip = new LayerConfig("relu7_1_ip", RELU);
	layer_config.push_back(relu7_1_ip);

	LayerConfig* ip7_2 = new LayerConfig("ip7_2", DENSE, true);
	layer_config.push_back(ip7_2);

	LayerConfig* ip8_1 = new LayerConfig("ip8_1", DENSE, (string)"pool3");
	layer_config.push_back(ip8_1);

	LayerConfig* relu8_1_ip = new LayerConfig("relu8_1_ip", RELU);
	layer_config.push_back(relu8_1_ip);

	LayerConfig* ip8_2 = new LayerConfig("ip8_2", DENSE, true);
	layer_config.push_back(ip8_2);





	if (Key == "")
		this->model = new Model(FilePath);
	else
		this->model = new Model(FilePath, Key);



	this->proc_layers(layer_config);


}

void CnnNet::proc_layers(vector<LayerConfig*> layer_config) {
	map<string, int> dic;//store the layer name and corresponding index
	int v_counter = 0;
	for (vector<LayerConfig*>::iterator x = layer_config.begin(); x != layer_config.end(); x++, v_counter++) {
		//if not assigned the name of last layer, take last layer as default
		if ((*x)->parent_names.size() == 0 && (*x)->type != INPUT) {
			(*x)->parent_names.push_back((*(x - 1))->name);
		}
		dic[(*x)->name] = v_counter;
	}
	//compute if exist link from A->B.graph[A][B]=1 means link from A->B exists
	vector<vector<int>> graph;
	int node_size = layer_config.size();
	for (int A = 0; A < node_size; A++) {
		vector<int> tmp;
		for (int B = 0; B < node_size; B++) {
			tmp.push_back(0);
		}
		graph.push_back(tmp);
	}



	v_counter = 0;
	for (vector<LayerConfig*>::iterator x = layer_config.begin(); x != layer_config.end(); x++, v_counter++) {
		for (vector<string>::iterator name = (*x)->parent_names.begin();
		name != (*x)->parent_names.end(); name++) {
			graph[dic[*name]][v_counter] = 1;
		}
	}


	vector<int> sorted_list;//store the index, the as order as the network forward
	for (int num = 0; num < node_size; num++) {
		//push one result to sorted_list each iteration
		for (int B = 0; B < node_size; B++) {
			//find the result where degree equals to 0, ignore it if it has already exist in sorted_list
			vector<int>::iterator ret;
			ret = std::find(sorted_list.begin(), sorted_list.end(), B);
			if (ret != sorted_list.end())
				continue;
			bool is_zero = true;
			for (int A = 0; A < node_size; A++) {
				if (graph[A][B]) {
					is_zero = false;
					break;
				}
			}
			if (is_zero) {
				sorted_list.push_back(B);
				for (int C = 0; C < node_size; C++)
					graph[B][C] = 0;
			}
		}
	}

	//read weights accoring to the name 
	for (vector<int>::iterator x = sorted_list.begin(); x != sorted_list.end(); x++) {
		//traverse sorted_list, get the layer according to the result, push to structure after handing it
		LayerConfig* single_layer_config = layer_config[*x];
		switch (single_layer_config->type) {
		case INPUT:
		{
			Input* i_la = new Input();
			i_la->w = single_layer_config->w;
			i_la->h = single_layer_config->h;
			this->structure.push_back(i_la);
			if (single_layer_config->is_output == true) {
				this->result_layer.push_back(this->structure.size() - 1);
			}
			break;
		}
		case CONV:
		{
			Conv* c_la = new Conv();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
			it != single_layer_config->parent_names.end(); it++) {
				c_la->parents.push_back(dic[*it]);// store parent_names to parents of this layer, according to string-int table
			}

			vector<int> _dim = this->model->get_dim(single_layer_config->name);
			for (int ci = 0; ci < 4; ci++)
				//c_la->dim[ci] = single_layer_config->dim[ci];//copy dim[4]
				c_la->dim[ci] = _dim[ci];//copy dim[4]
			c_la->weight = this->model->get_weight(single_layer_config->name);//copy weight
			c_la->bias = this->model->get_bias(single_layer_config->name);//copy bias
			c_la->pad = single_layer_config->pad;
			this->structure.push_back(c_la);
			if (single_layer_config->is_output == true) {
				this->result_layer.push_back(this->structure.size() - 1);
			}
			break;
		}
		case POOLING:
		{
			Pooling* p_la = new Pooling();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
			it != single_layer_config->parent_names.end(); it++) {
				p_la->parents.push_back(dic[*it]);//store parent_names to parents of this layer, according to string-int table
			}
			p_la->poolsize = single_layer_config->poolsize;
			this->structure.push_back(p_la);
			if (single_layer_config->is_output == true) {
				this->result_layer.push_back(this->structure.size() - 1);
			}
			break;
		}
		case RELU:
		{
			Relu* r_la = new Relu();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
			it != single_layer_config->parent_names.end(); it++) {
				r_la->parents.push_back(dic[*it]);//store parent_names to parents of this layer, according to string-int table
			}
			this->structure.push_back(r_la);
			if (single_layer_config->is_output == true) {
				this->result_layer.push_back(this->structure.size() - 1);
			}
			break;
		}
		case DENSE:
		{

			Dense* d_la = new Dense();
			for (vector<string>::iterator it = single_layer_config->parent_names.begin();
			it != single_layer_config->parent_names.end(); it++) {
				d_la->parents.push_back(dic[*it]);//store parent_names to parents of this layer, according to string-int table
			}
			vector<int> _dim = this->model->get_dim(single_layer_config->name);
			for (int ci = 0; ci < 4; ci++)
				//c_la->dim[ci] = single_layer_config->dim[ci];//copy dim[4]
				d_la->dim[ci] = _dim[ci];//copy dim[4]
			d_la->weight = this->model->get_fc_weight(single_layer_config->name);//copy weight
			d_la->bias = this->model->get_bias(single_layer_config->name);//copy bias
			this->structure.push_back(d_la);
			if (single_layer_config->is_output == true) {
				this->result_layer.push_back(this->structure.size() - 1);
			}
			break;
		}
		default:
			break;
		}
	}
}

void CnnNet::forward(const cv::Mat &im) {
	((Input*)this->structure[0])->load_image(im);
	for (vector<CnnLayer*>::iterator it = this->structure.begin(); it != this->structure.end(); it++) {
		(*it)->forward(this->structure);
	}
}

void CnnNet::forward(const std::string path, int mode) {
	Mat im;
	if (mode == GRAY) {
		im = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
	}
	else {
		im = imread(path);
	}
	((Input*)this->structure[0])->load_image(im);
	int count = 0;
	for (vector<CnnLayer*>::iterator it = this->structure.begin(); it != this->structure.end(); it++) {
		count++;
		(*it)->forward(this->structure);
	}
}


std::vector<int> CnnNet::argmax(const std::vector<int>& layer_nums) {
	vector<int> result;
	for (vector<int>::const_iterator num = layer_nums.begin(); num != layer_nums.end(); num++) {
		float tmp = (this->structure[*num]->result)[0].at<float>(0, 0);
		int label = 0, iter = 0;
		for (vector<Mat>::iterator it = this->structure[*num]->result.begin(); it != this->structure[*num]->result.end(); it++, iter++) {
			if ((*it).at<float>(0, 0) > tmp) {
				label = iter;
				tmp = (*it).at<float>(0, 0);
			}
		}
		result.push_back(label);
	}
	return result;
}

std::vector<int> CnnNet::argmax() {
	vector<int> result = this->argmax(this->result_layer);
	return result;
}


// return the result of our model, here the result is only the first element of each Mat, since it should be
// one value for each output for the Inner product layers 
// for other kinds of layers, there may exist several elements in each Mat

std::vector<vector<float> > CnnNet::face_info(const std::vector<int>& layer_nums) {
	vector<vector<float> > result;
	for (vector<int>::const_iterator num = layer_nums.begin(); num != layer_nums.end(); num++) {
		vector<float> result_vector;
		for (vector<Mat>::iterator it = this->structure[*num]->result.begin();
		it != this->structure[*num]->result.end(); it++) {
			result_vector.push_back((*it).at<float>(0, 0));
		}
		result.push_back(result_vector);
	}
	return result;
}

std::vector<vector<float> > CnnNet::face_info() {
	vector<vector<float> > result = this->face_info(this->result_layer);
	return result;
}
