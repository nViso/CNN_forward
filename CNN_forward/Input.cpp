/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include"Input.h"
#include<iostream>

using namespace std;
using namespace cv;

void Input::forward(const std::vector<CnnLayer*>& structure) {}
void Input::load_image(Mat im){
	this->result.clear();
	if (this->w > 0 && this->h > 0)
		resize(im, im, Size(this->w, this->h));
	vector<Mat> channels(im.channels());
	split(im, channels);
	// normalize to [0,1] scale
	float beta = 1.0f / 255;
	for (int i = 0; i < im.channels(); i++){
		Mat tmp;
		channels[i].convertTo(tmp, CV_32F);
		this->result.push_back(tmp * beta);
		Mat re = tmp * beta;
	}

}
