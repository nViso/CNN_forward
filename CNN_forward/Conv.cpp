/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include"Conv.h"
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace std;
using namespace cv;
using namespace tbb;

//TODO: 1.TBB parallel optimation
//      2.change iterator to pointer, no much difference in efficiency




//void Conv::forward(const std::vector<CnnLayer*>& structure){
//  // no multi-thread version
//	//this->weight need store values after transpose and flip
//  this->result.clear();
//	vector<Mat> input = structure[this->parents[0]]->result;
//	for (int output_num = 0; output_num < this->dim[0]; output_num++){
//		Mat tmp(input[0].rows - (this->dim[2]) + 1, input[0].cols - (this->dim[3]) + 1, CV_32F);
//		tmp.setTo(this->bias.at<float>(output_num,0));
//		Range r_a(this->dim[2] / 2, this->dim[2] / 2 + tmp.rows);
//		Range r_b(this->dim[3] / 2, this->dim[3] / 2 + tmp.cols);
//		for (int input_num = 0; input_num < this->dim[1]; input_num++){
//			Mat tmp_conv;
//			filter2D(input[input_num], tmp_conv, input[input_num].depth(), this->weight[output_num][input_num]);
//			Mat a= tmp_conv(r_a,r_b);
//			cv::add(tmp, a, tmp);
//		}
//		this->result.push_back(tmp);
//		
//	}
//}

class Parallel_conv : public cv::ParallelLoopBody
{

private:
	const vector<Mat>& inImages;
	const Mat& biasMat;
	const vector<vector<Mat>>& weight;
	const int* dim;
	const int& pad;
	vector<Mat>& outImages;
	vector<Mat>& inImages_pad;

public:
	Parallel_conv(const vector<Mat>& inputImgage, const Mat& bias, const vector<vector<Mat>>& weight, vector<Mat>& outImage, const int* dim, const int& pad, vector<Mat>& inImages_pad)
		: inImages(inputImgage), outImages(outImage), biasMat(bias), weight(weight), dim(dim),pad(pad), inImages_pad(inImages_pad){
		// check if pad>0, if so, add the padding parameter
		if (pad > 0)
		{
			for (int index = 0; index < inImages.size(); index++)
			{
				Mat temp_input = inImages[index];
				Mat top = Mat(pad, temp_input.cols, CV_32F, 0.0);
				Mat bot = Mat(pad, temp_input.cols, CV_32F, 0.0);
				top.push_back(temp_input);
				top.push_back(bot);
				Mat left = Mat(top.rows, pad, CV_32F, 0.0);
				Mat right = Mat(top.rows, pad, CV_32F, 0.0);
				hconcat(left, top, top);
				hconcat(top, right, top);
				inImages_pad.push_back(top);
			}
		}
		else {
			inImages_pad = inImages;
		}
	}

	virtual void operator()(const cv::Range& range) const
	{

		for (int output_num = range.start; output_num < range.end; output_num++)
		{
			Mat tmp(inImages_pad[0].rows - (this->dim[2]) + 1, inImages_pad[0].cols - (this->dim[3]) + 1, CV_32F);
			tmp.setTo(biasMat.at<float>(output_num, 0));
			Range r_a(this->dim[2] / 2, this->dim[2] / 2 + tmp.rows);
			Range r_b(this->dim[3] / 2, this->dim[3] / 2 + tmp.cols);
			for (int input_num = 0; input_num < this->dim[1]; input_num++){
				Mat tmp_conv;
				filter2D(inImages_pad[input_num], tmp_conv, inImages_pad[input_num].depth(), weight[output_num][input_num]);
				cv::add(tmp, tmp_conv(r_a, r_b), tmp);
			}
			outImages[output_num] = tmp;
			tmp.release();
		}
	}
};


void Conv::forward(const std::vector<CnnLayer*>& structure){
	//multi-thread optimation
	this->result.clear();
	this->result.resize(this->dim[0]);//allocate space, able to random access parallelly
	vector<Mat> inImages_with_pad;
	cv::parallel_for_(cv::Range(0, this->dim[0]), Parallel_conv(structure[this->parents[0]]->result, this->bias, this->weight, this->result, this->dim, this->pad, inImages_with_pad));


}