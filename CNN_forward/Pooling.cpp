/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include"Pooling.h"
#include<iostream>
#include<math.h>


using namespace std;
using namespace cv;

void Pooling::forward(const std::vector<CnnLayer*>& structure){

	this->result.clear();
	int p_size = this->poolsize;
	vector<Mat> ret;
	vector<Mat> input = structure[this->parents[0]]->result;
	for (vector<Mat>::const_iterator it = input.begin(); it != input.end(); it++){
		//Mat is too slow for random access, transfer to vector
		vector<vector<float>> vec_mat;
		MatConstIterator_<float> it_r = it->begin<float>();
		for (int A = 0; A < it->rows; A++){
			vector<float> tmp_mat_row;
			for (int B = 0; B < it->cols; B++,it_r++){
				tmp_mat_row.push_back(*it_r);
			}
			vec_mat.push_back(tmp_mat_row);
		}
		//traverse vec<vec<float>> to get the pooling
		int new_rows = (it->rows + p_size - 1) / p_size;
		int new_cols = (it->cols + p_size - 1) / p_size;
		Mat tmp_w(new_rows, new_cols, it->type());
		MatIterator_<float> it_w = tmp_w.begin<float>();
		for (int A = 0; A < new_rows; A++){
			for (int B = 0; B < new_cols; B++,it_w++){
				//in order to get the value of [A][B], we need traverse [A*p_size-(A+1)*p_size][B~]
				int A_start = A*p_size, A_end = A_start + p_size;
				int B_start = B*p_size, B_end = B_start + p_size;
				float tp = vec_mat[A_start][B_start];
				for (int A_i = A_start; A_i < A_end; A_i++){
					for (int B_i = B_start; B_i < B_end; B_i++){
						// if the number of rows is odd, then the last pooling block may exceed the range,
						//check it before compare
						if(A_i<=(vec_mat.size()-1)&&B_i<=vec_mat[A_i].size()-1)
						tp = std::max(tp, vec_mat[A_i][B_i]);
					}
				}
				*it_w = tp;
			}
		}
		ret.push_back(tmp_w);
	}
	this->result=ret;

}