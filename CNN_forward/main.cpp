/*
Copyright 2015 By ihciah
https://github.com/ihciah/CNN_forward
*/
#include "CnnNet.h"
#include "utils.h"
#include <iostream>
#include <time.h>
using namespace std;

int main(){

	int t_before;
	cout << "Initializing CNN-net...";
	t_before = clock();
	CnnNet net;
	net.init("nViso_model", "");
	cout << "Done. " << clock() - t_before << "ms"<<endl;
//======
	t_before = clock();
	cout << "Net forwarding...";
	net.forward("002e2067-401b-44bc-abfb-831aeec7f303.png", GRAY);
	cout << "Done. " << clock() - t_before << "ms"<<endl;

	t_before = clock();
	vector<vector<float> > result = net.face_info();
	
	// output the result, each i means the result in each layer, while j represent the index of value in this layer
	for (int i = 0; i < result.size(); i++)
	{
		for (int j = 0; j < result[i].size(); j++)
		{
			cout << "the result of " << i << " row and " << j << " column is " << result[i][j] << endl;
		}
	}

	cout << "Done. " << clock() - t_before << "ms" << endl;
//=====


	return 0;
}
