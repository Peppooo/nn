#define _USE_MATH_DEFINES
#include <iostream>
#include <Eigen/Eigen>
#include <random>
#include "nn.h"
#include "mnist.h"


using namespace std;

mt19937 global_rng;

int main() {
	Sequential model = {{
		new Dense(784,256,true),
		new ReLU(),
		new Dropout(0.35),

		new Dense(256,128,true),
		new ReLU(),
		new Dropout(0.2),

		new Dense(128,10,true),
		new Softmax()
	}};

	vector<Eigen::Vector<double,784>> train_X;
	vector<Eigen::Vector<double,10>>  train_Y; vector<uint8_t> train_Y_labels;

	vector<Eigen::Vector<double,784>> test_X;
	vector<Eigen::Vector<double,10>>  test_Y; vector<uint8_t> test_Y_labels;

	mnist::read_images("C:\\Users\\pietr\\source\\repos\\neural-net\\training\\digits\\train-images.idx3-ubyte",train_X);
	mnist::read_labels("C:\\Users\\pietr\\source\\repos\\neural-net\\training\\digits\\train-labels.idx1-ubyte",&train_Y,train_Y_labels);
	
	mnist::read_images("C:\\Users\\pietr\\source\\repos\\neural-net\\training\\digits\\test-images.idx3-ubyte",test_X);
	mnist::read_labels("C:\\Users\\pietr\\source\\repos\\neural-net\\training\\digits\\test-labels.idx1-ubyte",&test_Y,test_Y_labels);

	int epochs = 128;
	int batch_size = 64;
	double lr = 0.03;

	vector<size_t> indecies(train_X.size(),0);
	iota(indecies.begin(),indecies.end(),0ull);

	for(int epoch = 1; epoch <= epochs; epoch++) {

		double cost = 0;

		shuffle(indecies.begin(),indecies.end(),global_rng);
		for(int _i = 0; _i < train_X.size(); _i += 1) {
			auto i = indecies[_i];
			auto pred = model.grad_forward(train_X[i]);
			auto err = train_Y[i].array() / (-pred.array());
			cost += -(pred.array().log()*train_Y[i].array()).sum();
			model.compute_gradients(err);

			if(_i % batch_size == 0 && _i != 0) {
				model.apply_gradients(lr);
			}
		}
		model.apply_gradients(lr);

		// compute accuracy

		int accuracy_tests = 10000;
		int test_correct = 0,train_correct = 0;

		for(int i = 0; i < accuracy_tests; i++) {
			auto pred = model.forward(test_X[i]);
			uint8_t pred_label = 0;
			pred.maxCoeff(&pred_label);
			test_correct += (pred_label == test_Y_labels[i]);
		}

		for(int i = 0; i < accuracy_tests; i++) {
			auto pred = model.forward(train_X[i]);
			uint8_t pred_label = 0;
			pred.maxCoeff(&pred_label);
			train_correct += (pred_label == train_Y_labels[i]);
		}
		cost /= train_X.size();
		cout << "epoch " << epoch << " cost " << cost << " test accuracy " << test_correct / (double)accuracy_tests << " train accuracy " << train_correct / (double)accuracy_tests << endl;
		if((test_correct / (double)accuracy_tests) > 0.97)
			break;

	}
}