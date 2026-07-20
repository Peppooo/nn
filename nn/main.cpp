#define _USE_MATH_DEFINES
#include <iostream>
#include <Eigen/Eigen>
#include <random>
#include "nn.h"
#include "mnist.h"


using namespace std;

mt19937 global_rng;

int main() {
	Sequential model = {{ // 784 - 128 - 64 - 10
		new Dense(784,32,act::relu),
		new Dense(32, 32,act::relu),
		new Dense( 32, 10,act::linear),
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

	int epochs = 1000;
	int batch_size = 32;
	double lr = 0.03;

	vector<size_t> indecies(train_X.size(),0);
	iota(indecies.begin(),indecies.end(),0ull);

	for(int epoch = 0; epoch < epochs; epoch++) {
		shuffle(indecies.begin(),indecies.end(),global_rng);
		for(int _i = 0; _i < train_X.size(); _i += 1) {
			auto i = indecies[_i];
			auto pred = model.grad_forward(train_X[i]);
			auto err = train_Y[i].array() / (-1*pred.array());
			
			model.compute_gradients(err);

			if(_i % batch_size == 0) {
				model.apply_gradients(lr);
			}
		}
		model.apply_gradients(lr);

		// compute accuracy

		int accuracy_tests = 1000;
		int correct = 0;

		for(int i = 0; i < accuracy_tests; i++) {
			auto pred = model.forward(test_X[i]);
			uint8_t pred_label = 0;
			pred.maxCoeff(&pred_label);
			correct += (pred_label == test_Y_labels[i]);
		}

		cout << "epoch " << epoch << " accuracy " << correct/(double)accuracy_tests << endl;
	}
}