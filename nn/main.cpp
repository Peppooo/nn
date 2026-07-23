#define _USE_MATH_DEFINES
#include <iostream>
#include <Eigen/Eigen>
#include <random>
#include <SDL3/SDL.h>
#include "nn.h"
#include "mnist.h"


using namespace std;

mt19937 global_rng;

int main() {
	Sequential model = {{
		new Dense(784,256,0.01,true),
		new ReLU(),
		new Dropout(0.4),

		new Dense(256,128,0.01,true),
		new ReLU(),
		new Dropout(0.3),

		new Dense(128,10,0.01,true),
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

	int epochs = 64;
	int batch_size = 64;
	double lr = 0.1;

	vector<size_t> indecies(train_X.size(),0);
	iota(indecies.begin(),indecies.end(),0ull);

	for(int epoch = 1; epoch <= epochs; epoch++) {
		double cost = 0;

		shuffle(indecies.begin(),indecies.end(),global_rng);
		for(int _i = 0; _i < train_X.size(); _i += 1) {
			auto i = indecies[_i];
			auto pred = model.grad_forward(train_X[i]);
			auto err = train_Y[i].array() / (-pred.array()); // cross entropy error partial derivative
			cost += -(pred.array().log()*train_Y[i].array()).sum(); // cross entropy error value
			model.compute_gradients(err);

			if(_i % batch_size == 0 && _i != 0) {
				model.apply_gradients(lr);
			}
		}
		model.apply_gradients(lr); // apply possible remaining gradients

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
		double weights_sq_sum = ((Dense*)model.layers[0])->weights.array().pow(2).sum() +
			((Dense*)model.layers[3])->weights.array().pow(2).sum() +
			((Dense*)model.layers[6])->weights.array().pow(2).sum();
		cout << "epoch " << epoch << "weights sqsum: " << weights_sq_sum << " cost " << cost << "accuracy: test " << test_correct / (double)accuracy_tests << " train " << train_correct / (double)accuracy_tests << endl;
		if((test_correct / (double)accuracy_tests) >= 0.977)
			break; 

	}
	SDL_Window* window;
	SDL_Renderer* renderer;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer("test model",560,560,0,&window,&renderer);
	SDL_SetRenderLogicalPresentation(renderer,28,28,SDL_LOGICAL_PRESENTATION_INTEGER_SCALE);

	Eigen::Vector<double,784> buffer;

	SDL_Event e;
	float mousex = 0,mousey = 0;
	while(1) {
		SDL_GetMouseState(&mousex,&mousey);
		mousex /= 20; mousey /= 20;
		mousex = SDL_clamp(floor(mousex),0,27);
		mousey = SDL_clamp(floor(mousey),0,27);
		while(SDL_PollEvent(&e)) {
			if(e.button.button == SDL_BUTTON_LEFT) {
				buffer[mousey * 28 + mousex] += 0.4;
				buffer[mousey * 28 + mousex] = SDL_clamp(buffer[mousey * 28 + mousex],0,1);
			}
			if(e.type == SDL_EVENT_KEY_DOWN) {
				if(e.key.key == SDLK_C) {
					buffer.setZero();
				}
			}
		}
		SDL_RenderClear(renderer);
		for(int i = 0; i < buffer.size(); i++) {
			Uint8 c = buffer(i) * 255;
			SDL_SetRenderDrawColor(renderer,c,c,c,255);
			SDL_RenderPoint(renderer,i % 28,i / 28);
		}
		SDL_RenderPresent(renderer);
		int pred_label = 0;
		model.forward(buffer).maxCoeff(&pred_label);
		char tex_buff[3];
		_itoa_s(pred_label,tex_buff,10);
		SDL_SetWindowTitle(window,tex_buff);
	}
}