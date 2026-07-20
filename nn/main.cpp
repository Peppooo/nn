#include <iostream>
#include <math.h>
#include <assert.h>
#include <vector>
#include <Eigen/Eigen>

using namespace std;

typedef double(*xy_func)(double);

double _relu(double x) {
	return std::max(x,0.0);
}
double d_relu(double x) {
	return x > 0 ? 1 : 0;
}
#define K_LEAKY 0.001 // leaky relu coefficient
double _lrelu(double x) {
	return std::max(x,K_LEAKY * x);
}
double d_lrelu(double x) {
	return x > 0 ? 1 : K_LEAKY;
}
double _linear(double x) {
	return x;
}
double d_linear(double x) {
	return 1;
}
double d_tanh(double x) {
	return (1 - std::pow(std::tanh(x),2));
}

double _sign(double x) {
	return (!signbit(x)) * 2.0 - 1.0;
}

class deriv_func {
public:
	xy_func f,deriv;
	__forceinline double operator()(const double x) {
		return f(x);
	}
	__forceinline Eigen::VectorXd operator()(const Eigen::VectorXd& x) {
		return x.unaryExpr(f);
	}
};

namespace act {
	const deriv_func relu{_relu,d_relu};
	const deriv_func linear{_linear,d_linear};
	const deriv_func tanh{std::tanh,d_tanh};
	const deriv_func leaky_relu{_lrelu,d_lrelu};
}


class Dense {
public:
	Eigen::VectorXd linear;
	Eigen::VectorXd activation;
	Eigen::VectorXd in_activation;

	Eigen::MatrixXd weights;
	Eigen::VectorXd biases;

	Eigen::MatrixXd w_grad_sum;
	Eigen::VectorXd b_grad_sum;

	size_t grad_samples;

	deriv_func activ_f;

	Dense(size_t in_dim,size_t out_dim,deriv_func activation = act::linear):activ_f(activation) {
		weights = Eigen::MatrixXd(out_dim,in_dim);
		biases = Eigen::VectorXd(out_dim);
		weights.setRandom();
		biases.setRandom();
		w_grad_sum = weights;
		b_grad_sum = biases;
		reset_gradients();
	}
	inline Eigen::VectorXd forward(const Eigen::VectorXd& in) {
		return activ_f(weights * in + biases);
	}
	inline void grad_forward(const Eigen::VectorXd& in) {
		in_activation = in;
		linear = weights * in + biases;
		activation = activ_f(linear);
	}
	inline Eigen::VectorXd operator()(const Eigen::VectorXd& in,bool grad = false) {
		if(!grad) {
			return forward(in);
		}
		grad_forward(in);
		return activation;
	}
	inline void reset_gradients() {
		w_grad_sum.setZero();
		b_grad_sum.setZero();
		grad_samples = 0;
	}
	void apply_gradients(double lr) {
		if(grad_samples > 0) {
			weights -= lr * (w_grad_sum/grad_samples);
			biases -= lr * (b_grad_sum/grad_samples);
			reset_gradients();
		}
	}
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) { // returns the partial derivative of the cost in respect to input vector
		assert(flow_back.size() == biases.size()); // "flow_back partial gradients vector dim must match out_dim";
		Eigen::VectorXd pd_act_lin = linear.unaryExpr(activ_f.deriv);
		Eigen::VectorXd pd_fb_lin = pd_act_lin.array() * flow_back.array();
		
		w_grad_sum += pd_fb_lin * in_activation.transpose();
		b_grad_sum += pd_fb_lin;

		grad_samples++;

		Eigen::VectorXd ret_flow_back;
		ret_flow_back.resize(weights.rows());

		for(int i = 0; i < weights.cols(); i++) {
			ret_flow_back(i) = (pd_fb_lin.array() * weights.col(i).array()).sum();
		}

		return ret_flow_back;
	}
};

int main() {
	auto layer0 = Dense(2,2,act::linear);
	auto layer1 = Dense(2,2,act::linear);

	// simple test dataset
	vector<Eigen::Vector2d> x = {{1,0},{-1,1},{0,1},{0,0},{0.1,0.5}},y = {{0,1},{1,-1},{1,0},{0,0},{0.5,0.1}};
	
	for(int epoch = 0; epoch < 10000; epoch++) {
		double cost = 0;

		for(int i = 0; i < x.size(); i++) {
			auto out = layer1(layer0(x[i],true),true); // run for compute_gradients
			Eigen::VectorXd error = 2*(out-y[i]); // mean squared error derivative;
			Eigen::VectorXd _err = (out-y[i]).array().pow(2); // mse
			cost += _err.sum();
			layer0.compute_gradients(layer1.compute_gradients(error));
		}
		cost = cost / x.size();
		if(epoch % 100 == 0) {
			cout << "epoch=" << epoch << " cost=" << cost << endl;
		}
		layer0.apply_gradients(0.3);
		layer1.apply_gradients(0.3);
	}

	cout << "layer0 weights matrix: " << endl << layer0.weights << endl;
	cout << "layer0 biases vector: " << endl << layer0.biases << endl;
	cout << "output: " << endl << layer1(layer0(x[1])) << endl;

}