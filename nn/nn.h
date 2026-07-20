#pragma once
#include <math.h>
#include <assert.h>
#include <vector>
#include <Eigen/Eigen>

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

class Pass {
public:
	virtual Eigen::VectorXd forward(const Eigen::VectorXd& in) = 0;
	virtual Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) = 0;
	virtual void reset_gradients() = 0;
	virtual void apply_gradients(double lr) = 0;
	virtual Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back,bool compute_flow_back = true) = 0;  // returns the partial derivative of the cost in respect to input vector
};

class Softmax : public Pass {
	Eigen::VectorXd activation;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		Eigen::ArrayXd ex = (in.array() - in.maxCoeff()).exp(); // avoids overflow without changing output since
		return (ex / ex.sum());;
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		activation = forward(in);
		return activation;
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back,bool compute_flow_back = false) override {
		double dot = activation.dot(flow_back);
		return activation.array() * (flow_back.array() - dot);
	}
};

class Dense : public Pass {
public:
	Eigen::VectorXd linear;
	Eigen::VectorXd activation;
	Eigen::VectorXd in_activation;

	Eigen::MatrixXd weights;
	Eigen::VectorXd biases;

	Eigen::MatrixXd w_grad_mean;
	Eigen::VectorXd b_grad_mean;

	size_t grad_samples;

	deriv_func activ_f;

	Dense(size_t in_dim,size_t out_dim,deriv_func activation = act::linear):activ_f(activation) {
		weights = Eigen::MatrixXd(out_dim,in_dim);
		biases = Eigen::VectorXd(out_dim);
		weights.setRandom();
		biases.setRandom();
		w_grad_mean = weights;
		b_grad_mean = biases;
		reset_gradients();
	}
	inline Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		return activ_f(weights * in + biases);
	}
	inline Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		in_activation = in;
		linear = weights * in + biases;
		activation = activ_f(linear);
		return activation;
	}
	inline void reset_gradients() override {
		w_grad_mean.setZero();
		b_grad_mean.setZero();
		grad_samples = 0;
	}
	void apply_gradients(double lr) override {
		if(grad_samples > 0) {
			weights -= lr * w_grad_mean;
			biases -= lr * b_grad_mean;
			reset_gradients();
		}
	}
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back,bool compute_flow_back = true) override { // returns the partial derivative of the cost in respect to input vector
		assert(flow_back.size() == biases.size()); // "flow_back partial gradients vector dim must match out_dim";
		Eigen::VectorXd pd_act_lin = linear.unaryExpr(activ_f.deriv);
		Eigen::VectorXd pd_fb_lin = pd_act_lin.array() * flow_back.array();

		
		grad_samples++;

		double alpha = 1.0 / grad_samples;

		w_grad_mean += alpha * (pd_fb_lin * in_activation.transpose() - w_grad_mean);
		b_grad_mean += alpha * (pd_fb_lin - b_grad_mean);


		Eigen::VectorXd ret_flow_back;
		ret_flow_back.resize(weights.cols());

		if(compute_flow_back) {
			for(int i = 0; i < weights.cols(); i++) {
				ret_flow_back(i) = (pd_fb_lin.array() * weights.col(i).array()).sum();
			}
		}

		return ret_flow_back;
	}
};

class Sequential {
public:
	std::vector<Pass*> layers;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) {
		Eigen::VectorXd pass = in;
		for(auto l : layers) {
			pass = l->forward(pass);
		}
		return pass;
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) {
		Eigen::VectorXd pass = in;
		for(auto l : layers) {
			pass = l->grad_forward(pass);
		}
		return pass;
	} 
	void compute_gradients(const Eigen::VectorXd& pd_cost_act) {
		Eigen::VectorXd flow_back = pd_cost_act;
		for(int i = layers.size() - 1; i >= 0; i--) {
			flow_back = layers[i]->compute_gradients(flow_back,i!=0);
		}
	}
	void reset_gradients() {
		for(auto l : layers) {
			l->reset_gradients();
		}
	}
	void apply_gradients(double lr) {
		for(auto l : layers) {
			l->apply_gradients(lr);
		}
	}
};