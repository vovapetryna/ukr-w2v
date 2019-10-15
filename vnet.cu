#include "vnet.cuh"

vnet::vnet(int input_size, int hidden_size, int output_size)
{
	hidden_num_cols = input_size + 1;
	hidden_num_rows = hidden_size;
	out_num_cols = hidden_size + 1;
	out_num_rows = output_size;
	srand(time(NULL));
	hidden = malloc_2d(input_size + 1, hidden_size);
	out = malloc_2d(hidden_size + 1, output_size);
	hidden_out = (double *)malloc(hidden_size * sizeof(double));
	output = (double *)malloc(output_size * sizeof(double));
	output_deltas = (double *)malloc(output_size * sizeof(double));
	hidden_deltas = (double *)malloc(hidden_size * sizeof(double));
	hidden_buffer = malloc_2d(32, hidden_size);
	out_buffer = malloc_2d(32, output_size);
	load_generate();
}

vnet::~vnet() {
	free_2d(hidden);
	free_2d(out);
}

void vnet::load_generate() {
	for (int row = 0; row < hidden_num_rows; row++) {
		for (int col = 0; col < hidden_num_cols; col++) {
			hidden[row * hidden_num_cols + col] = 2 * ((double)rand() / RAND_MAX) - 1;
		}
	}
	for (int row = 0; row < out_num_rows; row++) {
		for (int col = 0; col < out_num_cols; col++) {
			out[row * out_num_cols + col] = 2 * ((double)rand() / RAND_MAX) - 1;
		}
	}
}

double vnet::neuron_output(double* weights, double* inputs, int size)
{
	double sum = 0.0;
	//normall
	for (int weight = 0; weight < size - 1; weight++) {
		sum += weights[weight] * inputs[weight];
	}
	//bias
	sum += weights[size - 1] * 1;
	return sum;
}

double* vnet::feed_forward(double* input)
{
	//hidden layer
	
	//matrix_m(hidden, input, hidden_out, hidden_buffer, hidden_num_cols, hidden_num_rows, avgfunction);
	//matrix_m(out, hidden_out, output, out_buffer, out_num_cols, out_num_rows, sigmoid);
	//cudaDeviceSynchronize();
	for (int neuron = 0; neuron < hidden_num_rows; neuron++) {
		double sum = 0.0;
		for (int weight = 0; weight < hidden_num_cols - 1; weight++) {
			sum += hidden[neuron * hidden_num_cols + weight] * input[weight];
		}
		sum += hidden[neuron * hidden_num_cols + hidden_num_cols - 1];
		hidden_out[neuron] = avgfunction(sum);
	}

	for (int neuron = 0; neuron < out_num_rows; neuron++) {
		double sum = 0.0;
		for (int weight = 0; weight < out_num_cols - 1; weight++) {
			sum += out[neuron * out_num_cols + weight] * hidden_out[weight];
		}
		sum += out[neuron * out_num_cols + out_num_cols - 1];
		output[neuron] = sum;
	}

	logistic(output, out_num_rows);
	return output;
}

double* vnet::fast_feed_forward(int* input)
{
	for (int neuron = 0; neuron < hidden_num_rows; neuron++) {
		double sum = 0.0;
		for (int iter = 0; iter < scan_diam - 1; iter++) {
			sum += hidden[neuron * hidden_num_cols + input[iter]];
		}
		sum += hidden[neuron * hidden_num_cols + hidden_num_cols - 1];
		hidden_out[neuron] = avgfunction(sum);
	}

	double glob_sum = 0.0;

	for (int neuron = 0; neuron < out_num_rows; neuron++) {
		double sum = 0.0;
		for (int weight = 0; weight < out_num_cols - 1; weight++) {
			sum += out[neuron * out_num_cols + weight] * hidden_out[weight];
		}
		sum += out[neuron * out_num_cols + out_num_cols - 1];
		output[neuron] = sum;
		glob_sum += exp(sum);
	}

	qlogistic(output, out_num_rows, glob_sum);
	return output;
}



__global__ void delta_error(double * output, double * target, double * deltas, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		deltas[i] = output[i] - target[i];
}


__global__ void error_t(double * output, double * target, double * errorr, int n) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		*errorr += output[i] - target[i];
}

__global__ void fast_weight_corection(double *deltas, double *h_out, double *w, int num_cols, int num_rows, double lambda) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0;
	if (row < num_rows) {
		for (int col = 0; col < num_cols - 1; col++) {
			w[row * num_cols + col] -= lambda * deltas[row] * h_out[col];
		}
		w[row * num_cols + num_cols - 1] -= lambda * deltas[row];
	}
}


double vnet::back_propagate(double* input, double* target)
{
	double lambda = 0.002;
	feed_forward(input);
	/*int blockSize = 32;
	int numBlocks = (out_num_rows + blockSize - 1) / blockSize;
	delta_error << < numBlocks, blockSize >> > (output, target, output_deltas, out_num_rows);
	cudaDeviceSynchronize();*/
	for (int neuron = 0; neuron < out_num_rows; neuron++) {
		output_deltas[neuron] = output[neuron] - target[neuron];
	}
	//out
	/*fast_weight_corection << < out_num_rows / 32 + 1, 32 >> > (output_deltas, hidden_out, out, out_num_cols, out_num_rows, lambda);
	cudaDeviceSynchronize();*/
	for (int neuron = 0; neuron < out_num_rows; neuron++) {
		for (int hidden_neuron = 0; hidden_neuron < hidden_num_rows; hidden_neuron++) {
			out[neuron * out_num_cols + hidden_neuron] -= lambda * output_deltas[neuron] * hidden_out[hidden_neuron];
		}
		//bias
		out[neuron * out_num_cols + hidden_num_rows] -= lambda * output_deltas[neuron];
	}
	////hidden
	for (int hidden_neuron = 0; hidden_neuron < hidden_num_rows; hidden_neuron++) {
		hidden_deltas[hidden_neuron] = 0.0;
		for (int neuron = 0; neuron < out_num_rows; neuron++) {
			hidden_deltas[hidden_neuron] += output_deltas[neuron] * out[neuron* out_num_cols + hidden_neuron];
		}
		hidden_deltas[hidden_neuron] *= 1.0 / (double)3;
		for (int weight = 0; weight < hidden_num_cols - 1; weight++) {
			hidden[hidden_neuron * hidden_num_cols + weight] -= lambda * hidden_deltas[hidden_neuron] * input[weight];
		}
		//bias
		hidden[hidden_neuron * hidden_num_cols + hidden_num_cols - 1] -= lambda * hidden_deltas[hidden_neuron];
	}
	//error
	double errsum = 0.0;
	for (int eri = 0; eri < out_num_rows; eri++) {
		//std::cout << output[eri] << ' ';
		if (target[eri] != 0.0)
			errsum -= log(output[eri]);
	}
	//std::cout << '\n';
	return errsum;
}

double vnet::fast_back_propagate(int* input, int* target){
	double lambda = 0.02;
	fast_feed_forward(input);

	for (int neuron = 0; neuron < out_num_rows; neuron++) {
		output_deltas[neuron] = output[neuron] - ((target[0] == neuron) ? 1.0 : 0.0);
	}
	for (int neuron = 0; neuron < out_num_rows; neuron++) {
		for (int hidden_neuron = 0; hidden_neuron < hidden_num_rows; hidden_neuron++) {
			out[neuron * out_num_cols + hidden_neuron] -= lambda * output_deltas[neuron] * hidden_out[hidden_neuron];
		}
		out[neuron * out_num_cols + hidden_num_rows] -= lambda * output_deltas[neuron];
	}
	for (int hidden_neuron = 0; hidden_neuron < hidden_num_rows; hidden_neuron++) {
		hidden_deltas[hidden_neuron] = 0.0;
		for (int neuron = 0; neuron < out_num_rows; neuron++) {
			hidden_deltas[hidden_neuron] += output_deltas[neuron] * out[neuron* out_num_cols + hidden_neuron];
		}
		hidden_deltas[hidden_neuron] *= 1.0 / (double)(scan_diam - 1);
		for (int index = 0; index < scan_diam - 1; index++) {
			hidden[hidden_neuron * hidden_num_cols + input[index]] -= lambda * hidden_deltas[hidden_neuron];
		}
		hidden[hidden_neuron * hidden_num_cols + hidden_num_cols - 1] -= lambda * hidden_deltas[hidden_neuron];
	}
	double errsum = 0.0;
	errsum -= log(output[target[0]]);
	return errsum;
}

double sigmoid(double x){ return 1.0 / (1.0 + exp(-x));}
double avgfunction(double x) { return x / (double)(scan_diam - 1); }

void logistic(double * M, int size) 
{
	double glob_sum = 0.0;
	for (int neuron = 0; neuron < size; neuron++) {
		glob_sum += exp(M[neuron]);
	}
	for (int neuron = 0; neuron < size; neuron++) {
		M[neuron] = exp(M[neuron]) / glob_sum;
	}
}

void qlogistic(double * M, int size, double glob_sum) {
	for (int neuron = 0; neuron < size; neuron++) {
		M[neuron] = exp(M[neuron]) / glob_sum;
	}
}

void vnet::store_weights(const char * filename) {
	std::ofstream ofs;
	ofs.open(filename);
	ofs.precision(15);
	for (int hidden_n = 0; hidden_n < hidden_num_rows; hidden_n++) {
		for (int hidden_w = 0; hidden_w < hidden_num_cols; hidden_w++) {
			ofs << hidden[hidden_n * hidden_num_cols + hidden_w] << ' ';
		}
	}
	for (int out_n = 0; out_n < out_num_rows; out_n++) {
		for (int out_w = 0; out_w < out_num_cols; out_w++) {
			ofs << out[out_n * out_num_cols + out_w] << ' ';
		}
	}
	ofs.close();
}

void vnet::restore_weights(const char * filename) {
	std::ifstream ifs;
	ifs.open(filename);
	double weight_num = 0.0;
	for (int hidden_n = 0; hidden_n < hidden_num_rows; hidden_n++) {
		for (int hidden_w = 0; hidden_w < hidden_num_cols; hidden_w++) {
			ifs >> weight_num;
			hidden[hidden_n * hidden_num_cols + hidden_w] = weight_num;
		}
	}
	for (int out_n = 0; out_n < out_num_rows; out_n++) {
		for (int out_w = 0; out_w < out_num_cols; out_w++) {
			ifs >> weight_num;
			out[out_n * out_num_cols + out_w] = weight_num;
		}
	}
	ifs.close();
}
