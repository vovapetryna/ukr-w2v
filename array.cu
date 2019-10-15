#include "array.cuh"

double * malloc_2d(int num_cols, int num_rows) {
	int size = num_cols * num_rows;
	double * data;
	cudaMallocManaged(&data, size * sizeof(double));
	return data;
}

double * calloc_2d(int num_cols, int num_rows) {
	int size = num_cols * num_rows;
	double * data;
	cudaMallocManaged(&data, size * sizeof(double));
	for (int iter = 0; iter < num_rows * num_cols; iter++) {
		data[iter] = 0.0;
	}
	return data;
}

void free_2d(double * data) {
	cudaFree(data);
}

__global__ void large_matrix_multiply_kernel(double *x, double *y, double *z, int num_cols, int num_rows, bool is_sigma) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	double sum = 0;
	if (row < num_rows) {
		for (int col = 0; col < num_cols-1; col++) {
			sum += x[row * num_cols + col] * y[col];
		}
		sum += x[row * (num_cols - 1)];
		z[row] = sum;
		if (is_sigma)
			z[row] = z[row] / 3;
	}
}

__global__ void matrix_multiply_kernel(double *x, double *y, double *z, int num_cols) {
	int row = blockIdx.x;
	int index = threadIdx.x;
	int stride = blockDim.x;
	double sum = 0;
	for (int cord = index; cord < num_cols - 1; cord += stride) {
		sum += x[row * num_cols + cord] * y[cord];
	}
	z[row * 32 + index] = sum;
}

void matrix_m(double * M1, double * M2, double * M3, double *Buffer, int num_cols, int num_rows, double(*modifier)(double)) {
	/*large_matrix_multiply_kernel << < num_rows / 32 + 1, 32 >> > (M1, M2, M3, num_cols, num_rows, false);
	cudaDeviceSynchronize();*/
	if (num_rows >= num_cols) {
		large_matrix_multiply_kernel << < num_rows / 32 + 1, 32 >> > (M1, M2, M3, num_cols, num_rows, false);
		cudaDeviceSynchronize();
	} else {
		matrix_multiply_kernel << < num_rows, 32 >> > (M1, M2, Buffer, num_cols);
		cudaDeviceSynchronize();
		for (int row = 0; row < num_rows; row++) {
			double sum = 0.0;
			for (int thr = 0; thr < 32; thr++) {
				sum += Buffer[row * 32 + thr];
			}
			sum += M1[row * num_cols + num_cols - 1];
			M3[row] = modifier(sum);
		}
	}
}

double ** _malloc_2d(int num_cols, int num_rows) {
	double ** res = (double **)malloc(num_rows * sizeof(double *));
	for (int iter = 0; iter < num_rows; iter++) {
		res[iter] = (double *)calloc(num_cols, sizeof(double));
	}
	return res;
}

void _zerro_2d(double ** res, int num_cols, int num_rows) {
	for (int iter = 0; iter < num_rows; iter++) {
		memset(res[iter], 0.0, num_cols * sizeof(double));
	}
}
