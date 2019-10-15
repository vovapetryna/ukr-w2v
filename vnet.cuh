#pragma once
#include "array.cuh"
#include <ctime>
#include <iostream>
#include <fstream>
#define scan_diam 5
#define dict_size 26355
#define LOGFILE "weights_test_2.txt"
#define h_size 50

double avgfunction(double);
void logistic(double *, int);
void qlogistic(double *, int, double);
double sigmoid(double);

class vnet {
private:
	//back_prop variables
	double* output_deltas;
	double* hidden_deltas;
	

	//weights matrix
	//double* hidden; int hidden_num_rows; int hidden_num_cols; double* hidden_buffer;
	double* out; double* out_buffer;

public:
	//weights matrix
	double* hidden; int hidden_num_rows; int hidden_num_cols; double* hidden_buffer;
	int out_num_rows; int out_num_cols;
	//forward variables
	double* hidden_out;
	double* output;

	vnet(int, int, int);
	void load_generate();
	double* feed_forward(double *);
	double* fast_feed_forward(int *);
	double back_propagate(double*, double*);
	double fast_back_propagate(int*, int*);
	void store_weights(const char *);
	void restore_weights(const char *);
	double neuron_output(double*, double*, int);

	~vnet();
};