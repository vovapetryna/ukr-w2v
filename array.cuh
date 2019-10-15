#pragma once 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

double * malloc_2d(int, int);
double * calloc_2d(int, int);
void free_2d(double *);
void matrix_m(double *, double *, double *, double *, int, int, double(*)(double));
double ** _malloc_2d(int, int);
void _zerro_2d(double **, int, int);