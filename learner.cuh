#pragma once
#include "vnet.cuh"
#include "hash_table.h"
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <array>
#include <thread>

void add_word_to_onehot(std::string, hash_table *, std::vector < int > &, double *, bool *);
void add_word_to_liniar_onehot(std::string, hash_table *,int * , bool *, int);
void clear_buffer_by_change_log(std::vector < int > &, double *);
void learn(double *, double *, vnet *, hash_table *, const char *);
void fast_learn(int *, int *, vnet *, hash_table *, const char *);
double euclidean_distance(double *, double *, int);
double * get_coordinates(std::string, vnet *, hash_table *);
void print_vector(double *, int, bool = true);
void print_k_nearest_to_point(double *, int, vnet *, hash_table *);
double * vector_substruct(double *, double *, int);
double * vector_add(double *, double *, int);
void by_word_seek(int, std::ifstream &);
//double validate(int *, int *, vnet *, hash_table *, const char *);