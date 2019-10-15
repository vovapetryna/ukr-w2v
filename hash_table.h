#pragma once
#include <iostream>
#include <string>
#include <math.h>
#include <fstream>

struct word
{
	int id = 0;
	int frequency;
	std::string key;
	struct word * next;
};

struct hash_table
{
	int id = 0;
	int prime_set[7] = { 701, 691, 499, 397, 797, 293, 401 };
	int size = 15013;
	word ** table;
};

hash_table * create_dict();
word * add_to_dict(hash_table *, std::string);
word * add_to_list(hash_table *, int, std::string);
word * search_in_dict(hash_table *, std::string);
void free_hash_table(hash_table *);
void free_list(word *);
int hash(std::string, hash_table *);
char ctolower(char);
void stolower(std::string &);
void balanse(hash_table *);
void print_table(hash_table *, const char *);
void fill_hash_from_file(hash_table *, const char *);