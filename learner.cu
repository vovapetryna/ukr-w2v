#include "learner.cuh"

void add_word_to_onehot(std::string s, hash_table * ht, std::vector < int > & buffer, double * one_hot, bool * not_detected) {
	word * fnd = search_in_dict(ht, s);
	if (fnd == nullptr) {
		if (not_detected != NULL)
			*not_detected = true;
	}
	else {
		int cw = fnd->id;
		one_hot[cw] = 1.0;
		buffer.push_back(cw);
	}
}

void add_word_to_liniar_onehot(std::string s, hash_table * ht, int * liniar_onehot, bool * not_detected, int iter) {
	word * fnd = search_in_dict(ht, s);
	if (fnd == nullptr) {
		if (not_detected != NULL)
			*not_detected = true;
	}
	else {
		int cw = fnd->id;
		liniar_onehot[iter] = cw;
	}
}

void clear_buffer_by_change_log(std::vector < int > & buffer, double * one_hot) {
	for (int usedw = 0; usedw < buffer.size(); usedw++) {
		one_hot[buffer[usedw]] = 0.0;
	}
	buffer.erase(buffer.begin(), buffer.end());
}

void learn(double * cbow, double * outw, vnet * net, hash_table * ht, const char * filename)
{
	std::vector < int > buffer;
	std::vector < int > outbuffer;
	std::string s;
	std::ifstream ifs;

	ifs.open(filename);
	double stat_error = 0.0;
	int startpos = 1;
	int diviator = 0;
	for (int i = 0; i < 1000000; i++) {
		int read_word = scan_diam;
		bool not_detected = false;
		while (read_word > 0) {
			if (ifs >> s) {
				stolower(s);
				if (read_word != scan_diam / 2 + 1) {
					add_word_to_onehot(s, ht, buffer, cbow, &not_detected);
				}
				else {
					add_word_to_onehot(s, ht, outbuffer, outw, &not_detected);
				}
				read_word--;
			}
			else {
				ifs.clear();
				ifs.seekg(0);
				for (int skipper = 0; skipper < startpos; skipper++) {
					ifs >> s;
				}
				startpos = (startpos + 1) % scan_diam;
				std::cout << "next_evaluetion\n";
			}
		}
		if (!not_detected) {
			stat_error += net->back_propagate(cbow, outw);
			diviator++;
		}
		if (i % 25000 == 0) {
			net->store_weights(LOGFILE);
		}
		if (i % 1000 == 0) {
			std::cout << i / 1000 << "Avg Error ~ " << stat_error / diviator << '\n';
			stat_error = 0.0;
			diviator = 0;
		}
		clear_buffer_by_change_log(buffer, cbow);
		clear_buffer_by_change_log(outbuffer, outw);
	}
}

void fast_learn(int * cbow, int * outw, vnet * net, hash_table * ht, const char * filename)
{
	std::string s;
	std::ifstream ifs;
	ifs.open(filename);
	by_word_seek(5 * 500000, ifs);

	double stat_error = 0.0;
	int startpos = 1;
	int diviator = 0;
	int real_i = 0;
	for (int i = 0; i < 5000000; i++) {
		int read_word = scan_diam;
		int cbow_iter = 0;
		bool not_detected = false;
		while (read_word > 0) {
			if (ifs >> s) {
				stolower(s);
				if (read_word != scan_diam / 2 + 1) {
					add_word_to_liniar_onehot(s, ht, cbow, &not_detected, cbow_iter++);
				}
				else {
					add_word_to_liniar_onehot(s, ht, outw, &not_detected, 0);
				}
				read_word--;
			}
			else {
				ifs.clear();
				ifs.seekg(0);
				for (int skipper = 0; skipper < startpos; skipper++) {
					ifs >> s;
				}
				startpos = (startpos + 1) % scan_diam;
				std::cout << "next_evaluetion\n";
			}
		}
		if (!not_detected) {
			stat_error += net->fast_back_propagate(cbow, outw);
			diviator++;
		}

		if (i % 100000 == 0) {
			net->store_weights(LOGFILE);
		}
		if (i % 1000 == 0) {
			std::cout << i / 1000 << "Avg Error ~ " << stat_error / diviator << '\n';
			stat_error = 0.0;
			diviator = 0;
		}
	}
}


double euclidean_distance(double * V1, double * V2, int size) {
	double sum = 0.0;
	for (int iter = 0; iter < size; iter++) {
		sum += pow(V1[iter] - V2[iter], 2);
	}
	return sqrt(sum);
}

double * get_coordinates(std::string s, vnet * net, hash_table * ht) {
	//double * onehot = calloc_2d(1, dict_size);
	double * word_vec = (double *)malloc(h_size * sizeof(double));
	//std::vector < int > log_buffer;

	//add_word_to_onehot(s, ht, log_buffer, onehot, NULL);
	//net->feed_forward(onehot);

	word * fnd = search_in_dict(ht, s);
	int id = fnd->id;
	for (int i = 0; i < h_size; i++) {
		word_vec[i] = net->hidden[i * net->hidden_num_cols + id];
		//word_vec[i] = net->hidden[id * net->hidden_num_cols + i];
	}
	//memcpy(word_vec, net->hi, h_size * sizeof(double));
	//clear_buffer_by_change_log(log_buffer, onehot);
	return word_vec;
}

void print_vector(double * V, int size, bool enter) {
	for (int iter = 0; iter < size; iter++) {
		std::cout << V[iter] << ' ';
	}
	if (enter)
		std::cout << '\n';
}

void print_k_nearest_to_point(double * point, int k, vnet * net, hash_table * ht) {
	double * cur_word = (double *)malloc(h_size * sizeof(double));
	std::vector < std::pair < double, std::string > > result;
	std::ifstream ifs;
	std::string s;
	ifs.open("dict.txt");
	for (int w = 0; w < dict_size; w++) {
		ifs >> s;
		stolower(s);
		std::cout << w << '\n';
		result.push_back(std::make_pair(euclidean_distance(get_coordinates(s, net, ht), point, h_size), s));
	}
	std::sort(result.begin(), result.end());
	for (int iter = 0; iter < k; iter++) {
		std::cout << "word : " << result[iter].second << " distance : " << result[iter].first << '\n';
	}
}

double * vector_substruct(double * V1, double * V2, int size) {
	double * result = (double *)malloc(h_size * sizeof(double));
	for (int iter = 0; iter < size; iter++) {
		result[iter] = V1[iter] - V2[iter];
	}
	return result;
}

double * vector_add(double * V1, double * V2, int size) {
	double * result = (double *)malloc(h_size * sizeof(double));
	for (int iter = 0; iter < size; iter++) {
		result[iter] = V1[iter] + V2[iter];
	}
	return result;
}

void by_word_seek(int count, std::ifstream & ifs) {
	std::string s;
	for (int i = 0; i < count; i++)
		ifs >> s;
}

//double validate(int * cbow, int * outw, vnet * net, hash_table * ht, const char * filename) {
//	std::string s;
//	std::ifstream ifs;
//	ifs.open(filename);
//
//	unsigned long long correct_ans = 0;
//	int startpos = 1;
//	unsigned long long real_i = 0;
//	for (int i = 0; i < 5000000; i++) {
//		int read_word = scan_diam;
//		int cbow_iter = 0;
//		bool not_detected = false;
//		while (read_word > 0) {
//			if (ifs >> s) {
//				stolower(s);
//				if (read_word != scan_diam / 2 + 1) {
//					add_word_to_liniar_onehot(s, ht, cbow, &not_detected, cbow_iter++);
//				}
//				else {
//					add_word_to_liniar_onehot(s, ht, outw, &not_detected, 0);
//				}
//				read_word--;
//			}
//			else {
//				ifs.clear();
//				ifs.seekg(0);
//				for (int skipper = 0; skipper < startpos; skipper++) {
//					ifs >> s;
//				}
//				startpos = (startpos + 1) % scan_diam;
//				std::cout << "next_evaluetion\n";
//			}
//		}
//		if (!not_detected) {
//			//stat_error += net->fast_back_propagate(cbow, outw);
//			real_i++;
//		}
//	}
//}