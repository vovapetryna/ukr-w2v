#include "learner.cuh"

int main()
{	
	setlocale(LC_ALL, "Russian");
	/*double * cbow = calloc_2d(1, dict_size);
	double * outw = calloc_2d(1, dict_size);*/

	int * cbow = (int *)malloc((scan_diam - 1)* sizeof(int));
	int * outw = (int *)malloc(sizeof(int));

	vnet * net = new vnet(dict_size, h_size, dict_size);
	net->restore_weights(LOGFILE);

	hash_table * ht = create_dict();
	fill_hash_from_file(ht, "corp2.txt");
	//print_table(ht, "dict34.txt");
	//balanse(ht);
	//////////////////////////////////
	//The hash table and net ready to work
	//print_k_nearest_to_point(get_coordinates(s, net, ht), 20, net, ht);
	fast_learn(cbow, outw, net, ht, "corp2.txt");
	//////////////////////////////////
	net->store_weights(LOGFILE);
	free_hash_table(ht);

	std::cin.get();
}