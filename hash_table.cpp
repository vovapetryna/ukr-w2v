#include "hash_table.h"

word * search_in_dict(hash_table * dict, std::string key) {
	int key_hash = hash(key, dict);
	word * temp = dict->table[key_hash];
	while (temp != nullptr) {
		if (temp->key == key)
			return temp;
		temp = temp->next;
	}
	return nullptr;
}

void free_hash_table(hash_table * dict) {
	for (int hiter = 0; hiter < dict->size; hiter++) {
		free_list(dict->table[hiter]);
	}
	delete dict;
}

void free_list(word * head) {
	word * temp = nullptr;
	while (head != nullptr) {
		temp = head->next;
		delete head;
		head = temp;
	}
}

int hash(std::string key, hash_table * dict) {
	int iter = 0;
	for (int ch = 0; ch < key.size(); ch++) {
		iter += abs(key[ch])*dict->prime_set[ch % 7];
		iter %= dict->size;
	}
	return iter;
}

hash_table * create_dict() {
	hash_table * new_table = new hash_table;
	new_table->table = new word *[new_table->size];
	for (int iter = 0; iter < new_table->size; iter++) {
		new_table->table[iter] = nullptr;
	}
	return new_table;
}

word * add_to_dict(hash_table * dict, std::string key) {
	int key_hash = hash(key, dict);
	return add_to_list(dict, key_hash, key);
}

word * add_to_list(hash_table * dict, int key_hash, std::string key) {
	if (dict->table[key_hash] == nullptr) {
		word * new_word = new word;
		new_word->key = key;
		new_word->frequency = 1;
		new_word->next = nullptr;
		new_word->id = dict->id++;
		dict->table[key_hash] = new_word;
		return new_word;
	}
	else {
		word * lliter = dict->table[key_hash];
		word * previter = lliter;
		while (lliter != nullptr) {
			if (lliter->key == key) {
				lliter->frequency++;
				return lliter;
			}
			previter = lliter;
			lliter = lliter->next;
		}
		word * new_word = new word;
		new_word->key = key;
		new_word->frequency = 1;
		new_word->next = nullptr;
		new_word->id = dict->id++;
		previter->next = new_word;
		return new_word;
	}
	return nullptr;
}

char ctolower(char c) {
	if (c >= char(-64) && c <= char(-33)) {
		return c += ' ';
	}
	else if (c == char(-86) || c == char(-81)) {
		return c += char(16);
	}
	else if (c == char(-78)) {
		return c += char(1);
	}
	else if (c == char(-110)) {
		return char(39);
	}
	else if (c == char(-105)) {
		return char(45);
	}
	return c;
}

void stolower(std::string & s) {
	for (int iter = 0; iter < s.size(); iter++) {
		s[iter] = ctolower(s[iter]);
	}
}

void balanse(hash_table * ht) {
	int nozerro = 0;
	for (int iter = 0; iter < ht->size; iter++) {
		if (ht->table[iter] != nullptr) {
			int len = 0;
			word * temp = ht->table[iter];
			while (temp != nullptr) {
				len++;
				temp = temp->next;
			}
			std::cout << len << ' ';
			nozerro++;
		}
		else {
			std::cout << 0 << ' ';
		}
	}
	std::cout << (float)nozerro / (float)ht->size;
}

void print_table(hash_table * ht, const char * filename) {
	std::ofstream ofs;
	if (filename != "")
		ofs.open(filename);
		
	for (int iter = 0; iter < ht->size; iter++) {
		if (ht->table[iter] != nullptr) {
			word * temp = ht->table[iter];
			while (temp != nullptr) {
				if (filename != ""){
					if (temp->frequency > 0)
						ofs << temp->key << '\n';
				}
				else {
					std::cout << temp->key << '\n';
				}
					
				temp = temp->next;
			}
		}
	}

	ofs.close();
}

void fill_hash_from_file(hash_table * dict, const char * filename) {
	std::ifstream ifs;
	std::string s;
	ifs.open(filename);
	while (ifs >> s) {
		stolower(s);
		add_to_dict(dict, s);
	}
	ifs.close();
}