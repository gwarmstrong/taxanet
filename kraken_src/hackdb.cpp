/*
 * Copyright 2013-2019, Derrick Wood, Jennifer Lu <jlu26@jhmi.edu>
 *
 * This file is part of the Kraken taxonomic sequence classification system.
 *
 * Kraken is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Kraken is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Kraken.  If not, see <http://www.gnu.org/licenses/>.
 * *********************************
 * TODO MODIFIED BY GEORGE ARMSTRONG
 * *********************************
 */
#include "kraken_headers.hpp"
#include "krakendb.hpp"
#include "krakenutil.hpp"
#include "quickfile.hpp"
#include "seqreader.hpp"
#include <math.h>

// const size_t DEF_WORK_UNIT_SIZE = 500000;

using namespace std;
using namespace kraken;

// initializes strings that we will need for filenames
string DB_filename, Index_filename, Nodes_filename;
// initializes database to user later
KrakenDB Database;

int main(int argc, char **argv) {
    DB_filename = "../kraken-dbs/small_demo/database.kdb";
    Index_filename = "../kraken-dbs/small_demo/database.idx";
    Nodes_filename = "ijkl";

    printf("DB: %s\nIndex: %s\nNodes: %s\n", DB_filename.c_str(),
            Index_filename.c_str(),
            Nodes_filename.c_str()
            );

    QuickFile db_file;
    db_file.open_file(DB_filename);
    db_file.load_file();
    Database = KrakenDB(db_file.ptr());
    KmerScanner::set_k(Database.get_k());
    QuickFile idx_file;
    idx_file.open_file(Index_filename);
    idx_file.load_file();
    KrakenDBIndex db_index(idx_file.ptr());
    Database.set_index(&db_index);
    cerr << "Complete loading." << endl;

    uint8_t db_kmer_length = Database.get_k();
    uint64_t db_num_kmers = Database.get_key_ct();
    printf("DB k-mer size: %u\n", db_kmer_length);
    printf("DB number of k-mers: %lld\n", db_num_kmers);

    // index.at(42);
    KrakenDBIndex* index_ptr = Database.get_index();
    uint64_t node_idx = index_ptr->at(42);
    printf("DB at 42: %lld\n", node_idx);
    uint64_t node_idx2 = index_ptr->at(12);
    printf("DB at 12: %lld\n", node_idx2);

    // is key_len the same as minimizer length? probs...
    uint64_t key_len = Database.get_key_len();
    printf("DB key_len: %lld\n", key_len);

    char* ptr = Database.get_pair_ptr();
    // TODO what does this do? pair_sz...
    size_t pair_sz = Database.pair_size();
    uint64_t comp_kmer;
    comp_kmer = 0;
    printf("comp_kmer before: %lld\n", comp_kmer);
    // fetches comp_kmer to compare this kmer to.
    memcpy(&comp_kmer, ptr + 20 * pair_sz, key_len);
    printf("comp_kmer after: %lld\n", comp_kmer);
    // mocks what you would get if kmers match, from kmer_query
//    uint32_t* returned_from_query = ptr + 20 * pair_sz + key_len;

//    uint64_t idx;
//    idx = 20;
//    uint64_t at_val;
//    at_val = db_index.at(idx);
//    printf("blah: %lld\n", at_val);

    uint64_t kmer = 0;
    printf("kmer1: %lld\n", kmer);
    string dna = "ACGTAGAT";
    int total_len = 8;
    int current_pos = 0;
    int k_len = 4;
    int loaded_nt = 0;
    uint64_t kmer_mask = pow(2, 2 * k_len) - 1;
    while (current_pos < total_len){
        if (loaded_nt){
            loaded_nt--;
        }
        while (loaded_nt < k_len){
            loaded_nt++;
            kmer <<= 2;
            switch ((dna)[current_pos++]) {
                case 'A': case 'a':
                    break;
                case 'C': case 'c':
                    kmer |= 1;
                    break;
                case 'G': case 'g':
                    kmer |= 2;
                    break;
                case 'T': case 't':
                    kmer |= 3;
                    break;
            }
        }
        kmer &= kmer_mask;
        printf("kmer2: %lld\n", kmer);
    }

    // printf("%s\n", Index_filename.c_str());
    // printf("%s\n", Nodes_filename.c_str());



}
