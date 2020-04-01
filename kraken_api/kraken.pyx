# disutils: language = c++

from ckraken cimport KrakenDB, QuickFile, KrakenDBIndex, KmerScanner, \
    uint64_t, uint32_t, int64_t, build_parent_map, resolve_tree, int_vec
from libcpp.string cimport string
from libcpp.map cimport map as mapcpp
from libcpp cimport bool as boolcpp
from libcpp.vector cimport vector

cdef class PyKmerScanner:
    cdef KmerScanner *c_kmer_scanner
    cdef uint64_t *kmer
    cdef boolcpp is_ambiguous
    cdef string seq
    def __cinit__(self, string &seq):
        # saving seq is important so the string doesn't go out of scope!
        self.seq = seq
        self.c_kmer_scanner = new KmerScanner(self.seq)
        self.kmer = NULL

    def __dealloc__(self):
        del self.c_kmer_scanner

    cdef next_kmer(self):
        self.kmer = self.c_kmer_scanner.next_kmer()
        self.is_ambiguous = self.c_kmer_scanner.ambig_kmer()

    # def next_kmer(self):
    #     cdef uint64_t *kmer
    #     kmer = self.c_kmer_scanner.next_kmer()
    #     return kmer

cdef class PyKraken:
    cdef KrakenDB db
    cdef KrakenDBIndex index
    cdef mapcpp[uint32_t, uint32_t] parent_map
    cdef QuickFile db_file
    cdef QuickFile idx_file
    def __cinit__(self, str db_filename, str index_filename,
                  str nodes_filename,
                  ):
        cdef string db_path_c_string = db_filename.encode()
        cdef string index_path_c_string = index_filename.encode()
        cdef string nodes_path_c_string = nodes_filename.encode()
        self.parent_map = build_parent_map(nodes_path_c_string)
        print("parent map", self.parent_map)
        self.db_file.open_file(db_path_c_string)
        self.db_file.load_file()
        self.db = KrakenDB(self.db_file.ptr())
        self.idx_file.open_file(index_path_c_string)
        self.idx_file.load_file()
        self.index = KrakenDBIndex(self.idx_file.ptr())
        print("Index opened!")
        self.db.set_index(&self.index)
        print("Index set!")
        KmerScanner.set_k(self.db.get_k())
        print("K set to {}".format(self.db.get_k()))
        print("Done constructing.")

    cpdef classify_reads(self, reads):
        cdef string current_string
        cdef int_vec classes
        cdef int counter = 0
        classes = int_vec(len(reads))
        for string_ in reads:
            current_string = string_.encode()
            classes[counter] = self._classify_read(current_string)
            counter += 1
        return classes

    cpdef classify_read(self, str read):
        cdef string to_classify = read.encode('UTF-8')
        return self._classify_read(to_classify)

    cdef _classify_read(self, string dna_seq):
        cdef uint64_t *kmer_ptr = NULL
        cdef uint32_t taxon = 0
        cdef uint32_t hits = 0

        cdef uint64_t current_bin_key
        cdef int64_t current_min_pos = 1
        cdef int64_t current_max_pos = 0
        cdef uint32_t *val_ptr = NULL
        cdef uint64_t canon_rep
        cdef mapcpp[uint32_t, uint32_t] hit_counts

        if dna_seq.size() < self.db.get_k():
            raise ValueError("Sequence: {} is shorter than k={}".format(
                dna_seq, self.db.get_k()))

        cdef PyKmerScanner scanner = PyKmerScanner(dna_seq)
        scanner.next_kmer()
        kmer_ptr = scanner.kmer
        while kmer_ptr is not NULL:
            taxon = 0
            if not scanner.is_ambiguous:
                # use [0] to dereference as opposed to *
                canon_rep = self.db.canonical_representation(kmer_ptr[0])
                val_ptr = self.db.kmer_query(canon_rep,
                                             # &current_bin_key,
                                             # &current_min_pos,
                                             # &current_max_pos,
                    )

            # print("kmer: {}".format(kmer_ptr[0]))
            # print("val", val_ptr[0])
            taxon = val_ptr[0] if val_ptr else 0
            print("taxon! {}".format(taxon))
            if taxon:
                hit_counts[taxon] += 1
            print("hit_counts", hit_counts)

            kmer_ptr = scanner.c_kmer_scanner.next_kmer()

        return resolve_tree(hit_counts, self.parent_map)

    # def __dealloc__(self):
    #     del &self.db
    #     del &self.index

def main(str db_path):
    # TODO db_path is unused
    DB_filename = "../kraken-dbs/small_demo/database.kdb"
    Index_filename = "../kraken-dbs/small_demo/database.idx"
    nodes_filename = "../kraken-dbs/small_demo/taxonomy/nodes.dmp"

    pk = PyKraken(DB_filename, Index_filename, nodes_filename)
    read = "TACGTGAGACGGCAT"
    print("{} classified as: {}".format(read, pk.classify_read(read)))
    # print("also verified, K={}".format(pk.get_k()))
    # cdef string db_path_c_string = DB_filename
    # cdef string indx_path_c_string = Index_filename
    # cdef QuickFile db_file
    # cdef QuickFile idx_file
    #
    # print("Try opening files...")
    # db_file.open_file(DB_filename)
    # db_file.load_file()
    # cdef KrakenDB Database = KrakenDB(db_file.ptr())
    # print("Database opened!")
    # idx_file.open_file(Index_filename)
    # idx_file.load_file()
    # cdef KrakenDBIndex db_index = KrakenDBIndex(idx_file.ptr())
    # print("Index opened!")
    # Database.set_index(&db_index)
    # print("Index set!")
    # KmerScanner.set_k(Database.get_k())
    # print("K set to {}".format(Database.get_k()))
