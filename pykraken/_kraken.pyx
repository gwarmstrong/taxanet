# disutils: language = c++

from _ckraken cimport KrakenDB, QuickFile, KrakenDBIndex, KmerScanner, \
    uint64_t, uint32_t, int64_t, build_parent_map, resolve_tree, int_vec
from libcpp.string cimport string
from libcpp.map cimport map as mapcpp
from libcpp cimport bool as boolcpp

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
        self.db_file.open_file(db_path_c_string)
        self.db_file.load_file()
        self.db = KrakenDB(self.db_file.ptr())
        self.idx_file.open_file(index_path_c_string)
        self.idx_file.load_file()
        self.index = KrakenDBIndex(self.idx_file.ptr())
        self.db.set_index(&self.index)
        KmerScanner.set_k(self.db.get_k())

    @property
    def key_count(self):
        return self._key_count()

    @property
    def k(self):
        return self.db.get_k()

    cdef _key_count(self):
        cdef uint64_t count = self.db.get_key_ct()
        return count

    cpdef taxa_counts(self):
        cdef uint32_t taxa
        cdef mapcpp[uint32_t, uint32_t] count_map
        for i in range(self._key_count()):
            taxa = self.db.taxa_at(i)[0]
            # TODO make sure this won't result in segfault
            count_map[taxa] += 1
        return count_map

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
                                             &current_bin_key,
                                             &current_min_pos,
                                             &current_max_pos,
                    )

            taxon = val_ptr[0] if val_ptr else 0
            if taxon:
                hit_counts[taxon] += 1

            kmer_ptr = scanner.c_kmer_scanner.next_kmer()

        return resolve_tree(hit_counts, self.parent_map)
