from libcpp.string cimport string
from libcpp cimport bool
from libcpp.map cimport map as mapcpp
from libcpp.vector cimport vector

# cdef extern from "src/kraken_headers.hpp":
#     pass
#
ctypedef vector[int] int_vec

cdef extern from "<stdint.h>" nogil:

    # 7.18.1 Integer types
    ctypedef   unsigned char  uint8_t
    ctypedef unsigned int   uint32_t
    ctypedef   unsigned long  long uint64_t
    ctypedef   signed long  int64_t

# cdef extern from "src/krakenutil.cpp":
#     pass
#
cdef extern from "seqreader.hpp" namespace "kraken":
    ctypedef struct DNASequence:
        string id
        string header_line
        string seq
        string quals

    cdef cppclass DNASequenceReader:
        DNASequence next_sequence()
        bool is_valid()

    cdef cppclass FastaReader(DNASequenceReader):
        FastaReader(string filename)

    cdef cppclass FastqReader(DNASequenceReader):
        FastqReader(string filename)



cdef extern from "quickfile.hpp" namespace "kraken":
    cdef cppclass QuickFile:
        # Null constructor
        QuickFile()
        QuickFile(string filename)
        QuickFile(string filename, string mode, size_t size)

        void open_file(string filename);
        void open_file(string filename, string mode, size_t size);
        char *ptr();
        size_t size();
        void load_file();
        void sync_file();
        void close_file();


cdef extern from "krakendb.hpp" namespace "kraken":
    cdef cppclass KrakenDBIndex:
        KrakenDBIndex()
        # ptr points to mmap'ed existing file opened in read or read/write mode
        KrakenDBIndex(char *ptr)

        uint8_t index_type();
        uint8_t indexed_nt();
        uint64_t *get_array();
        uint64_t at(uint64_t idx);

    cdef cppclass KrakenDB:

        # Null constructor
        KrakenDB()

        # ptr points to start of mmap'ed DB in read or read/write mode
        KrakenDB(char *ptr)

        char *get_ptr()
        char *get_pair_ptr()
        KrakenDBIndex *get_index()

        char *get_ptr();            # Return the file pointer
        char *get_pair_ptr();       # Return pointer to start of pairs
        KrakenDBIndex *get_index(); # Return ptr to assoc'd index obj
        uint8_t get_k();            # how many nt are in each key?
        uint64_t get_key_bits();    # how many bits are in each key?
        uint64_t get_key_len();     # how many bytes does each key occupy?
        uint64_t get_val_len();     # how many bytes does each value occupy?
        uint64_t get_key_ct();      # how many key/value pairs are there?
        uint64_t pair_size();       # how many bytes does each pair occupy?

        size_t header_size();  # Jellyfish uses variable header sizes
        uint32_t *kmer_query(uint64_t kmer);  # return ptr to pair w/ kmer

        # perform search over last range to speed up queries
        uint32_t *kmer_query(uint64_t kmer, uint64_t *last_bin_key,
                             int64_t *min_pos, int64_t *max_pos,
                             bool retry_on_failure);

        uint32_t *kmer_query(uint64_t kmer, uint64_t *last_bin_key,
                             int64_t *min_pos, int64_t *max_pos,
                             );

        # return "bin key" for kmer, based on index
        # If idx_nt not specified, use index's value
        uint64_t bin_key(uint64_t kmer, uint64_t idx_nt);
        uint64_t bin_key(uint64_t kmer);

        # Code from Jellyfish, rev. comp. of a k-mer with n nt.
        # If n is not specified, use k in DB, otherwise use first n nt in kmer
        uint64_t reverse_complement(uint64_t kmer, uint8_t n);
        uint64_t reverse_complement(uint64_t kmer);

        # Return lexicographically smallest of kmer/revcom(kmer)
        # If n is not specified, use k in DB, otherwise use first n nt in kmer
        uint64_t canonical_representation(uint64_t kmer, uint8_t n);
        uint64_t canonical_representation(uint64_t kmer);

        void make_index(string index_filename, uint8_t nt);

        void set_index(KrakenDBIndex *i_ptr);

cdef extern from "krakenutil.hpp" namespace "kraken":
    cdef cppclass KmerScanner:
        KmerScanner(string &seq);
        KmerScanner(string &seq, size_t start, size_t finish);
        uint64_t *next_kmer();  # NULL when seq exhausted
        bool ambig_kmer();  # does last returned kmer have non-ACGT?

        @staticmethod
        uint8_t get_k();

        @staticmethod
        void set_k(uint8_t n);

    mapcpp[uint32_t, uint32_t] build_parent_map(string filename)
    uint32_t resolve_tree(mapcpp[uint32_t, uint32_t] &hit_counts,
                          mapcpp[uint32_t, uint32_t] &parent_map,
                          )
