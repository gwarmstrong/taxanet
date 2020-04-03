#!/usr/bin/python
# from kraken import main; main('../kraken-dbs/small_demo/database.kdb')

from pykraken import PyKraken

DB_filename = "../kraken-dbs/small_demo/database.kdb"
Index_filename = "../kraken-dbs/small_demo/database.idx"
nodes_filename = "../kraken-dbs/small_demo/taxonomy/nodes.dmp"

pk = PyKraken(DB_filename, Index_filename, nodes_filename)
read = "AGAAGCGAAGGTTCATATGGTCTGACGGATCTCTTCGAGCACCCGAGAATTCCA"
print("{} classified as: {}".format(read, pk.classify_read(read)))
print("read length: {}".format(len(read)))

reads = ["AGGGAGAGAGAGA", "AGAAGCGAAGGTTCAT", "CTGACGGATCTCTTCGAGCA", "CCGAGAATTCCA"]
cls_ = pk.classify_reads(reads)
print(cls_)
print("Database k: {}".format(pk.k))
print("Database number of Keys: {}".format(pk.key_count))
print("Database taxa counts: {}".format(pk.taxa_counts()))

# cls_ = pk.classify_file('./small_demo/library/added/j35B9K65WH.fna', False)
