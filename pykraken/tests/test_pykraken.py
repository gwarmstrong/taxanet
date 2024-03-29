import unittest
from pykraken import PyKraken


class TestPyKraken(unittest.TestCase):

    def test_pykraken_runs(self):
        DB_filename = "./data/small_demo_db/database.kdb"
        Index_filename = "./data/small_demo_db/database.idx"
        nodes_filename = "./data/small_demo_db/taxonomy/nodes.dmp"

        pk = PyKraken(DB_filename, Index_filename, nodes_filename)
        read = b"AGAAGCGAAGGTTCATATGGTCTGACGGATCTCTTCGAGCACCCGAGAATTCCA"
        print("{} classified as: {}".format(read, pk.classify_read(read)))
        print("read length: {}".format(len(read)))

        reads = [b"AGGGAGAGAGAGA", b"AGAAGCGAAGGTTCAT",
                 b"CTGACGGATCTCTTCGAGCA", b"CCGAGAATTCCA"]
        cls_ = pk.classify_reads(reads)
        print(cls_)
        print("Database k: {}".format(pk.k))
        print("Database number of Keys: {}".format(pk.key_count))
        print("Database taxa counts: {}".format(pk.taxa_counts()))


if __name__ == "__main__":
    unittest.main()
