import unittest
import numpy.testing as npt
from taxanet.dataset import (one_hot, DefaultMapper, DictMapper,
                             FastaCollection,
                             )
from taxanet.testing import TestCase as TNTestCase
from pykraken import PyKraken
from torch.utils.data import DataLoader


class MapperTestCases(unittest.TestCase):

    def test_dict_mapper(self):
        dict_ = {'a': 0, 'b': 1}
        mapper = DictMapper(dict_)
        self.assertEqual(mapper('a'), 0)
        self.assertEqual(mapper('b'), 1)
        with self.assertRaises(KeyError):
            mapper('c')

    def test_dict_mapper_with_default(self):
        dict_ = {'a': 0, 'b': 1}
        mapper = DictMapper(dict_, default='fish')
        self.assertEqual(mapper('a'), 0)
        self.assertEqual(mapper('b'), 1)
        self.assertEqual(mapper('c'), 'fish')

    def test_default_mapper(self):
        list_ = [1, 3, 2, 'fish', 'dog', 'puppy']
        mapper = DefaultMapper()
        first_pass = []
        for item in list_:
            first_pass.append(mapper(item))

        self.assertListEqual(first_pass, [0, 1, 2, 3, 4, 5])

        second_pass = []
        for item in reversed(list_):
            second_pass.append(mapper(item))
        self.assertListEqual(second_pass, [5, 4, 3, 2, 1, 0])


class OneHotTestCase(unittest.TestCase):

    def test_one_hot(self):
        test_seqs = [[b'A', b'V'], [b'C', b'G']]
        obs = one_hot(test_seqs)
        exp = [[[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
               [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
               ]
        npt.assert_array_equal(obs, exp)


class FastaCollectionTestCase(TNTestCase):

    # used in get_data_path to find testing data
    package = 'taxanet.testing'

    def test_fasta_collection(self):
        # TODO could make a smaller example
        files = [
            self.get_data_path('test1.fa'),
            self.get_data_path('test2.fa'),
        ]

        fa_collection = FastaCollection(files, 15)

    def test_fasta_collection_pykraken_integration(self):
        files = [
            self.get_data_path('test1.fa'),
            self.get_data_path('test2.fa'),
        ]
        DB_filename = self.get_data_path(
            "small_demo_db/database.kdb", package='pykraken.tests',
        )
        Index_filename = self.get_data_path(
            "small_demo_db/database.idx", package='pykraken.tests',
        )
        nodes_filename = self.get_data_path(
            "small_demo_db/taxonomy/nodes.dmp",
            package='pykraken.tests',
        )

        pk = PyKraken(DB_filename, Index_filename, nodes_filename)

        def classifier(X):
            """

            Parameters
            ----------
            X : list of np.array of bytes

            Returns
            -------

            """
            to_classify = [x.tostring() for x in X]
            return pk.classify_reads(to_classify)

        fa_collection = FastaCollection(files, 15,
                                        mapper=classifier,
                                        mapper_type='read')

    def test_fast_collection_with_dataloader(self):
        files = [
            self.get_data_path('test1.fa'),
            self.get_data_path('test2.fa'),
        ]

        fa_collection = FastaCollection(files, 15)

        loader = DataLoader(fa_collection, batch_size=4)

        for sample, label in loader:
            self.assertTupleEqual(sample.shape[1:], (15, 5))
            self.assertTrue(set(label.numpy()).issubset(set(
                fa_collection.labels)))

        self.assertEqual(fa_collection.n_classes, 2)


if __name__ == '__main__':
    unittest.main()
