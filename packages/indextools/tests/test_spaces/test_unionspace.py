import itertools as itt
import unittest

import indextools

from .templates import templates


class UnionBase(unittest.TestCase):
    def setUp(self):
        self.space = self.new_space()

    @staticmethod
    def new_space():
        return indextools.UnionSpace(
            indextools.BoolSpace(),
            indextools.DomainSpace('abc'),
            indextools.RangeSpace(10, 14),
        )

    @property
    def values(self):
        return itt.chain([False, True], 'abc', range(10, 14))


class UnionSpaceTest(UnionBase, templates.SpaceTest):
    def test_nelems(self):
        self.assertEqual(self.space.nelems, 9)


class UnionElemTest(UnionBase, templates.ElemTest):
    pass
