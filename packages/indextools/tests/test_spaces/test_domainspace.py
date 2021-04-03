import unittest

import indextools

from .templates import templates


class DomainBase:
    @staticmethod
    def new_space():
        return indextools.DomainSpace('abc')

    @property
    def values(self):
        return 'abc'


class DomainSpaceTest(DomainBase, templates.SpaceTest):
    def test_nelems(self):
        self.assertEqual(self.space.nelems, 3)


class DomainElemTest(DomainBase, templates.ElemTest):
    pass


class DomainOtherTests(unittest.TestCase):
    def test_invalid_values(self):
        self.assertRaises(TypeError, indextools.DomainSpace, [[1], [2], [3]])
