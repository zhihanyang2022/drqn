import unittest

import indextools

from .templates import templates


def is_prime(n):
    return n > 1 and all(n % i != 0 for i in range(2, n))


class SubBase(unittest.TestCase):
    def setUp(self):
        self.space = self.new_space()

    @staticmethod
    def new_space():
        return indextools.SubSpace(indextools.RangeSpace(50), is_prime)

    @property
    def values(self):
        return (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


class SubSpaceTest(SubBase, templates.SpaceTest):
    def test_nelems(self):
        self.assertEqual(self.space.nelems, 15)


class SubElemTest(SubBase, templates.ElemTest):
    pass


class SubOtherTests(unittest.TestCase):
    def test_invalid_values(self):
        self.assertRaises(TypeError, indextools.SubSpace, object(), object())
