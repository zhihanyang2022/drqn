import unittest

import indextools

from .templates import templates


class RangeBase:
    @staticmethod
    def new_space():
        return indextools.RangeSpace(5, 10)

    @property
    def values(self):
        return range(5, 10)


class RangeSpaceTest(RangeBase, templates.SpaceTest):
    def test_nelems(self):
        self.assertEqual(self.space.nelems, 5)


class RangeElemTest(RangeBase, templates.ElemTest):
    pass


class RangeInterfaceTest(unittest.TestCase):
    def test_range_init(self):
        # checks that same errors as `range` are thrown
        self.assertRaises(TypeError, indextools.RangeSpace)
        self.assertRaises(ValueError, indextools.RangeSpace, 0, 0, 0)
        self.assertRaises(TypeError, indextools.RangeSpace, 0, 0, 0, 0)

    def test_range_stop(self):
        space = indextools.RangeSpace(10)
        self.assertEqual(space.nelems, 10)
        self.assertCountEqual(space.values, range(10))

    def test_range_start(self):
        space = indextools.RangeSpace(-6, 10)
        self.assertEqual(space.nelems, 16)
        self.assertCountEqual(space.values, range(-6, 10))

    def test_range_step(self):
        space = indextools.RangeSpace(-6, 10, 2)
        self.assertEqual(space.nelems, 8)
        self.assertCountEqual(space.values, range(-6, 10, 2))
