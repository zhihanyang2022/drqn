import itertools as itt
import unittest

import indextools

from .templates import templates


class JointBase:
    def setUp(self):
        self.spaces = (
            indextools.BoolSpace(),
            indextools.DomainSpace('abc'),
            indextools.RangeSpace(4),
        )
        super().setUp()

    def new_space(self):
        return indextools.JointSpace(*self.spaces)

    @property
    def values(self):
        return itt.product([False, True], 'abc', range(4))


class JointSpaceTest(JointBase, templates.SpaceTest):
    def test_nelems(self):
        self.assertEqual(self.space.nelems, 24)

    def test_getitem(self):
        self.assertIs(self.space[-3], self.spaces[0])
        self.assertIs(self.space[-2], self.spaces[1])
        self.assertIs(self.space[-1], self.spaces[2])
        self.assertIs(self.space[0], self.spaces[0])
        self.assertIs(self.space[1], self.spaces[1])
        self.assertIs(self.space[2], self.spaces[2])

        self.assertRaises(IndexError, self.space.__getitem__, -4)
        self.assertRaises(IndexError, self.space.__getitem__, 3)

    def test_elem_values(self):
        for e in self.space.elems:
            self.assertTrue(self.spaces[0].iselem(e[0]))
            self.assertTrue(self.spaces[1].iselem(e[1]))
            self.assertTrue(self.spaces[2].iselem(e[2]))


class JointNamedBase(unittest.TestCase):
    def setUp(self):
        self.spaces = dict(
            a=indextools.BoolSpace(),
            b=indextools.DomainSpace('abc'),
            c=indextools.RangeSpace(4),
        )
        super().setUp()

    def new_space(self):
        return indextools.JointNamedSpace(**self.spaces)

    @property
    def values(self):
        for a, b, c in itt.product([False, True], 'abc', range(4)):
            yield (a, b, c)


class JointNamedSpaceTest(JointNamedBase, templates.SpaceTest):
    def test_nelems(self):
        self.assertEqual(self.space.nelems, 24)

    def test_getattr(self):
        self.assertIs(self.space.a, self.spaces['a'])
        self.assertIs(self.space.b, self.spaces['b'])
        self.assertIs(self.space.c, self.spaces['c'])

    def test_getattr_error(self):
        self.assertRaises(AttributeError, self.space.__getattr__, '0')
        self.assertRaises(AttributeError, self.space.__getattr__, 'd')

    def test_elem_values(self):
        for e in self.space.elems:
            self.assertTrue(self.spaces['a'].iselem(e.a))
            self.assertTrue(self.spaces['b'].iselem(e.b))
            self.assertTrue(self.spaces['c'].iselem(e.c))


class JointOtherTests(unittest.TestCase):
    def test_invalid_values(self):
        self.assertRaises(TypeError, indextools.JointSpace, object(), object())
        self.assertRaises(
            TypeError, indextools.JointNamedSpace, a=object(), b=object()
        )
