import unittest

import more_itertools as mitt

import indextools

from .templates import templates


class PowerBase(unittest.TestCase):
    def setUp(self):
        self.space = self.new_space()

    @staticmethod
    def new_space():
        return indextools.PowerSpace('abc')

    @property
    def values(self):
        return (set(v) for v in mitt.powerset('abc'))


class PowerSpaceTest(PowerBase, templates.SpaceTest):
    def test_nelems(self):
        self.assertEqual(self.space.nelems, 8)


class PowerElemTest(PowerBase, templates.ElemTest):
    def test_update(self):
        e = self.space.elem(value=set())

        e.update(include='abc')
        self.assertSetEqual(e.value, set('abc'))

        e.update(exclude='b')
        self.assertSetEqual(e.value, set('ac'))

        e.update(include='b', exclude='c')
        self.assertSetEqual(e.value, set('ab'))

        self.assertRaises(
            TypeError, e.update, include=object(), exclude=object()
        )

    def test_include(self):
        e = self.space.elem(value=set())

        e.include('a')
        self.assertSetEqual(e.value, set('a'))

        e.include('ab')
        self.assertSetEqual(e.value, set('ab'))

        e.include('abc')
        self.assertSetEqual(e.value, set('abc'))

        self.assertRaises(TypeError, e.include, object())

    def test_exclude(self):
        e = self.space.elem(value=set('abc'))

        e.exclude('a')
        self.assertSetEqual(e.value, set('bc'))

        e.exclude('ab')
        self.assertSetEqual(e.value, set('c'))

        e.exclude('abc')
        self.assertSetEqual(e.value, set())

        self.assertRaises(TypeError, e.exclude, object())
