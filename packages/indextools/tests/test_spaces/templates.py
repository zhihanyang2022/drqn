import copy
import unittest


#  unittest inheritance hack https://stackoverflow.com/a/25695512
class templates:
    class SpaceTest(unittest.TestCase):
        def setUp(self):
            self.space = self.new_space()

        @property
        def values(self):
            raise NotImplementedError

        def new_space(self):
            raise NotImplementedError

        def test_value(self):
            values = (self.space.value(i) for i in range(-self.space.nelems, 0))
            self.assertCountEqual(values, self.values)

            values = (self.space.value(i) for i in range(self.space.nelems))
            self.assertCountEqual(values, self.values)

        def test_value_invalid(self):
            self.assertRaises(
                IndexError, self.space.value, -self.space.nelems - 1
            )
            self.assertRaises(IndexError, self.space.value, self.space.nelems)

        def test_idx(self):
            indices = (self.space.idx(v) for v in self.values)
            self.assertCountEqual(indices, range(self.space.nelems))

        def test_index_invalid(self):
            self.assertRaises(ValueError, self.space.idx, object())

        def test_elem(self):
            for i in range(self.space.nelems):
                self.assertIsInstance(self.space.elem(i), self.space.Elem)
            self.assertRaises(
                ValueError, self.space.elem, -self.space.nelems - 1
            )
            self.assertRaises(ValueError, self.space.elem, self.space.nelems)

            for v in self.values:
                self.assertIsInstance(self.space.elem(value=v), self.space.Elem)
            self.assertRaises(ValueError, self.space.elem, value=object())

        def test_isvalue(self):
            for v in self.values:
                self.assertTrue(self.space.isvalue(v))

        def test_iselem(self):
            for i in range(self.space.nelems):
                self.assertTrue(self.space.iselem(self.space.elem(i)))
            self.assertFalse(self.space.iselem(object()))

        def test_values(self):
            self.assertCountEqual(self.space.values, self.values)

        def test_elems(self):
            elems = (self.space.elem(i) for i in range(self.space.nelems))
            self.assertCountEqual(self.space.elems, elems)

            elems = (self.space.elem(value=v) for v in self.values)
            self.assertCountEqual(self.space.elems, elems)

        def test_items(self):
            indices, values = zip(*self.space.items())
            self.assertCountEqual(indices, range(self.space.nelems))
            self.assertCountEqual(values, self.values)

    class ElemTest(unittest.TestCase):
        def setUp(self):
            self.space = self.new_space()

        def new_space(self):
            raise NotImplementedError

        def test_space(self):
            for e in self.space.elems:
                self.assertIs(e.space, self.space)

        def test_idx(self):
            for i, e in enumerate(self.space.elems):
                self.assertEqual(e.idx, i)

        def test_index__(self):
            for e in self.space.elems:
                self.assertIsInstance(e.__index__(), int)
                self.assertEqual(e.__index__(), e.idx)

        def test_copy(self):
            for e in self.space.elems:
                c = copy.copy(e)
                self.assertIsNot(e, c)
                self.assertEqual(e, c)
                self.assertIs(e.space, c.space)

        def test_equality(self):
            for i in range(self.space.nelems):
                self.assertEqual(self.space.elem(i), self.space.elem(i))

            _space = self.new_space()
            for i in range(self.space.nelems):
                self.assertNotEqual(self.space.elem(i), _space.elem(i))

        def test_value_getter(self):
            for i, v in self.space.items():
                self.assertEqual(self.space.elem(i).value, v)

        def test_value_setter(self):
            e = self.space.elem(0)

            for i, v in self.space.items():
                e.value = v
                self.assertEqual(e.idx, i)
