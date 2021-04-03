from collections import namedtuple

import numpy as np

from .space import Space


class JointElem(Space.Elem):
    def __init__(self, space, idx):
        self._etuple = space.etuple(idx)
        super().__init__(space, idx)

    @property
    def idx(self):
        indices = tuple(e.idx for e in self._etuple)
        return self.space.ravel_multi_index(indices)

    @idx.setter
    def idx(self, idx):
        indices = self.space.unravel_index(idx)
        for eidx, elem in zip(indices, self._etuple):
            elem.idx = eidx

    def __getitem__(self, key):
        return self._etuple[key]


class JointSpace(Space):
    Elem = JointElem

    def __init__(self, *spaces):
        """Product space of input spaces, themselves accessed via index.

        :param *spaces: Spaces.
        """
        if not all(isinstance(space, Space) for space in spaces):
            raise TypeError('All inputs should be Spaces')

        self.__spaces = spaces
        self.__shape = tuple(s.nelems for s in spaces)
        self.nelems = np.prod(self.__shape)

    def ravel_multi_index(self, indices):
        return np.ravel_multi_index(indices, self.__shape)

    def unravel_index(self, idx):
        return np.unravel_index(idx, self.__shape)

    def __getitem__(self, key):
        return self.__spaces[key]

    def idx(self, value):
        try:
            value = tuple(value)
        except TypeError:
            raise ValueError(f'Invalid value ({value}) is not iterable')

        if len(value) != len(self.__spaces):
            raise ValueError(
                f'Invalid value ({value}) should have {len(self.__spaces)} elements'
            )

        indices = tuple(s.idx(v) for s, v in zip(self.__spaces, value))
        return self.ravel_multi_index(indices)

    def value(self, idx):
        idx = self._check_idx(idx)
        indices = self.unravel_index(idx)
        return tuple(s.value(sidx) for s, sidx in zip(self.__spaces, indices))

    def etuple(self, idx):
        idx = self._check_idx(idx)
        indices = self.unravel_index(idx)
        return tuple(s.elem(sidx) for s, sidx in zip(self.__spaces, indices))


class JointNamedElem(JointElem):
    def __getattr__(self, attr):
        return getattr(self._etuple, attr)


class JointNamedSpace(Space):
    Elem = JointNamedElem

    def __init__(self, **spaces):
        """Product space of named input spaces, themselves accessed via attribute.

        :param **spaces: Named spaces.
        """
        if not all(isinstance(space, Space) for space in spaces.values()):
            raise TypeError('All inputs should be Spaces')

        self.__spaces = spaces
        self.__shape = tuple(s.nelems for s in spaces.values())
        self.nelems = np.prod(self.__shape)
        self.Value = namedtuple('Value', spaces.keys())

    def ravel_multi_index(self, indices):
        return np.ravel_multi_index(indices, self.__shape)

    def unravel_index(self, idx):
        return np.unravel_index(idx, self.__shape)

    def __getattr__(self, attr):
        try:
            return self.__spaces[attr]
        except KeyError:
            raise AttributeError

    def idx(self, value):
        try:
            value = tuple(value)
        except TypeError:
            raise ValueError(f'Invalid value ({value})')

        # if not all(hasattr(value, k) for k in self.__spaces):
        #     raise ValueError(
        #         f'Invalid value ({value}) does not have the space attributes '
        #         f'{tuple(self.__spaces.keys())}'
        #     )

        indices = tuple(s.idx(v) for s, v in zip(self.__spaces.values(), value))
        return self.ravel_multi_index(indices)

    def value(self, idx):
        idx = self._check_idx(idx)
        indices = self.unravel_index(idx)
        vdict = {
            k: s.value(sidx)
            for (k, s), sidx in zip(self.__spaces.items(), indices)
        }
        return self.Value(**vdict)

    def etuple(self, idx):
        idx = self._check_idx(idx)
        indices = self.unravel_index(idx)
        edict = {
            k: s.elem(sidx)
            for (k, s), sidx in zip(self.__spaces.items(), indices)
        }
        return self.Value(**edict)
