import numpy as np

from .space import Space

# TODO I need a good way of keeping the interface to the supelem
# class SubElem(Space.Elem):
#     def __init__(self, space, idx):
#         self._selem = space.supelem(idx)
#         super().__init__(space, idx)

#     @property
#     def idx(self):
#         sidx = self._selem.idx
#         return self.space.sidx_to_idx(sidx)

#     @idx.setter
#     def idx(self, idx):
#         try:
#             self._selem.sidx = self.space.idx_to_sidx(idx)
#         except AttributeError:
#             self._selem = self.space.supelem(idx)


class SubSpace(Space):
    # Elem = SubElem

    def __init__(self, space, *filters):
        """Subspace, containing only filtered elements.

        :param space: A Space.
        :param *filters: Filters.
        """
        if not isinstance(space, Space):
            raise TypeError('Invalid space ({space}) should be of type Space')
        if not filters:
            raise TypeError('Invalid filters ({filters}) should not be empty')

        self.space = space
        self.filters = filters

        _if = np.array(
            [[idx, self.filter(value)] for idx, value in space.items()]
        )
        _f = _if[:, 1].astype(bool)

        self._valid_indices = _if[_f, 0]
        self._num_nonvalids_before = np.cumsum(~_f)
        self.nelems = space.nelems - self._num_nonvalids_before[-1]

    def filter(self, value):
        if not self.space.isvalue(value):
            raise ValueError(
                f'Invalid value ({value}) does not belong to space'
            )

        return all(f(value) for f in self.filters)

    def value(self, idx):
        idx = self._check_idx(idx)
        sidx = self.idx_to_sidx(idx)
        return self.space.value(sidx)

    def idx(self, value):
        if not self.filter(value):
            raise ValueError(f'Invalid value ({value}) does not satisfy filter')
        sidx = self.space.idx(value)
        return self.sidx_to_idx(sidx)

    def idx_to_sidx(self, idx):
        return self._valid_indices[idx]

    def sidx_to_idx(self, sidx):
        return sidx - self._num_nonvalids_before[sidx]

    def supidx(self, idx):
        """Return SuperSet index corresponding to the SubSpace's index."""
        idx = self._check_idx(idx)
        return self.idx_to_sidx(idx)

    # TODO maybe only provide supidx
    def supelem(self, idx):
        """Return SuperSet element corresponding to the SubSpace's index."""
        sidx = self.supidx(idx)
        return self.space.elem(sidx)
