import numpy as np

from .space import Space

# TODO NamedUnionSpace?  to allow for conflicts?
# this still won't work very well;  what if the subvalue spaces are different..
# how to know how to use a subspace?!


class UnionSpace(Space):
    def __init__(self, *spaces):
        super().__init__()
        self.spaces = spaces

        values = set().union(*(s.values for s in spaces))
        nelems = tuple(s.nelems for s in spaces)
        if len(values) != sum(nelems):
            raise ValueError('Spaces must have non-overlapping values')

        self._nelems = nelems
        self._nelems_cumsum = np.cumsum(nelems)
        self._nelems_cumsum_m1 = self._nelems_cumsum - nelems
        self.nelems = sum(nelems)

    def value(self, idx):
        idx = self._check_idx(idx)
        si, sidx = self._si_sidx(idx)
        return self.spaces[si].value(sidx)

    def idx(self, value):
        for si, space in enumerate(self.spaces):
            try:
                sidx = space.idx(value)
            except ValueError:
                pass
            else:
                return self._nelems_cumsum_m1[si] + sidx

        raise ValueError(
            f'Invalid value ({value}) does not belong to any space'
        )

    def _si_sidx(self, idx):
        _idx_mask = idx < self._nelems_cumsum
        si = np.where(_idx_mask)[0][0]
        sidx = idx - np.dot(self._nelems, ~_idx_mask)
        return si, sidx
