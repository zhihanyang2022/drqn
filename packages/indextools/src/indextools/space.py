import warnings

_novalue = object()  # singleton placeholder for optional inputs


class Elem:
    def __init__(self, space, idx):
        """Element of an indexing space.

        :param space: A Space, an indexing space.
        :param idx: An int, an index in the space.
        """
        self.space = space
        self.idx = idx

    def __index__(self):
        # NOTE has to be very specific type (e.g. np.int64 not allowed)
        return int(self.idx)

    def __copy__(self):
        return self.space.elem(self.idx)

    @property
    def value(self):
        """Return element value."""
        return self.space.value(self.idx)

    @value.setter
    def value(self, value):
        """Set element value."""
        self.idx = self.space.idx(value)

    def __eq__(self, other):
        """Check equality against other element."""
        if not isinstance(other, Elem):
            warnings.warn(
                'Element comparison is restricted between Elem types; did you'
                'intend to compare the respective values?'
            )
            return NotImplemented

        return self.space is other.space and self.idx == other.idx

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'Elem({self.idx}: {self.value})'


class Space:
    Elem = Elem

    def value(self, idx):
        """Return value in this indexing space corresponding to ``idx``."""
        raise NotImplementedError

    def idx(self, value):
        """Return index in this indexing space corresponding to ``value``."""
        raise NotImplementedError

    def elem(self, idx=None, *, value=_novalue):
        """Return element of this indexing space.

        Usage::

            >>> import indextools
            >>> space = indextools.DomainSpace(('red', 'green', 'blue'))
            >>> elem = space.elem(1)
            >>> elem = space.elem(value='red')

        :param idx: The index of the element.
        :param value: The value of the element.
        :rtype: A :class:`Elem <Elem>`
        """
        if idx is None and value is _novalue:
            raise ValueError('Neither index nor value is given')

        if idx is not None and idx not in self.indices:
            raise ValueError(f'Invalid index ({self.idx})')

        if (
            idx is not None
            and value is not _novalue
            and self.value(idx) != value
        ):
            raise ValueError(f'Index ({idx}) and value ({value}) do not match')

        idx = self.idx(value) if idx is None else idx
        return self.Elem(self, idx)  # NOTE this might be annoying for union...

    def isvalue(self, value):
        """ Check whether ``value`` belongs to this indexing space."""
        try:
            self.idx(value)
        except ValueError:
            return False
        return True

    def iselem(self, elem):
        """Check whether ``elem`` belongs to this indexing space."""
        try:
            return elem.space is self and elem.idx in self.indices
        except AttributeError:
            return False

    def isitem(self, idx, value):
        """Check whether ``idx`` and ``value`` are consistent in this space."""
        try:
            return idx == self.idx(value)
        except ValueError:
            return False

    @property
    def values(self):
        """Return a generator over value-space"""
        return map(self.value, self.indices)

    @property
    def indices(self):
        """Return a generator over index-space"""
        return range(self.nelems)

    @property
    def indices_neg(self):
        """Return a generator over negative index-space"""
        return range(-self.nelems, 0)

    @property
    def indices_ext(self):
        """Return a generator over extended index-space"""
        return range(-self.nelems, self.nelems)

    @property
    def elems(self):
        """Return a generator over elements"""
        return map(self.elem, self.indices)

    # TODO change into property
    def items(self):
        """Return a generator over index-value pairs"""
        for idx in self.indices:
            yield idx, self.value(idx)

    # TODO where is this used?
    def _check_idx(self, idx):
        """Check index within range, and returns non-negative equivalent."""
        if idx not in self.indices_ext:
            raise IndexError(
                f'Invalid index ({idx}) is outside {self.indices_ext}'
            )
        return idx % self.nelems

    def __contains__(self, other):
        """Check whether `other` is element or value of this space."""
        return self.iselem(other) or self.isvalue(other)

    def __len__(self):
        """Return number of elements in this space."""
        return self.nelems
