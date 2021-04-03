from .space import Space

# Less efficient
# class RangeSpace(DomainSpace):
#     def __init__(self, *args):
#         """Alias for DomainSpace(range(*args))."""
#         super().__init__(range(*args))


class RangeSpace(Space):
    """More efficient than DomainSpace(range(*args))."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__range = range(*args, **kwargs)

    def __hash__(self):
        return hash(self.__range)

    @property
    def nelems(self):
        return len(self.__range)

    def value(self, idx):
        return self.__range[idx]

    def idx(self, value):
        return self.__range.index(value)
