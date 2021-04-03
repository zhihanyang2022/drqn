from .space import Space


class PowerElem(Space.Elem):
    def update(self, *, include=(), exclude=()):
        include = set(include)
        exclude = set(exclude)
        self.value = (self.value - exclude) | (include - exclude)

    def include(self, value):
        self.value |= set(value)

    def exclude(self, value):
        self.value -= set(value)


class PowerSpace(Space):
    Elem = PowerElem

    def __init__(self, values):
        """Powerset space.

        :param values: An iterable, the values of the powerset space.
        """
        super().__init__()

        values = set(values)
        self.__bitmap = {v: 1 << i for i, v in enumerate(values)}
        self.nelems = 1 << len(self.__bitmap)

    def value(self, idx):
        idx = self._check_idx(idx)
        return set(value for value, bit in self.__bitmap.items() if idx & bit)

    def idx(self, value):
        try:
            value = set(value)
        except TypeError:
            raise ValueError(f'Invalid value ({value}) is not iterable')

        return sum(self.__bitmap[v] for v in value)
