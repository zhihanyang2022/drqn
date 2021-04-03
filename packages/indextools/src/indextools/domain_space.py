from .space import Space


class DomainSpace(Space):
    def __init__(self, domain):
        """Domain-based indexing space.

        :param iterable: An iterable, the domain.
        """
        super().__init__()
        if len(set(domain)) != len(domain):
            raise ValueError('Domain should not have values which are equal.')

        self.__domain = tuple(domain)
        self._indices = {value: idx for idx, value in enumerate(self.__domain)}

    @property
    def nelems(self):
        return len(self.__domain)

    def value(self, idx):
        return self.__domain[idx]

    def idx(self, value):
        """Return index corresponding to ``value``."""
        try:
            return self._indices[value]
        except KeyError:
            raise ValueError(
                f'Invalid value ({value}) does not belong to this domain space'
            )
