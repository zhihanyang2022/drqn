from .domain_space import DomainSpace


class BoolSpace(DomainSpace):
    def __init__(self):
        """Alias for DomainSpace((False, True))."""
        super().__init__((False, True))
