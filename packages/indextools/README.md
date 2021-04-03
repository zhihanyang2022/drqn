# indextools This package provides classes which implement bijective mappings
between sequential indices and structured values.

## Why

I developed this package when I became in need for a way to represent tables of
values associated with arbitrary and semantically rich objects.  I needed a way
to quickly determine which entry of the table was associated with a specific
object instance, without the hassle of having to implement ad-hoc indexing on a
case-by-case fashion.

## Nomenclature and Notation

Space : Class which represents a set of values and implements value-to-index
and index-to-value mapping.

Elem : Wrapper over an index-value pair associated with a specific Space;
Changing the index automatically updates the corresponding value, and
viceversa.

## Available Spaces

This package contains various types of Spaces:

### DomainSpace, RangeSpace, BoolSpace

Represents the space of elements from a given set.

### JointSpace, JointNamedSpace

Represents cartesian products of values from other sets.

In JointSpace, the subspaces are referenced via indices;  In JointNamedSpace,
the subspaces are referenced via associated names.

### SubSpace

Represents a subset of the values from another space.

### PowerSpace

Represents the power-set of the values from another space.

### UnionSpace

Represents the union of values from other spaces.
