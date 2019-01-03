class CustomVector(object):
    """Represent a vector in a multidimensional space."""

    def __init__(self, d):
        """Create d-dimensional vector of zeros."""
        self._coords = [0] * d

    def __len__(self):
        """Return the dimension of the vector."""
        return len(self._coords)

    def __getitem__(self, j):
        """Return j-th coordinate of vector."""
        return self._coords[j]
    
    def __setitem__(self, j, val):
        """Set j-th coordinate of vector to given value val."""
        self._coords[j] = val

    def __add__(self, other):
        """Return sum of two vectors"""
        if not isinstance(other, CustomVector):
            raise TypeError("Sum only applies to instances of CustomVector.")
        if len(self) != len(other):
            raise ValueError("Dimensions of the vectors must agree.")
        result = CustomVector(len(self))  # start with vector of zeros
        for j in range(len(self)):
            result[j] = self[j] + other[j]
        return result

    def __eq__(self, other):
        """Return True if vector has same coordinates as other."""
        return self._coords == other._coords

    def __ne__(self, other):
        """Return True if vector differs from other."""
        return not (self == other)

    def __str__(self):
        """Produce string representation of vector."""
        return '<{0}>'.format(str(self._coords)[1:-1])

if __name__ == "__main__":
    v = CustomVector(5)
    v[1] = 1
    v[-1] = -1
    print(v)
    u = v + v
    print(u)
    total = 0
    for entry in v:
        total += entry
    print(v, total)
