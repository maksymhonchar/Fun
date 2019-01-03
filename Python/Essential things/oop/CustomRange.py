"""
xrange Python2 and range in Python3 use lazy evaluation strategy. Rather than creating a new list instance, range is a class that can effectively represent the desired range of elements without ever storing them explicitly in memory.
"""

class CustomRange(object):
    """A class that mimics the built-in range class."""

    def __init__(self, start, stop=None, step=1):
        """Initialize a Range instance.

        Semantics is similar to built-in range class from Python3.
        """
        if step == 0:
            raise ValueError('Step cannot be 0.')
        
        if stop is None:
            start, stop = 0, start
        
        # Get the length once.
        self._length = max(0, (stop - start + step - 1) // step)

        # We need to know start and step (but not stop) to support __getitem__
        self._start = start
        self._step = step
    
    def __len__(self):
        """Return number of entries in the range."""
        return self._length

    def __getitem__(self, k):
        """Return entry at index k (using standard interpretation if negative)."""
        if k < 0:
            k += len(self)  # an attempt to convert negative index
        
        if not 0 <= k < self._length:
            raise IndexError('Index {0} is out of range.'.format(k))
        
        return self._start + k * self._step

if __name__ == "__main__":
    for i in CustomRange(10, 20, 3):
        print(i)
