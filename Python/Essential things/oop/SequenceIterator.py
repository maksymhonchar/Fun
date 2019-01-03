"""
An Iterator for a collection provides one key behavior - It supports a special method __next__ that returns the next element of the collection, if any, or raises a StopIteration exception to indicate that there are no further elements.

Python provides an automatic implementation for any class that defines both __len__ and __getitem__. 
"""

class SequenceIterator(object):
    """An iterator for any of Python's sequence types."""

    def __init__(self, sequence):
        """Create an iterator for the given sequence.
        
        Keep a reference to the underlying data in self._seq variable.
        Save value self._k which will increment to 0 on first call to next.
        """
        self._seq = sequence
        self._k = -1

    def __next__(self):
        """Return the next element, or else raise StopIteration error."""
        self._k += 1
        if self._k < len(self._seq):
            return self._seq[self._k]
        else:
            raise StopIteration()

    def __iter__(self):
        """By convention, an iterator must return itself as an iterator"""
        return self

if __name__ == "__main__":
    s = SequenceIterator([1,2,3,4])

    # Way 1 to use it.
    # for element in s:
    #     print(element)
    # print(next(s))

    # Way 2 to use it (with next function call)
    while True:
        try:
            element = next(s)
            print(element)
        except StopIteration:
            print('StopIteration catched!')
            break
