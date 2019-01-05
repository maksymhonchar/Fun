from abc import ABCMeta, abstractmethod

class Sequence(metaclass=ABCMeta):
    """Own version of collection.Sequence abstract base class.
    
    ABCMeta declaration assures that the constructor for the class raises an error.

    @abstractmethod means that we do not provide an implementation within the Sequnce base class, but that we expect any concrete subclasses to support those two methods.
    """

    @abstractmethod
    def __len__(self):
        """Return the length of the sequence"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, j):
        raise NotImplementedError

    def __contains__(self, val):
        """Return True if val found in the sequence; False otherwise."""
        for obj in self:
            if obj == val:
                return True
        return False

    def count(self, val):
        """Return the number of elements equal to given value."""
        k = 0
        for obj in self:
            if obj == val:
                k += 1
        return k