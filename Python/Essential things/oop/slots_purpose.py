"""
The special attribute __slots__ allows to explicitly state which instance attributes you expect your object instances to have.

Expected results of __slots__:
1. Faster attribute access.
2. Space savings in memory.

The space savings is from:
1. Storing value references in slots instead of __dict__
2. Denying __dict__ and __weakref__ creation if parent classes deny them and you declare __slots__.

Requirements:
1. To have attributes named in __slots__ to actually be stored in slots instead of a __dict__, a class must inherit from object.
2. To prevent the creation of a __dict__, you must inherit from object and all classes in the inheritance must declare __slots__ and none of them can have a '__dict__' entry.
"""

"""Why use __slots__: Faster attribute access."""

import timeit

class Foo(object):
    __slots__ = 'foo'

class Bar(object):
    pass

slotted = Foo()
not_slotted = Bar()

def get_set_delete_fn(obj):
    def get_set_delete():
        obj.foo = 'foo'
        obj.foo
        del obj.foo
    return get_set_delete
print(min(timeit.repeat(get_set_delete_fn(slotted))))  # 0.156
print(min(timeit.repeat(get_set_delete_fn(not_slotted))))  # 0.196


"""Why use __slots__: Memory savings."""
from sys import getsizeof
print(Foo.foo)
print(type(Foo.foo))
print(getsizeof(Foo.foo))

"""Demonstration of __slots__"""
class Base(object):
    __slots__ = ()
# now:
b = Base()
# b.a = 'a'  # AttributeError: 'Base' object has no attribute 'a'

class Child(Base):
    __slots__ = ('a', )
c = Child()
c.a = 'a'
print(c.a)
# BUT:
# c.b = 'b'  # AttributeError: 'Child' object has no attribute 'b'
