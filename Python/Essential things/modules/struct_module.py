# Examples of [struct] module.
# Purpose pf this module: to convert between strings and binary data.

# NOTE, that [fmt] string  in arguments means a [formatted string]

# Packing and unpacking data into/from strings
from struct import *
import sys

# A simple example
buffer = pack('ihb', 1, 2, 3)  # int short signed_char
print(repr(buffer))
print(unpack('ihb', buffer))  # (1, 2, 3)

# Another example.
packed_str = pack('hhl', 1, 2, 3)  # short short long
print(packed_str)
sh_1, sh_2, long_1 = unpack('hhl', packed_str)
print(sh_1, sh_2, long_1)  # 1 2 3

# Another example - pack a list with big-endian
data = [1, 2, 3]
buffer = pack('>ihb', *data)  # [>] for big-endian
print(repr(buffer))
print(unpack('>ihb', buffer))

# Get size of the struct.
print(calcsize('hhl'))

# Native byteorder
print('Native byteorder:', sys.byteorder)

values = (1, 'ab', 2.6)
s = Struct('I 2s f')
print(s.format, s.size, sep='\n')  # [b'I 2s f'] [2]

record = b'raymond   \x32\x12\x08\x01\x08'
name, serialnum, school, gradelevel = unpack('<10sHHb', record)
# String [name] will be as binary string, so we should decode it:
name = name.decode('ascii').strip()
print(name, serialnum, school, gradelevel)

# Unpacked fields can be named.
from collections import namedtuple
record = b'raymond   \x32\x12\x08\x01\x08'
student = namedtuple('Student', 'name serialnum school gradelevel')
student._make(unpack('<10sHHb', record))
print(student)

# We can change byte orders:
data = [3, 4, 5]
buffer = pack('hhh', *data)  # native
print('Native byte order', buffer)
buffer = pack('!hhh', *data)  # network
print('Network byte order', buffer)
