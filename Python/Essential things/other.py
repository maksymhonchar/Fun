# [argv] from [sys] module example
from sys import argv
def f1():
    try:
        a, b, c, d = argv
    except ValueError:
        print('Wrong usage. It should be 3 arguments.')
        return
    print(r"b,c,d:", b, c, d)
    print(r'a was', a)

# yield example
def func():
    y = 0
    y += 1
    yield y
for i in func():
    print(i)
    print(func().__next__())


# Creating a reserve copy
import os
import time
import zipfile

source = ['C:/Users/Max/Documents/PythonWorkspace/test/text1.txt',
          'C:/Users/Max/Documents/PythonWorkspace/test/text2.txt']
target_dir = ['C:/Users/Max/Documents/PythonWorkspace/test/']
target = target_dir[0] + time.strftime('%Y-%m-%d %H-%M-%S') + '.zip'
print('Starting archiving...')
zf = zipfile.ZipFile(target, mode="w")
try:
    print('Adding text1.txt')
    zf.write(source[0])
finally:
    print('Closing stream')
    zf.close()
print('Archiving ended')


# Dynamic typing example
from types import *

def what(x):
    if isinstance(x, int):
        print('This is an integer')
    else:
        print('This is something else')

what(123)
what([12, 23, 2])
what("asdf")

dict1 = {
    int: lambda x, y: x+y,
    str: lambda x, y: y.join(x.split()),
    list: lambda x, y: [i**2 for i in range(x+y)]
}

def something(a,b):
    if isinstance(a, type(b)):
        print('Arguments are not the same type')
    if not type(a) in dict1.keys():
        raise 'Don\'t know what are you talking about'
    else:
        return dict1[type(a)](a, b)

print(something(4, 5))


# Exception example
def f2():
    try:
        sentence = input('Enter something')
    except EOFError:
        print('Don`t pick Ctrl+D !')
    except KeyboardInterrupt:
        print("Don't pick Ctrl+C !")
    else:
        print('You have inputed {0}'.format(sentence))

# Some shit here
def total(initial=5, *numbers, **keywords):
    count = initial;
    for number in numbers:
        count += number
    for key in keywords:
        count += keywords[key]
    return count


def total(initial=5, *numbers, extra_number):
    count = initial
    for number in numbers:
        count += number
    count += extra_number
    return count


def maximum(x, y):
    """
    Return the maximal value from x and y.

    :param x: first value to compare.
    :param y: second value to compare.
    :return: Maximal value
    """
    if x > y:
        return x
    elif y > x:
        return y
    else:
        print('Equal')


def nonefunction():
    pass

a = maximum(123, 23)
print(a)
a = maximum.__doc__
print(a)


# Another shitty example
def sayhi():
    print('hi!')
__version__ = '0.1'
print(__name__)


# Playing with numbers
import math

pi_rational = float.as_integer_ratio(math.pi)
print(pi_rational)
testFloat_rational = 123.123.as_integer_ratio()
print(testFloat_rational)

testFloat_2 = 123.456
if testFloat_2.is_integer():
    print('This float can be represented as integer value!')
else:
    print('This cannot be represented as integer value.')

veryLongValue = 999999999999999999999999999999999999999999999
numOfBytesInVeryLongValue = veryLongValue.bit_length()
print(numOfBytesInVeryLongValue, 'bytes')

# Purpose of [is] keyword
list1 = [1, 2, 3, 4]
list1_pointer = list1
if list1 is list1_pointer:
    print('pointer catched!')
else:
    print('there is no pointer...')

# Some lambda anonymous functions
g = lambda x: x ** 2
print(g(8))
print(int(g(2 * math.sqrt(2))))

list2 = [2, 18, 9, 22, 17, 24, 8, 12, 27]
ffilter = filter(lambda x: x % 3 == 0, list2)
for i in ffilter:
    print(i, end=' ')
else:
    print('')

# integer -> number w/ floating point -> complex number
a = 123 + 213  # integer
b = 123.23 + 23.23  # double
c = (1 + 2j) + (2 + 3j)  # complex

d = 123 + 123.23  # double
e = 123.23 + (12 + 8j)  # complex
f = 123 + (5 + 5j)  # complex
print(a, b, c, d, e, f)

qq = float(123.23)
print(qq, type(qq))

# repr() - extended format. str() - user-friendly format.
# both return type <class 'str'>
num = 1 / 3
print(repr(num), type(repr(num)))  # No changes! why?
print(str(num), type(str(num)))
print('Raw representation: %r' % num)

result = 5 < 123 < 1232, 32 > 122 > 23
print(result)
print(1 < 2 < 3 < 4 < 5 < 6 < 7)

print(int(True == True))

# Playing with numbers -- v2
import random
import decimal
from fractions import Fraction

eval("print('hello, world!')")

x = 192836416238471238647891263498126384689123648
print(bin(x), oct(x), hex(x), sep='\n')
print('{0:b}\n{1:o}\n{2:X}'.format(x, x, x))

print(x.bit_length(), len(bin(x)) - 2)
print(random.randint(0, 123))
print(random.choice(['first', 'second', 'third']))

print(0.1 + 0.1 + 0.1 - 0.3)
print(Fraction(0.1) + Fraction(0.1) - Fraction(0.2))

x1 = Fraction(1, 3)
x2 = Fraction(4, 6)
print(x1, x2)
print(x1 - x2, x1 * x2, x1 / x2)

decimal.getcontext().prec = 5
print(decimal.Decimal(1) / decimal.Decimal(3))

print(2.5.as_integer_ratio())
f = 2.5
z = Fraction(*f.as_integer_ratio())
print(z)

s1 = set('asdfasdf')
print(s1)
print(type(s1))
if 'd' in s1:
    print('ye')
s2 = set('asss')
print(s1 > s2, s1 < s2)  # надмножество, подмножество

# try-finally example
import time

try:
    f = open('poem.txt')
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        print(line, end=' ')
        time.sleep(2)
except KeyboardInterrupt:
    print('\nUser stopped reading file process.')
finally:
    f.close()
    print('File stream was closed in [finally] block.')

# test for the python version
import sys
import warnings

print(sys.version_info[0])

if sys.version_info[0] < 3:
    warnings.warn('Use python >3.0 pls', RuntimeWarning)
else:
    print('Everything is cool')

# TODO: do something with exceptions (another file maybe)
class ShortInputException(Exception):
    ''' Class for user exceptions '''
    def __init__(self, length, atleast):
        Exception.__init__(self)
        self.length = length
        self.atleast = atleast
def testExceptions():
    try:
        text = input('Enter something: >>')
        if len(text) < 3:
            raise ShortInputException(len(text), 3)
    except EOFError:
        print("Oh no, EOF error...")
    except ShortInputException as ex:
        print('Short input exception: Length of the inputed string is {0:d}. It should be at least {1:d}'.format(
            ex.length, ex.atleast))
    else:
        print('Hurray, no exceptions!')

# some random things
a = 123123;
print(a.__int__)
print(type(a.__int__))

# Pickling example
import pickle
# Name of the file to save object.
filename = 'myownlist.data'
# List to save into the file.
myownlist = ['first', 'second', 'third']
list2 = ['fourth', 'fifth']

# Write into the file.
f = open(filename, "wb")
pickle.dump(list2, f)
pickle.dump(myownlist, f)
f.close()

del myownlist

# Read list from the file
f = open(filename, "rb")
storedlist = pickle.load(f)
print(storedlist)

# Pickling _v2
D = {'a': 1, 'b': 2, 'c': 3}

# Save data into .pkl file
F = open('datafile.pkl', 'wb')
pickle.dump(D, F)
F.close()

# Read data from .pkl file
F = open('datafile.pkl', 'rb')
D_v2 = pickle.load(F)
print(D_v2)
# F.close()

print('%r' % open('datafile.pkl', 'rb').read())

# Alternative to [else] block after [while] block
found = False
while x and not found:
    if x[0] == target:
        print('Found!')
        found = True
    else:
        x = x[1:]
if not found:
    print('not found')

# The same thing, but WITH [while-else] block
while x:
    if x[0] == target:
        print('Found', target, '!')
        break
    x = x[1:]
else:  # this block will execute, only if whole [while] block will be processed
    print('Not found')

# Alternatives fot c-like [while] loops, like:
# while ( (x=next()) != NULL) { do something with x }

# Alternative #1
"""
while True:
    x = next()
    if not x: break
    ... do something with x ...
"""
# Alternative #2
"""
x = True
while x:
    x = next()
    if x:
        ... do something with x ...
"""
# Alternative #3
"""
x = next()
while x:
    ... do something with x ...
    x = next()
"""

# zip() function
# zip() function uses for loop for going through one ore more sequences
L1 = [1, 2, 3, 4]
L2 = [5, 6, 7, 8]
print(zip(L1, L2), type(zip(L1, L2)))  # <zip object at ...>  <class 'zip'>
print(list(zip(L1, L2)))  # [ (1,5), (2,6), (3,7), (4,8) ]

for (x,y) in zip(L1, L2):
    print(x, y, '--', x+y, type(x), type(y))  # 1 5 -- 6 int int ...

# map() function
result = list(map(ord, 'spam'))
print(result, type(result))  # [115, 112, 97, 109] <class 'list'>

# enumerate() function
S = 'spam'
index = 0
for item in S:
    print(item, 'appears at index', index)
    index += 1
# Alternative for above, using enumerate() function
S2 = 'spam2'
for (index, item) in enumerate(S2):
    print(item, 'appears at index', index)  # Same result

# [and], [or] logical operators DO NOT return [True] or [False]
# In python, logical operators return either left, or right object.

# [or] operation
# Returns first [True] object, or [right object], if everything is False
print(2 or 3, 3 or 2)  # (2, 3)
print([] or 3)  # 3
print([] or {})  # {}

# [and] operation
# Returns first [False] object, or [right object] if everything if True
print(2 and 3, 3 and 2)  # (3, 2)
print([] and {})  # []
print(3 and [])  # []

# Ternary operator example:
# [A = X if Y else Z] instead of [A = Y ? X : Z]
A = 't' if 'spam' else 'f'
print(A)  # 't'
B = 't' if [] else 'f'
print(B)  # 'f'
# Alternative in older versions:
# ((X and Y) or Z)  == if X then Y else Z
C = (('t' and 'spam') or 'f')
print(C)  # 'spam'
C = (([] and 'spam') or 'f')
print(C)  # 'f'
# Cool alternative:
# A = [Z, Y][bool(X)]
D = ['no', 'yes'][bool([])]
D_2 = ['no', 'yes'][bool('not empty object')]
print(D, D_2)  # no yes

# To get NONEPMTY element from the set:
# nonempty = A or B or C or None

# Assigning with default value
# X = A or default

# Unpacking example
# Unpacking - it's like slicing.
# Python can unpack not only [lists], but also [strings]
# Element with '*' always will be an asterisk
seq = [1, 2, 3, 4]
a, b, *c = seq
print(a, b, c)  # 1, 2, [3, 4]
print(type(a), type(b), type(c))  # int int list

a, *b, c = seq
print(a, b, c)  # 1, [2, 3], 4
print(type(a), type(b), type(c))  # int list int

a, b, *c, d = seq
print(a, b, c, d)  # 1, 2, [3], 4
print(type(a), type(b), type(c), type(d))  # int int list int

*a, b = seq
print(a, b)  # [1, 2, 3], 4
print(type(a), type(b))  # list int

# Another example
L = [1, 2, 3, 4]
while L:
    front, *L = L
    print(front, L)

# If there are no elements for '*' variable, it'll be filled with []
# We don't care where the element [*e] stands.
a, b, *e, c, d = seq
print(a, b, c, d, e)  # 1 2 3 4 []

# *a, *b = seq  - Syntax error: double [*_] elements
# *a = seq - Syntax error
# but:
*a, = seq
print(a, type(a))  # [1, 2, 3, 4] <class 'list'>

# Unpacking in loops
for (a, *b, c) in [(1,2,3,4), (5,6,7,8)]:
    print(a, b, c)  # 1 [2,3] 4 |||  5 [6,7] 8

print(print('hi guys'))  # hi guys\nNone

# How to lose a data:
L = [1, 2, 3]
L = L.append(3)
print(L)  # None

# <class 'bool'> usage
# Note, that all (!) objects can be True or False
print(type(True), type(False))

print(bool(1), bool('eggs'), bool({}))  # True True False

# checking for types equality
print(type([1]) == type([]))  # True
print(type([1]) == type(list))  # False
print(isinstance([1], (list, tuple)))  # True

from types import FunctionType
def f(): pass
print(type(f) == FunctionType)
print(type(f))  # <class 'function'>
print(type(f) == "<class 'function'>")  # False, lulxd

import random
import string
import sys
# import warnings as myOwnNameForWarningsModule

# simple numbers
list = range(5, 10)
for i in range(len(list)):
    print(random.choice(list))

print('')
for i in range(10):
    b = random.random()
    print(b)

print('[i] variable exists! Here is it\'s value:', i)
del i  # NameError - i doesn't exists anymore

print('')
a = set('abcasdfasdf')
print(type(a))
print(a)

# simple strings
mystr = 'my amazing str'
for i in range(len(mystr)):
    print(i, mystr[i], sep=' ', end=' ')
print('')
for i in reversed(mystr):
    print(i, end=' ')
else:
    print('')
print(mystr[::-1])
print('\n')

# slice of numbers
slice1 = range(10)[::2]
print(type(slice1), slice1)
for i in slice1:
    print(i, end=' ')
# slice of characters (using "import strings")
slice2 = string.ascii_lowercase[0:10]  # First 10 letters
print('\n' + slice2)
slice3 = string.ascii_uppercase[:-11:-1]  # Last 10 letters
print(type(slice3))  # str
print(slice3)
# Copy strings
str11 = 'hello'
str12 = str11[:]
print(str11, str12)

print('')
st1 = 'very very long string'
result = st1.find('lon')
print(result)
result = st1.find('asdf')
print(result)

st2 = st1[:]
print(st2)
st2 = st2.replace('very', 'not really')
print(st2)

list_1 = st2.split(' ')
print(list_1)

''' DOESN'T WORK :c
str_to_list = 'hello world'
list_2 = []
for char in str_to_list:
    list_2.append(char)
data = bytes(list_2)
print(type(data))
for d in data:
    print(d)
'''

# simple strings
mystr = 'my amazing str'
for i in range(len(mystr)):
    print(i, mystr[i], sep=' ', end=' ')
print('')
for i in reversed(mystr):
    print(i, end=' ')
else:
    print('')
print(mystr[::-1])
print('\n')

# slice of numbers
slice1 = range(10)[::2]
print(type(slice1), slice1)
for i in slice1:
    print(i, end=' ')
# slice of characters (using "import strings")
slice2 = string.ascii_lowercase[0:10]  # First 10 letters
print('\n' + slice2)
slice3 = string.ascii_uppercase[:-11:-1]  # Last 10 letters
print(type(slice3))  # str
print(slice3)
# Copy strings
str11 = 'hello'
str12 = str11[:]
print(str11, str12)

print('')
st1 = 'very very long string'
result = st1.find('lon')
print(result)
result = st1.find('asdf')
print(result)

st2 = st1[:]
print(st2)
st2 = st2.replace('very', 'not really')
print(st2)

list_1 = st2.split(' ')
print(list_1)

''' DOESN'T WORK :c
str_to_list = 'hello world'
list_2 = []
for char in str_to_list:
    list_2.append(char)
data = bytes(list_2)
print(type(data))
for d in data:
    print(d)
'''

def recursion(depth):
    if depth >= 997:
        return depth
    ress = recursion(depth + 1)
    return ress

max_depth = recursion(0)
print(max_depth)  # 997 is maximal depth
print(sys.maxsize)

# Comprehension expression example
matrix_1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [20, 0, 0]
]
col2 = [raw[1] for raw in matrix_1]
print(col2)
col2_3_tuple = [[raw[1], raw[2]] for raw in matrix_1]
print(col2_3_tuple)

dict_1 = {i:2**i for i in range(11)}
print(dict_1)

set_1 = {ord(x) for x in 'hello world'}
print(set_1)
set_in_chars = {chr(x) for x in set_1}
print(set_in_chars)

# 'type' class
list_4 = [1,2,3]
print(type(type(type(list_4))))
if isinstance(list_4, type({})):
    print('This is a dict!')
else:
    print('This is not a dict')

T = [(1, 2), [3, 4], (5, 6)]
for item in T:
    print(type(item))  # tuple list tuple

for ((a, b), c) in [([1, 2], 3), ['XY', 6]]:
    print(a, b, c)  # 1 2 3\nX Y 6

for (a, *b) in [(1, 2, 3, 4), (5, 6, 7, 8)]:
    print(a, b, type(a), type(b))  # int list

# Dictionaries can be as [switch-case]:
# Alternative for [switch-case] alternative
choice = 'ham'
try:
    print(
        {
            'spam': 1,
            'eggs': 2,
            'helloworld': 3,
            'ham': 4
        }[choice]
    )
except:
    print('Bad choice')
else:
    print('Ended successfully')

stuff = {
    'name': 'tmp',
    'age': 9,
    'height': 180
}

# Some information about documentation things in python:
# Sources of documentation examples:
# 1. comments [#]
# 2. dir() and help() functions
# 3. __doc__ documentation strings
print(dir(sys))  # prints all objects attributes
print(dir([]), dir({}), dir(''), sep='\n')
print(dir(str) == dir(''))  # True

def f():
    pass
print(f.__doc__)  # None
def f2():
    """dummy function"""
    pass
print(f2.__doc__)  # dummy function

class t():
    """dummy class"""
    pass
tobj = t()
print(tobj.__doc__)  # dummy class
print(t.__doc__)  # dummy class

# help() prints detailed documentation. Returns None.
help_result = help(t)
print(help_result, type(help_result), sep='\n')  # None <class 'NoneType'>

