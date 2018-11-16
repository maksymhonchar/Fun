def f2():
    errmsg = 'Usage: python copyingthings.py [pathFromFile] [pathToFile]'
    try:
        from_path, to_path = argv[1], argv[2]
        print(from_path)
        print(to_path)
        print('Trying to copy content from %s to %r' % (from_path, to_path))
        file_content = open(from_path).read()
        print(file_content)
        open(to_path, "w").write(file_content)
    except IOError:
        print("IOError: Can't open file stream.")
    except (ValueError, IndexError):
        print(errmsg)
    except EOFError:
        print('Stop it!')
    else:
        print('End of the program')

def short_implement():
    open(argv[2], "w").write(open(argv[1]).read())

# Examele for files I/O
print('Writing to the file')
poem = 'Nice example to write files ahahaha this is not a joke actually'
f = open('example.txt', 'a', encoding='utf-8')
f.write(poem)
f.close()
print('End writing to the file')

print('Reading from the file example.txt')
f = open('example.txt')
while True:
    line = f.readline()
    if len(line) == 0:
        break
    print(len(line))
f.close()
print('')
print('End reading from the file.')

# In-out exmaple
def reverse(text):
    return text[::-1]
def is_palindrome(text):
    return text == reverse(text)
def test_for_inout_example():
    something = input('Enter your text:')
    # delete garbage from input
    actualstring = ""
    for i in something:
        if i.isalpha():
            actualstring += i.lower()
    # print if input is a palindrome
    print('Actual string is', actualstring)
    if is_palindrome(actualstring):
        print('Text is palindrome!')
    else:
        print('Text is not a palindrome')

# iterators+generators exmaples
# [open] function creates a kind of generator
f = open('p7.py')
for i in range(5):
    print(f.readline(), end='')  # Every time generator returns one value (here: one string)
# next(iterator) and iterator.__next__() do the same thing
print(f.readline(), f.__next__(), next(f), sep='')

# iter() functions gets iterator
L = [1, 2, 3]
listiter = iter(L)
print(type(listiter))  # <class 'list_iterator'>
print(next(listiter, 'enough!'), listiter.__next__(), next(listiter), next(listiter, 'enough!'))
try:
    print(listiter.__next__())
except StopIteration as e:
    print('StopIteration error', e)

f = open('test.txt')
print(iter(f) is f)  # True (for file thread - they have they own iterator)
L = [1, 2, 3]
print(iter(L) is L)  # False (lists don't have their own iterators)

# Default and manual way to use iterations
L = [1, 2, 3]
for x in L:
    print(x**2, end=' ')  # 1 4 9

print('')
I = iter(L)
while True:
    try:
        x = next(I)
    except StopIteration:
        print('\nHello from StopIteration exception!')
        break
    print(x ** 2, end=' ')  # 1 4 9

# Iterators and dictionaries
D = {'a': 1, 'b': 2, 'c': 3}
dictI = iter(D)
print(type(dictI))  # <class 'dict_keyiterator'>
for key in dictI:
    print(key)  # Iterator returns keys

import os
P = os.popen('dir')
print(type(P))  # <class 'os._wrap_close'>
for i in range(5):
    print(P.__next__(), end='')

# Range function
R = range(5)
print(R, type(R))  # range(0,5) <class 'range'>
I = iter(R)
for i in R:
    print(I.__next__(), end=' ')
print('\n', list(R), sep='')

# enumerates
E = enumerate('spam eggs ham')
print(E, type(E))  # <enumerate object at ...>  <class 'enumerate'>
I = iter(E)
print(next(I), next(I))  # (0,'s') (1,'p')
print(list(enumerate('spam eggs ham')))  # a big list of tuples with associated numbers

# List generators

# Alternative for the following:
L = [1, 2, 3, 4, 5]
for i in range(len(L)):
    L[i] += 10  # list will change
print(L)  # [11, 12, 13, 14, 15]
# Alternative here:
L_v2 = [x + 10 for x in L]
print(L_v2)  # [21, 22, 23, 24, 25]

# Lists generators for working with files
f = open('datafile.txt')
lines = f.readlines()
lines = [line.rstrip() for line in lines]
print(lines)

print([line.upper() for line in open('p4.py')])
print([line.rstrip().upper() for line in open('printing.py')])

l1 = [x + y for x in 'abc' for y in 'lmn']
print(l1)

uppers = [line.upper() for line in open('log.txt')]
print(uppers)

uppers_map = map(str.upper, open('log.txt'))
print(list(uppers_map))

print(
    sorted(open('log.txt'))
)

print(
    list(zip(open('log.txt'), open('test.txt')))
)

print(
    list(enumerate(open('log.txt')))
)

print(
    list(filter(bool, open('log.txt')))
)

import functools, operator
result = functools.reduce(operator.add, open('log.txt'))
print('%r' % result)

def f(a, b, c, d):
    print(a, b, c, d, sep='_')
f(1,2,3,4)
f(*range(4))
f(*[1,2,3,4])  # Unpacking list for arguments
f(*open('log.txt'))

# Unpacking packed tuples
X = [1, 2, 3]
Y = [4, 5, 6]
A, B = zip(*zip(X, Y))
print(A, B)  # Unpacked tuples: (1,2,3) (4,5,6)

M = map(abs, (-1, 0, 1))
print(list(M))

Z = zip((1,2,3), (10,20,30))
print(list(Z))
#  print(Z.__next__())  this operation is impossible, becuase iterator has ended
Z = zip((1,2,3), (10,20,30))
for pair in Z:
    print(pair, end=' ')
else:
    print('')

F = filter(bool, ['spam', '', 'ham'])
print(list(F))  # ['spam', 'ham']

# lambda exmaples
from math import sqrt


def make_sum(x):
    return lambda: x+x
suml = make_sum(5)
print(suml())  # 10


def print_str():
    return lambda: print('hello from lambda')
printl = print_str()
printl()


def make_incrementor(n):
    return lambda x: x+n
f1 = make_incrementor(2)
print(f1(4))


def func_with_lambda(fun, args):
    fun(args)


def print_content(args):
    print('\nPrinting stuff:')
    for i in args:
        print(i())

listOfLambdas = [make_sum(x) for x in range(1, 6)]
func_with_lambda(print_content, listOfLambdas)

# Filters
list1 = [2, 3, 142, 34, 123, 451, 32465, 234, 5123, 5132, 1123, 4]
filter1 = filter(lambda x: x % 3 == 0, list1)
print(list(filter1))
mapping1 = map(lambda x: x*2 + 10, list1)
print(list(mapping1))

# Computing prime numbers - doesn't work
#nums = range(1, 100)
#for i in range(2, int(sqrt(100)) + 1):
#    nums = filter(lambda x: x != i or x % i == 0, nums)
#print(list(nums))

words = ['hi', 'second', 'first', 'fee', 'foo', 'sixth']
length = map(lambda word: len(word), words)
print(words, '\n', list(length))
filter2 = filter(lambda word: len(word) > 3, words)
print(list(filter2))

length2 = map(lambda word: len(word), ['hello', 'world', 'askdhfasdf'])
print(list(length2))

# creating a simple log file
import os, platform, logging

if platform.platform().startswith('Windows'):
    logging_file = os.path.join(os.getenv('HOMEDRIVE'), \
                                os.getenv('HOMEPATH'), \
                                'test.log')
else:
    logging_file = os.path.join(os.getenv('HOME'), 'test.log')

print('Saving .log file into', logging_file)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s : %(levelname)s : %(message)s',
    filename=logging_file,
    filemode='w'
)

logging.debug('Start of the program')
logging.info('....')
logging.warning('Death of the program')

# write and read from/to file
from sys import argv
def f1():
    errmsg = "Usage: python \"write'n'read.py\" [filename]"
    try:
        filename, content = argv[1], argv[2]
        # content = argv[2]
        # Write content to the file.
        f = open(filename, "w")
        f.write(content)
        f.close()
        # Read the file you've created.
        f = open(filename, "r")
        print(f.read())
        f.close()
    except IndexError:
        print("IndexError.", errmsg)
    except ValueError:
        print("ValueError.", errmsg)
    except IOError:
        print("IOError. Can't create file stream.")
    else:
        print('End of the program')

# with-as example + yield example
"""
Understanding with statement.

We have this code:
-------------------------------
set things up
try:
  do something
finally:
  tear things down
-------------------------------end

Code we can use instead (old approach):
-------------------------------
def my_func():
    set things up
    try:
        yield thing
    finally:
        tear things down

def main():
    for thing in my_func():
        do something with thing
-------------------------------end

Code we can use instead (new approach):
-------------------------------
class controlled_execution:
    def __enter__(self):
        set things up
        return thing
    def __exit__(self, type, ......):
        tear things down

def main():
    with controlled_execution() as thing
        # do something with [thing] object!
-------------------------------end

"""


def my_func():
    f = open('test.txt')
    try:
        yield f
    finally:
        f.close()


def example2(filepath):
    # It executes f.__enter__() at the beginning - opens file stream
    # and f.__exit__() at the end - closes file stream
    with open(filepath) as f:
        content = f.read()
        print(content)

for thing in my_func():
    print(type(thing))


# Reading
def testReading():
    errmsg = 'Usage: python reading.py [fileToRead]'
    try:
        path = argv[1]
        f = open(path, mode='r')
        print('File content: \n%r' % f.read())
        f.close()
    except ValueError:
        print(errmsg)
    except IndexError:
        print(errmsg)
    except IOError:
        print('Error reading a file.')

# printing & strams example
# With help of __future__ print_function field, now you shouldn't care
# about what version of print() function you should use!
# With the next string print() function will work in all versions of Python.
# But there is also stuff with [strings building] (like '%s %s' '{0} {1]' etc)
from __future__ import print_function

import sys

temp = sys.stdout

# 'Cool' alternative to print() function
sys.stdout.write('hi guys\n')
# Writing to files with print() function
sys.stdout = open('log.txt', 'a')
print('first log message')

# Returning to main screen after redirecting stdout
temp = sys.stdout
# Do something here...
sys.stdout = open('log.txt', 'a')
print('spam, ham, eggs, 1, 2, 3')
sys.stdout.close()
# Return to default stdout
sys.stdout = temp
print('back into business!')
print('And here is a log.txt file:', open('log.txt').read(), sep='\n')

sys.stderr.write('this is an error')  # Will be highlighted with red color!

# Fake reader
# With it we can replace stdout and stdin with our own functions and print/read smt:
# stdout - write; stdin - read
# why so: because in print() function [file] needs only [write] function
# why so 2: because in input() function [file] stdin thread needs only [read] function
class FileFaker:

    # Example:
    # print(someObjects, file=myObj) where myObj is instance of FileFaker class

    def __init__(self):
        pass

    def write(self, string):
        # Do something with string
        pass

    def read(self):
        # Do something with string
        pass

# Example of shell command setting for output and input threads:
# python script.py < inputfile > outputfile

f = open('test.txt')
print(f.read(10))

f.seek(0)
print(f.read())

f.seek(0)
lines = f.readlines()
print(lines, type(lines))

f.seek(0)
print('%r' % f.read())

print('-------')
for line in open('test.txt'):
    print(line, end='')
else:
    print('\n-------')

f.close()

# Binary things:
# note, that all text in python is in Unicode
# But there is no problem working with ASCII

# Saving python objects in files
X, Y, Z = 43, 44, 45
S = 'spam'
D = {'a': 1, 'b': 2}
L = [1, 2, 3]

F = open('datafile.txt', 'w')
F.write(S + '\n')
F.write('%s, %s, %s\n' % (X, Y, Z))
F.write(str(L) + '$' + str(D) + '\n')

F.close()

# Getting saved objects from file.
F = open('datafile.txt')
S_v2 = F.readline().rstrip()  # or F.readline()[:-1] to delete the last '\n' char
X_v2, Y_v2, Z_v2 = [int(num) for num in F.readline().split(', ')]
L_v2, D_v2 = [eval(obj) for obj in F.readline().split('$')]

print(S_v2, type(S_v2))
print(X_v2, Y_v2, Z_v2, type(X_v2), type(Y_v2), type(Z_v2))
print(L_v2, type(L_v2))
print(D_v2, type(D_v2))

# Manager of file contexts
# Usually they are using for exceptions handling,
# also here it guarantees that after block file will be closed automatically.
with open('test.txt') as myfile:
    # here [myfile] object can be used
    for line in myfile:
        print(line, end='')  # prints context of the file
print('\n')

# But here can be used a block [try/finally]
try:
    myfile = open('test.txt')
    for line in myfile:
        print(line, end='')
finally:
    myfile.close()
    print('')
