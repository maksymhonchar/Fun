# Only in Python 3.X :
# old-style creating dictionaries with [zip] function
d_dict1 = dict(
    zip(
        ['a', 'b', 'c'],
        [1, 2, 3]
    )
)
print(d_dict1)  # {'a':1, 'b':2, 'c':3}
# new-style by creating dictionaries: generators
d_dict2 = {k: v for (k, v) in zip(['a', 'b', 'c'], [1, 2, 3])}
print(d_dict2)  # same as previous dict initialization
d_dict3 = {x: x ** 2 for x in [1, 2, 3, 4]}
print(d_dict3)
d_dict4 = {c.lower(): c + '!' for c in ['HELLO', 'GUYS', 'EGGS']}
print(d_dict4)

# fromkeys() and other things that do the same things
d_dict5 = dict.fromkeys(['a', 'b', 'c'], -1)
print(d_dict5)
d_dict5 = dict.fromkeys('Hello')
print(d_dict5)
# analog of fromkeys() func
d_dict6 = {k: -1 for k in ['a', 'b', 'c']}
print(d_dict6)
d_dict6 = {k: None for k in 'Hello'}
print(d_dict6)

# Checking equality of two dictionaries
if d_dict6 == d_dict5:
    print('woaw')
else:
    print('fuqu')

# Iterating through the dictionary
iter1 = iter(d_dict6)
print(type(iter1), list(iter1))  # iterator lists only keys.
try:
    print(iter1[0])  # Error - this is not a list
except TypeError as e:
    print('TypeError:', e)
# Now you can't use iter1, because it 'ended'
for k in iter(d_dict6):
    print(k, end=' ')
else:
    print('')

# Going through the dictionary
for k in d_dict6.keys():
    print(k, end=' ')
else:
    print('')
# The same thing
for k in d_dict6:
    print(k, end=' ')
else:
    print('')

# Iterators save all future changes in dictionaries
D = {'a': 1, 'b': 2, 'c': 3}
K = D.keys()
V = D.values()
print(list(K), list(V))
del D['a']
print(list(K), list(V))  # See changes here.

# Some logical operations with keys
print(K | {'x': 4}, type(K | {'x': 4}))  # {'b', 'c', 'x'} <class 'set'>
try:
    print(V & {'x': 4}.values())  # Can't do things with VALUES
    print(V & {'x': 4})  # This is also impossible, because here VALUES are compared
except TypeError as e:
    print('TypeError:', e)
D = {'a': 1, 'b': 2, 'c': 3}
print(D.keys() & D.keys())  # keys as a set
print(D.keys() & {'b'})  # set {'b'}
print(D.keys() & {'b': 1})  # set {'b'}

# If items (or values, or keys) contain only things
# that can't be changed, then they have <class 'set'> setting
D = {'a': 1}
print(D.items() | D.keys())

# Sorting dictionary:
# There are three ways to do that:
# First way to sort keys:
D = {'c': 1, 'b': 2, 'a': 3}
print(sorted(D))
print(type(sorted(D)))  # list
# Second way to sort keys:
# 1)cast to list
# 2)sort
# 3)go through dict with sorted keys
fromDict = list(D)
fromDict.sort()
print('Sorted dictionary:')
for k in fromDict:
    print(k, '-', D[k], sep=' ')
# Third way to sort keys:
# use sorted() default function
print('Sorted dict:')
D = {'c': 1, 'b': 2, 'a': 3}
for k in sorted(D):
    print(k, '-', D[k])

# This thing doesn't work, because in Python 3.X .keys() method returns iterator.
try:
    Ks = D.keys()
    print(type(D.keys()))
    Ks.sort()  # Cannot sort iterator (or <class 'dict_keys'> !
except AttributeError as e:
    print('AttributeError:', e)

# dict.get function
branch = {
    'spam': 1,
    'ham': 2,
    'eggs': 3
}
print(branch.get('spam', 'Bad choice'))  # second argument - default value
print(branch.get('bacon', 'Bad choice'))
# Alternative to dict.get function
choice = 'lul'
if choice in branch:
    print('found! Here it is:', branch[choice])
else:
    print(choice, 'does not exist here.')

# Dictionaries can be used as [switch-case] alternative:
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

# Easy dictionary example
print(stuff['name'])
stuff['name'] = 'Maxim'
print(stuff['name'])

stuff[3] = 'TmpName'
stuff[4] = 'TmpName2'

print(stuff[3])
print(stuff[4])

print(stuff)

del(stuff[3])
del(stuff[4])

print(stuff)

# Another one dictionary exmaple
dict = {'name': 'Maxim',
        'surname': 'Gonchar',
        }

print(dict)
print(len(dict))
print(dict['name'])
print(dict['surname'])

for name, address in dict.items():
    print(type(name))
    print(type(address))
    print(type(dict.items()))
    print('{0} {1}'.format(name, address))

dict['mail'] = 'maxgonchar9@gmail.com'
print(dict['mail'])

if 'asdfasd' not in dict:
    print("wow")
else:
    print('omg')

myList = ['asdxf', 'asdf', 'asdf']
myList2 = myList.copy()

dict = {
    'name': 'Maxim',
    'surname': 'Gonchar',
    'age': 17
}
print('%(name)s %(surname)s %(age)d' % dict)

s = repr(dict)
s2 = str(dict)
print(s)
print(s2)

target = 'q'
x = 'hi guys this is a \'q\' character!'

# Comparing two dicionaries
d1 = {'a': 1, 'b': 2}
d2 = {'a': 1, 'b': 3}

print(d1.__gt__(d2))  # Not implemented

# To get [greater] or [less] result, use lists cast:
l1 = list(d1.items())
l2 = list(d2.items())

lessresult = sorted(l1) < sorted(l2)
greaterresult = sorted(l1) > sorted(l2)

print(lessresult, greaterresult)  # True False

# or use the following:
print(sorted(d1.items()) < sorted(d2.items()))  # True
print(sorted(d1.items()) > sorted(d2.items()))  # False

D = {'food': 'cat', 'race': 'human'}
print(D.keys(), D.values(), D.items())

try:
    print(D['not exists'])  # Error
except KeyError as e:
    print('KeyError:', e, 'does not exist in dictionary.')

print(D.get('not exists'))  # None
print(D.get('lul', -1))  # -1

try:
    print(D.pop('lul'))
except KeyError as e:
    print('KeyError:', e, 'does not exist in dictionary.')

langs = {
    'Python': 'Guido van Rossum',
    'Perl': 'Lary Wall',
    'Tcl': 'John Ousterhout'
}
language = 'Python'
creator = langs[language]
print(creator)
for lang in langs:  # for [key] in [dict]
    print(lang, '\t', langs[lang])
else:
    print('This was a list of programming languages')

# To create something to replace this: a = []; a[99] = 123  # error
# Do the following:
# INSTEAD of this:
a = [0] * 100
a[99] = 'test'
# USE dict:
a = {}
a[99] = 'test'

# Another example to create 'half-empty' arrays (here: dicts)
matrix1 = {}
matrix1[(2, 3, 4)] = 100
matrix1[(5, 6, 7)] = 200
X = 2; Y = 3; Z = 4
print(matrix1[X, Y, Z])

# HOW TO handle error by getting non existing element in dict by key:
# First approach: 'if' check
if (2,3,4) in matrix1:
    print(matrix1[(2,3,4)])
else:
    print('Oh no, point (2,3,4) does not exist in matrix1')
# Second approach: 'try-except' block
try:
    print(matrix1[(1,2,4)])
except KeyError as e:
    print('KeyError:', e, 'does not exist in matrix1 obj')
# Third approach: dict 'get' method
print(matrix1.get((5, 6, 8), 'fuck you'))  # 'fuck you'

d2 = dict([('name', 'max'), ('age', 18)])
print(d2)

# Keys can be only strings in this method
d3 = dict(name='max2', surname='gonchar2')
print(list(d3.keys()), list(d3.values()), list(d3.items()), sep='\n')

d4 = dict()  # alternative: d4 = {}
d4['first'] = 1
d4['second'] = 2
print(d4)

# [zip] method examples
keys_list = ['first', 'second', 'third', 'fourth']
values_list = [1, 2, 3, 4]
d5 = dict(zip(keys_list, values_list))
print(d5)

# Easy initialization of the dictionary,
# if all keys are in dict should be the same.
d6 = dict.fromkeys(['a', 'b'], 0)  # 0 means default value to fill
print(d6)

# Going through the dictionary
L = [1, 2, 3]
D = {'a': L, 'b': L}
for key in D:
    print(key, '=>', D[key])

for (key, value) in D.items():
    print(key, '=>', value)

# Constructing dictionary with [zip]
keys = ['spam', 'eggs', 'ham']
values = [1, 2, 3]
resulting_dict = {}
for (k, v) in zip(keys, values):
    resulting_dict[k] = v
print(resulting_dict)

# Dictionaries have their own iterators, that return their keys.
# dict.keys() method also returns an iterator
D = dict(a=1, b=2, c=3)
K = D.keys()
#  print(next(K))  Error: 'dict_keys' object is not an iterator
I = iter(K)
print(next(I), next(I), next(I))
for k in D:
    print(k, end=' ')

# Dictionaries examples
rec = {12: 12}
rec = 0  # Dictionary memory is released

dict_1 = {
    'a': 1,
    'b': 2,
    'c': 3
}
print(dict_1)
Ks = dict_1.keys()
print(type(Ks))
for key in dict_1.keys():
    print(key, end=' ')
Ks_list = str(dict_1.keys())
print(Ks_list)
# Doesn't work following: list(dict_1.keys())

# print(type(Ks_list), Ks_list)
# dictt = {'Name': 'Zara', 'Age': 7}
# print("Value : %s" % dictt.keys())

# Regular printing dictionary key-value pairs
print('Key => Value in dict_1:')
for key in dict_1:
    print(key, '=>', dict_1[key])

# Sort by keys with 'for' loop
print('Sorted key-value pair in dict_1:')
for key in sorted(dict_1):
    print(key, '=>', dict_1[key])

key_to_find = 'a'
if key_to_find in dict_1:
    print('found:', dict_1[key_to_find])
else:
    print('missing')

if dict_1.get(key_to_find):
    print('Found:', dict_1[key_to_find])

value = dict_1['b'] if 'b' in dict_1 else -1
print('value:', value)
