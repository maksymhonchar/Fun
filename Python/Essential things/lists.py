a = [1, 'str1', 'str2', 12, ('tuple1', 'tuple2')]
print('%r' % a)
print('%r' % enumerate(a))
for i, item in enumerate(a):
    print('Index: {index} Item: {item}'.format(index=i, item=item))

b = [5, 'str', range(6)]
print(b)
print(type(b[2]))
for item in b:
    print(item)
    if isinstance(item, range):
        for i in item:
            print(i)

print('------ ------')
c = [15,]
c.extend(range(5))
for i in c:
    print(i)

print('------------')
d = [20,]
d[len(d):] = range(5)
for i in d:
    print(i)

print('------------')
squares = [x**2 for x in range(10)]
print(squares)

mylist = ['asdf', 'hiii', 'qert']

print('Pokupki:', end=' ')
for item in mylist:
    print(item, end=' ')

mylist.append('newItem')
print('\nPokupki v2:', end=' ')
for item in mylist:
    print(item, end=' ')

mylist.sort()
print()
for item in mylist:
    print(item, end=' ')

print('')
print('Size of list in butes:', mylist.__sizeof__())

# Some list examples
def prmatr(matr):
    print('')
    for i in matr:
        print(i)

def cmp_types(obj):
    if isinstance(obj, int):
        return -1
    else:
        return 1

a = []
a.append(id(a))
print(type(a[0]), a[0])
a.append(id(a[0]))
print(type(a[1]), a[1])

b = [1,2,3,4,5]
c = b.copy()
del(c[2])
print(b, c)

d = []
e = list()
print(d, e)

rep = list(['rep'] * 5)
rep_v2 = ['rep'] * 5
print(rep, rep_v2, sep='\n')

try:
    f = [1, 2, 3] + {'val': 'key'}
except TypeError as e:
    print('TypeError:', e)  # This will be executed

g = [1,2,3,4]
s1 = str(g) + '_hi'
print(s1)  # [1, 2, 3, 4]_hi

q = [1, 2] + list("hi")
print(q)
for i in q:
    print(type(i), end=' ')  # int int str str
else:
    print('')

w = list(map(abs, [-1, -2, 0, 1, 2]))
print(w)  # [1, 2, 0, 1, 2]

s2 = 'hi guys'
print(id(s2[0]), id(s2[1]))  # wtf

matr1 = [
    [1, 2],
    [3, 4]
]
print(type(matr1[0]))  # list, not string

e = [1, 2, 3, 4, 5]
e[1:4] = ['hi', 'guys']  # [1, 'hi', 'guys', 5] - `eats` elements
print(e)

# Deleting slices from a list
r = [1, 2, 3, 4, 5]
r[1:3] = []
print(r)  # [1, 4, 5]

matr2 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
for i in matr2:
    i[1:2] = []
    # if it will be like i[1]=[], then an empty list will add
prmatr(matr2)

matr3 = matr2.copy()
for i in matr3:
    del(i[1])
prmatr(matr3)

l1 = [1, 2]
l1.append(3)
l1[len(l1):] = [4]
print('\n', l1, sep='')

# Sorting examples
l2 = ['a', 'b', 'c', 'A', 'B', 'C']
l2.sort(reverse=True, key=str.upper)
print(l2)

# Sorting integers and strings
l3 = ['first', 123, 'second', 123, 'third', 3, 2, 'wassup', 1]
l3.sort(key=cmp_types)
print(l3)  # first - integers, then - strings and everything else

d1 = {
    'a':1,
    'b':2,
    'c':3,
    'd':4
}
print(sorted(d1))  # sorted by keys

l4 = [1, 2, 3]
l4.extend([1])
print(l4)
l4.pop()
l4.reverse()
print(l4)
print(list(reversed(l4)))  # [1, 2, 3]

try:
    l4.remove('i don\'t exist')
except ValueError as e:
    print('ValueError:', e)

# wtf
t = [1]
print(t, type(t))
tt = [1, ]
print(tt, type(tt))

# Assigning trap #1
list1 = ['yo', 'nigga']
complexlist = [1, ('hi', 'guys', list1), 3]
list2 = complexlist[:]
print(complexlist, list2, sep='\n')

list1[0] = 'i love you'
print(complexlist, list2, sep='\n')  # 'i love you' will appear in both lists, cause of non-deep copy.

# Assigning trap #2
L = [4, 5, 6]
X = L * 4  # equals to [4,5,6]+[4,5,6]+...
Y = [L] * 4  # equlas to [[4,5,6]]+[[4,5,6]]+... = [L]+[L]+... = [L,L,L...]
print(X, Y, sep='\n')
# trap is in following:
L[1] = 0
print(X, Y, sep='\n')  # Elements in [Y] will change, because that were links (pointers) to L object

# Concatenation trap
L = [1, 2]
M = L  # L and M now referring to the same object
L = L + [3, 4]  # Concatenation creates a NEW object
print(L, M)  # L changed, M didn't change

L = [1, 2]
M = L
L += [3, 4]  # Operation [+=] doesn't create a new object
print(L, M)  # Both L and M changed

# Some more lists examples
S = 'ham'
print(S[:0])  # no symbol

L = [1, 2, 3] + [4, 5, 6]
print(L[:0])  # empty list
print(L[-2:])

L = [0, 1, 2, 3]
L[3:1] = '?'
print(L)

L = []
L2 = [1, 2, 3, 4]
L2[2] = []  # assigning
print(L2)

L2[2:3] = []  # deleting
print(L2)

del L2[1]
print(L2)

L2.extend([1, 2, 3, 4])
del L2[2:4]
print(L2)

L2[2:3] = (1, 2)
print(L2)

D = {'a': 1, 'b': 2, 'c': 3}
# print(D['d'])  # KeyError

a = b = []
a.append(44)  # changes will be also appear in b
print(a, b)

# Copying lists
l1 = [1, 2, 3, 4]
l1_copy = l1[:]
l1_copy2 = l1.copy()
import copy
l1_copy3 = copy.copy(l1)

# Random things
import types

list1 = [42, 'heart', lambda x: x**2, [47, '11']]

print(list1)
print(type(list1))

print(list1[2](3))
print(type(list1[2]))
print(type(list1[2](3)))

print(list1[3])
print(type(list1[3]))

print(type(types))

print(type(type(1)))

# Other stuff
S = [x**2 for x in range(20)]
V = [2**i for i in range(13)]
M = [x for x in S if x % 2 == 0]
print(S, V, M, sep='\n')

# List of prime numbers
noprimes = [j for i in range(2, 8) for j in range(i*2, 50, i)]
primes = [x for x in range(2, 50) if x not in noprimes]
primesneg = [(-1)*x for x in range(2,50) if x not in noprimes]
print('', noprimes, primes, primesneg, sep='\n')

nestedCompr = [x for x in range(2, 50) if x not in [j for i in range(2, 8) for j in range(i*2, 50, i)]]
print(nestedCompr)

words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
stuff = [[len(w), w[0], w] for w in words]
print(stuff)

stuff2 = [[w.upper(), w.lower(), len(w)] for w in words]  # same as stuff.copy
stuff3 = map(lambda w: [w.upper(), w.lower(), len(w)], words)
print(list(stuff3))

# Random stuff
friends = ['first', 'second', 'third', 'fourth']
print(type(friends))
print(enumerate(friends))
for i, name in enumerate(friends):
    print(type(i), " ", type(name))
    print('Iteration {iteration} value {name}'.format(iteration=i, name=name))

# Creating list from user input
list = eval(input("Enter a list "))
print(list)
print(type(list))

x,y = eval(input("Enter two integers: "))
print(x, " ", y)

# Another one example

testlist = range(6)
testlist2 = []
for item in reversed(testlist):
    testlist2.append(item)
print(testlist2)

# Changing list while being in loop
L = [1, 2, 3, 4, 5]
for x in L:
    x += 1  # incorrect, L didn't change
print(L)  # [1, 2, 3, 4, 5]
# Correct version:
L = [1, 2, 3, 4, 5]
for x in range(len(L)):
    L[x] += 1
print(L)  # [2, 3, 4, 5, 6]
# Version, using a generator
L = [x+1 for x in L]
print(L)

list_with_ten_objects = [None] * 10
print(list_with_ten_objects)

# Changong list, using a generator
L = [x+1 for x in L]
print(L)

# old-style creating lists with [zip] function
l_list1 = list(
    zip(
        ['a', 'b', 'c'],
        [1, 2, 3]
    )
)
print(l_list1)  # [('a', 1), ('b',2), ('c', 3)]
print(type(l_list1[0]))  # <class 'tuple'>
