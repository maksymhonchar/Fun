zoo = ('python', 'penguin', 'elephant', 'monkey')
print('Pythons in zoo:', zoo.count('python'))
print(zoo)

print(zoo[0])
newzoo = zoo, 'animal1', 'animal2'
print(newzoo)
print('Size of tuple in bytes:', newzoo.__sizeof__())

# Empty tuple
emptyTuple = ()
tupleWithOneElement = (2,)

print(emptyTuple, end=' ')
print(type(emptyTuple))
print(tupleWithOneElement, end=' ')
print(type(tupleWithOneElement))

combined = emptyTuple, tupleWithOneElement

print(combined, end=' ')

# Changing tuple from inside
b = (4, 5, 6)
c = list(b)
c[0] = 0
b = tuple(c)

b_v2 = (0, ) + b[1:]

print(b, type(b))
print(b_v2, type(b_v2))

# Random stuff

parents, babies = 1, 1
print(parents, babies)
print(type(parents), type(babies))
parents2, babies2 = (1, 1)
print(parents2, babies2)
print(type(parents2), type(babies2))

t1 = ('one', 'two', ('three', 'four'))
print(t1[2], type(t1[2]))

t_empty = ()

t_oneelem_wrong = (1)  # integer
t_oneelem = (1, )
print(t_oneelem, type(t_oneelem))
t_oneelem_v2 = 1,  # Parentheses aren't necessary
print(t_oneelem_v2, type(t_oneelem_v2))  # same result

t2 = tuple('hello')
print(t2)  # ('h', 'e', 'l', 'l', 'o')

print(t2.index('l'), t2.index('l', 3), t2.count('l'))  # 2 3 2

# Sorting a tuple
T = ('c', 'a', 'b', 'q', 'd')
tmplist = list(T)
tmplist.sort()
T = tuple(tmplist)
print(T)
# Sorting a tuple v2
T_2 = ('c', 'a', 'b', 'q', 'd')
print(sorted(T_2), type(sorted(T_2)))  # sortedList + list

T2 = (1, [2, 3], 4)
try:
    T2[1] = 'eggs'  # Can't change tuples elements
    print(T2)
except TypeError as e:
    print('TypeError:', e)
# But! You can change elements, that are changeable!
T2[1][0] = 'spam'
print(T2)  # (1, ['spam', 3], 4)

tuple_1 = (1,2,3,56,2,352,35,13,4,4,4,4)
print(tuple_1.count(4))  # 4
try:
    print(tuple_1.index(123))  # ValueError
except ValueError:
    print('Value error')
print(tuple_1.index(56))  # 3
