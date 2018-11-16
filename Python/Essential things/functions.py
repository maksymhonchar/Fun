def f1():
    print('hi')
print(type(f1))  # <class 'function'>

# Calling function from a linked object.
a = f1
print(type(a))  # <class 'function'>

a = None
if a:
    def f2():
        pass
else:
    print('yo')
try:
    f2()
except NameError as e:
    print('NameError:', e)

# Add an attribute to function object.
# Note, that [def] instruction creates an object.
def f3(a):
    print(a)
    pass
f3.myattr = 10
print(f3.myattr)  # 10
f3.myattr += 1
print(f3.myattr)  # 11
print('myattr' in dir(f3))  # True

attr = 1
def times(x):
    times.myattr = attr
    return x * times.myattr
res = times(2)
print(res)  # 2
attr += 1
res = times(2)
print(res)  # 4

# print out function arguments via reflexion (does not work)
import inspect, itertools
def getarguments():
    def decorator(f):
        def wrapper(*args):
            # get arguments names as a list
            args_names = inspect.getargspec(f)[0]
            # get values as a list.
            args_dict = dict(itertools.izip(args_names, args))
            args_values = args_dict.values()
            print(args_names, args_values)

# @getarguments()  # does not work
# def my_function(x, y, z):
#    pass
# my_function(1, y=1)

def mymethod(x, y, z):
    pass
print(inspect.getargspec(mymethod))  # getargspec method is deprecated
print(inspect.signature(mymethod))  # (x, y, z)

def mymethod2(x, y):
    print(locals())
mymethod2(1, 2)  # {'x': 1, 'y': 2}


# Polymorphism in Python
# Sense of operation (here: function) depends on types of processed arguments.

# Example 2: sets intersection
a = '123'
b = '133456'
def findsame(a, b):
    return list(set(a) & set(b))
print(findsame(a, b))

def findsame_y(a, b):
    for char in a:
        if char in b:
            yield char
    else:
        yield 'before_end'
    yield 'end'
for item in findsame_y(a, b):
    print(item, end=' ')  # 1 3 before_end end
else:
    print('')
# same result can also be reached just with simple [x for x in a if x in b]
# same result, cause of polymorphism:
x = findsame([1, 2, 3], (3, 4))
print(x)  # [3]

# but here will be an error:
try:
    x = findsame(1, 2)
except TypeError as e:
    print('TypeError:', e)

# scope of the variables
x = 99
def func():
    global x
    x += 1
print(x)  # 99
func()
print(x)  # 100
class a():
    attr = 0
    def func(self):
        global attr
        attr += 1
    def pr_attr(self):
        print(attr)
aobj = a()
aobj.pr_attr()  # 2
aobj.func()
aobj.pr_attr()  # 3

def a():
    global aobj
    print(type(aobj))
a()  # <class '__main__.a'>

# Another example of functions scopes
# RULE: LEGB: local-enclosed-global-bultins
# Global scope
X = 99
def func_example(Y):
    # Local scope
    Z = X + Y
    return Z
print(func_example(1))  # 100

import builtins  # in Python 2.X it's a __builtin__
print(dir(builtins))  # preordered names C
builtins.print('hi guyz')

def hider():
    open = 'eggs'
    ...
    open('data.txt')  # 'eggs' str hides builtin 'open' function.
try:
    hider()  # error: 'str' object is not callable
except TypeError as e:
    print('TypeError:', e)

def hider_fuckyou():
    open = 'eggs'
    ...
    f = builtins.open('test.txt', 'w')
    f.write('fuck you, hider!')
    f.seek(0)  # f.close()
    print('result is:', builtins.open('test.txt').read())
hider_fuckyou()

"""
lul in Python 2.6. Invokes SyntaxError an error in Python 3.X, though.
True = False
print(True == False)  # True
"""

res = 0
y, z = 1, 2
def all_global():
    global res
    res = z + y
all_global()
print(res)  # 3

""" [import functions as fff] is just terrible.
var_to_change = 0
def change_var():
    import functions as fff
    fff.var_to_change += 1
change_var()
print(var_to_change)  # 1 0  ?!?
"""

# Examples for [E] variables scope
X = 99
def f1():
    X = 88
    def f2():
        print(X)
    f2()
f1()  # 88

def f1_2():
    X = 88
    def f2_2():
        print(X)
    return f2_2
action = f1_2()
# Here [f2_2] function will remember [X] variable.
action()  # 88

def maker(N):
    def action(X):
        return X ** N
    return action
res = maker(2)
print(res, type(res))  # <function action at ...> <class 'function'>
# Now, with [res] function maker, we can use action [2 ** N]
# The inner function will remember a number two, even that the maker functions ended.
print(res(2), res(4))  # 4 16

# Not like in C!
def f1():
    x = 'hello from f1()'
    f2(x)
def f2(x):
    print(x)
f1()  # 'hello from f1()'

# lambda expressions example
def func():
    x = 4
    action = (lambda n, x=x: x ** n)
    return action

def f1():
    x = 99
    def f2():
        def f3():
            print(x)
        f3()
    f2()
f1()  # 99 - interpretator will look for var in ALL def local scopes

# Some of [nonlocal] examples
def tester(start):
    state = start
    def nested(label):
        print(label, state)
    return nested
F = tester(0)
F('wassup')  # wassup 0
F = tester(5)
F('ham')  # ham 5

# With nonlocal result is different
def tester_2(start):
    state = start
    def nested_2(label):
        nonlocal state  # now we can change the [state] var above
        print(label, state)
        state += 1
    return nested_2
F = tester_2(0)
for i in range(3):
    F('msg')  # msg 0\nmsg 1\nmsg 2
else:
    print('')

class G:
    """dummy class"""
    pass

G.x = 6
G.y = 7
print(vars(G))  # list of variables

g = G()
print(vars(g))  # {}
g.first = 1
g.second = 2
print(vars(g))  # {'second': 2, 'first': 1}

# Adding function to the class
def a():
    print('hello, guys!')
G.dosomething = a
G.dosomething()
# Another way to do this
class tst(object):
    def __init__(self):
        self.testvar = 5
def tst_def(self):
    print("testvar is %d" % self.testvar)
tst.tst_def = tst_def
tst_object = tst()
tst_object.tst_def()  # testvar is 5
# Adding function object to an instance of a class
#import types
#f = types.MethodType(tst_def, tst_object, tst)
#tst_object.tst_def_v2 = f
#tst_object.tst_def_v2()  # testvar is 5


print(g.__dict__)  # print all class members
g.__dict__['first'] = 11
print(g.first)  # 11

# New Style Class
class NewStyleEgg(object):
    def __init__(self):
        self.__egg = "MyEgg"

    @property
    def egg(self):
        return self.__egg

    @egg.setter
    def egg(self, egg):
        self.__egg = egg

    @staticmethod
    def static_method():
        print('hi from static method')

# Stuff with inheritance
class a1(object):
    def f1(self):
        print('hi from f1 at a1 class')
    x = 10

class a2(a1):
    def f1(self):
        print('hi from f1 at a2 class')
        a1.f1(self)
    y = 9

a2_obj = a2()
a2.f1(a2_obj)  # hi from f1 at a2 class\nhi from f1 at a1 class
print(a2_obj.x, a2_obj.y)  # 10 9

print(vars(a1), vars(a2), sep='\n')

print(isinstance(a1, a2), issubclass(a2, a1))  # False True


class Unchangable(object):
    def __setattr__(self, key, value):
        print('Nice try')
uobj = Unchangable()
uobj.x = 12  # Nice try


# Exception in file
try:
    the_file = open('i_dont_exist.txt')
except IOError as e:
    print(e.args)  # (2, 'No such file or directory)

# Custom exceptions
class CustomException(Exception):
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)
# Using this custom exception
try:
    raise CustomException('An useful error message here!')
except CustomException as instance:
    print('Caught this:', instance.parameter)

