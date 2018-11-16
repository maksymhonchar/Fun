from abc import *  # for [@abstractmethod] and [ABCMeta]


# Example for introduction to classes
class Worker(object):
    """Class to describe an usual worker"""

    def __init__(self, name, payment):
        self.name = name
        self.payment = payment

    def last_name(self):
        return self.name.split()[-1]

    def give_raise(self, percent):
        self.payment *= (1.0 + percent/100)


def worker_test():
    w = Worker('Maxim Gonchar', 10000)
    print(w.name, w.payment)
    print(w.last_name())
    w.give_raise(50)
    print(w.payment)


# A simple class, that represents a Robot.
class Robot:
    """
    Represents robot with name
    """
    __myAmazingVar = 1
    population = 0

    def __init__(self, name=None):
        self.name = name
        print('Initialization of {0}'.format(self.name))
        Robot.population += 1

    def __del__(self):
        Robot.population -= 1
        print('Deleting robot', self.name)
        if Robot.population == 0:
            print('This robot was the last one...')
        else:
            print('There are {0:d} robots left.'.format(Robot.population))

    def say_hi(self):
        print('Hi! My name is {0}'.format(self.name))
        print('Amazing var:', Robot.__myAmazingVar)

    def how_many():
        print('There are {0:d} robots in angar.'.format(Robot.population))
    how_many = staticmethod(how_many)  # Or add a @staticmethod decorator


def robot_test():
    droid1 = Robot('firstRobot')
    droid1.say_hi()
    droid1.how_many()
    print('Some work...')
    print(Robot.__doc__)


# Inheritance example
class SchoolMember(metaclass=ABCMeta):
    """Represents anyone in school """

    def __init__(self, name, age):
        self.name = name
        self.age = age
        print('SchoolMember created: [{0}]'.format(self.name))

    @abstractmethod
    def tell(self):
        print('Name: {0}. Age{1:d}.'.format(self.name, self.age), end=" ")


class Teacher(SchoolMember):
    """Represents a teacher"""

    def __init__(self, name, age, salary):
        SchoolMember.__init__(self, name, age)
        self.salary = salary
        print('Teacher created: [{0}]'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Salary: {0:d}'.format(self.salary))


class Student(SchoolMember):
    """Represents a student"""

    def __init__(self, name, age, marks):
        SchoolMember.__init__(self, name, age)
        self.marks = marks
        print('Student created: {0}'.format(self.name))

    def tell(self):
        SchoolMember.tell(self)
        print('Оценки: "{0:d}"'.format(self.marks))


def test_inheritance():
    t = Teacher('Mrs. Test', 25, 100000)
    s = Student('Test', 25, 75)
    print()
    members = [t, s]
    for member in members:
        member.tell()

class BankAccount_v2():
    def __init__(self):
        self.balance = 0
    def withdraw(self, amount):
        self.balance -= amount
        return self.balance
    def deposit(self, amount):
        self.balance += amount
        return self.balance

class A:
    def f(self):
        return self.g()

    def g(self):
        return 'A'

class B(A):
    def g(self):
        return 'B'

a = A()
b = B()
print(a.f(), b.f())  # A B
print(a.g(), b.g())  # A B


# Example of iterator
class yrange:
    def __init__(self, n):
        self.i = 0
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        else:
            raise StopIteration

# Or do it as two objects
class zrange:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return zrange_iter(self.n)

class zrange_iter:
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __iter__(self):
        # Iterators are iterables too.
        # Adding this functions to make them so.
        return self

    def next(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        else:
            raise StopIteration()

