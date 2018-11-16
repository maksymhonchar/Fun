"""
FizzBuzz implementations I could imagine so far.

Assuming 'num' argument is always a correct positive number.

I also know an implementation with flags but it makes me sick, hate it.
"""

def impl_1(num): 
    '''if-else with readable expressions'''
    for i in range(1,num):
        div_by_3 = (i % 3) == 0  # not _expr_
        div_by_5 = (i % 5) == 0  # not _expr_
        if div_by_3 and div_by_5:
            print(i, 'FizzBuzz')
        elif div_by_3:
            print(i, 'Fizz')
        elif div_by_5:
            print(i, 'Buzz')
        else:
            print(i)

def impl_2(num):
    '''Basic if-else'''
    for i in range(1, num):
        if not (i % 3) and not (i % 5):
            print(i, 'FizzBuzz')
        elif not (i % 3):
            print(i, 'Buzz')
        elif not (i % 5):
            print(i, 'Fizz')
        else:
            print(i)

def impl_3(num):
    '''One line list comprehension'''
    values = list(range(1, num))
    values = [
        'FizzBuzz' if not (value % 3) and not (value % 5)
        else 'Fizz' if not (value % 3)
        else 'Buzz' if not (value % 5)
        else value
        for value in values 
    ]
    print('\n'.join([str(value) for value in values]))

def impl_4(num):
    '''map function'''
    values = list( 
        map(
            lambda x: 'FizzBuzz' if not (x % 3) and not (x % 5) else 'Fizz' if not (x % 3) else 'Buzz' if not (x % 5) else x, 
            range(1, num)
        )
    )
    print('\n'.join([str(value) for value in values]))

def impl_5(num):
    '''finding with filter. dummy impl'''
    values = list(range(1, num))
    div_by_3 = list(filter(lambda x: not (x % 3), values))
    div_by_5 = list(filter(lambda x: not (x % 5), values))
    div_by_3_5 = list(filter(lambda x: x in div_by_3 and x in div_by_5, values))
    for index, value in enumerate(values):
        val_to_change = value
        if value in div_by_3_5:
            val_to_change = 'FizzBuzz'
        elif value in div_by_3:
            val_to_change = 'Fizz'
        elif value in div_by_5:
            val_to_change = 'Buzz'
        values[index] = val_to_change
    print('\n'.join([str(value) for value in values]))
        
def impl_6(num):
    '''a simple generator'''
    def fizzbuzz_generator(num):
        for i in range(1, num):
            if not (i % 3) and not (i % 5):
                yield 'FizzBuzz'
            elif not (i % 3):
                yield 'Fizz'
            elif not (i % 5):
                yield 'Buzz'
            else:
                yield i
    print('\n'.join([str(value) for value in fizzbuzz_generator(num)]))

def impl_7(num):
    '''strings concat'''
    values = [ ( 'Fizz'*(i%3==0) + 'Buzz'*(i%5==0) ) or i for i in list(range(1, num)) ]
    

def impl_8(num):
    '''kind of lazy itertools cycling.'''
    from itertools import cycle, count, islice
    div_by_3 = islice(cycle([""] * 2 + ["Fizz"]), 20)
    div_by_5 = islice(cycle([""] * 4 + ["Buzz"]), 20)
    div_by_3_5 = (a + b for a,b in zip(div_by_3, div_by_5))  # zipping bc of div_by_3 and div_by_5 are ordered
    fizzbuzz_generator = (word or number for word, number in zip(div_by_3_5, count(1)))
    print('\n'.join([str(value) for value in list(islice(fizzbuzz_generator, num))]))


if __name__ == '__main__':
    # impl_X(20)
    pass
