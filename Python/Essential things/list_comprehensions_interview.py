# Question src:
# https://www.sanfoundry.com/python-questions-answers-list-comprehension/
# https://www.sanfoundry.com/python-questions-answers-list-comprehension-1/
# https://www.sanfoundry.com/python-questions-answers-list-comprehension-2/
# https://www.sanfoundry.com/python-questions-answers-matrix-list-comprehension/


def q1():
    # 1. What is the output of the code shown?
    l = [1, 2, 3, 4, 5]
    print([x & 1 for x in l])
q1()

def q2():
    # 2. What is the output of the code shown below?
    l1 = [1, 2, 3]
    l2 = [4, 5, 6]
    print([x*y for x in l1 for y in l2])
q2()

def q3():
    # Write the list comprehension to pick out only negative integers from a given list ‘l’.
    l = [-1, 2, -3, 4, -5, 6, -7]
    negatives = [neg_item for neg_item in l if neg_item < 0]
    print(negatives)
q3()

def q4():
    # 4. What is the output of the code shown?
    s = ["pune", "mumbai", "delhi"]
    print([(w.upper(), len(w)) for w in s])
q4()

def q5():
    # 5. What is the output of the code shown below?
    l1 = [2, 4, 6]
    l2 = [-2, -4, -6]
    for i in zip(l1, l2):
        print(i)
q5()

def q6():
    # 6. What is the output of the following code?
    l1 = [10, 20, 30]
    l2 = [-10, -20, -30]
    l3 = [x+y for x, y in zip(l1, l2)]
    print(l3)
q6()

def q7():
    # 7. Write a list comprehension for number and its cube for l=[1, 2, 3, 4, 5, 6, 7, 8, 9].
    l=[1, 2, 3, 4, 5, 6, 7, 8, 9]
    cubes = [item**3 for item in l]
    print(cubes)
q7()

def q8():
    # 8. What is the output of the code shown below?
    l = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print([[row[i] for row in l] for i in range(3)])
q8()

def q9():
    # 9. What is the output of the code shown below?
    import math
    print([str(round(math.pi)) for i in range (1, 6)])
q9()

def q10():
    # 10. What is the output of the code shown below?
    l1 = [1, 2, 3]
    l2 = [4, 5, 6]
    l3 = [7, 8, 9]
    for x, y, z in zip(l1, l2, l3):
        print(x, y, z)
q10()

def q11():
    # 1. What is the output of the following?
    my_string = 'asdlfklkzc..12.3'
    k = [print(i) for i in my_string if i not in "aeiou"]
q11()

def q12():
    # 2. What is the output of print(k) in the following?
    my_string = 'asdlfklkzc..12.3'
    k = [print(i) for i in my_string if i not in "aeiou"]
    print(k)  # list of None-s
q12()

def q13():
    # 3. What is the output of the following?
    my_string = "hello world"
    k = [(i.upper(), len(i)) for i in my_string]
    print(k)
q13()

def q14():
    # 4. Which of the following is the correct expansion of
    # list_1 = [expr(i) for i in list_0 if func(i)] ?
    lst = []
    for i in list_0:
        if func(i):
            lst.append(expr(i))
# q14()

def q15():
    # 5. What is the output of the following?
    x = [i**+1 for i in range(3)]
    print(x)
q15()

def q16():
    # 6. What is the output of the following?
    print([i.lower() for i in "HELLO"])
q16()

def q17():
    #7. What is the output of the following?
    print([i+j for i in "abc" for j in "def"])
q17()

def q18():
    # 8. What is the output of the following?
    print([[i+j for i in "abc"] for j in "def"])
q18()

# def q19():
#     # 9. What is the output of the following?
#     print([if i%2==0: i; else: i+1; for i in range(4)])
# q19()

def q20():
    # 10. Which of the following is the same as list(map(lambda x: x**-1, [1, 2, 3]))?
    print([x**-1 for x in [1, 2, 3]])
q20()

def q21():
    # 1. Read the information given below carefully and write a list comprehension such that the output is: [‘e’, ‘o’]
    w="hello"
    v=('a', 'e', 'i', 'o', 'u')
    print([char for char in w if char in v])
q21()

def q22():
    # 2. What is the output of the code shown below?
    print([ord(ch) for ch in 'abc'])
q22()

def q23():
    # 3. What is the output of the code shown below?
    t = 32.00
    print([round((x-32)*5/9) for x in t])

def q24():
    # 4. Write a list comprehension for producing a list of numbers between 1 and 1000 that are divisible by 3.
    print([num for num in range(1000) if num % 3 == 0])
q24()

def q25():
    # 5. Write a list comprehension equivalent for the code shown below:
    # for i in range(1, 101):
	#   if int(i*0.5)==i*0.5:
	# 	    print(i)
    [print(i) for i in range(1, 101) if int(i * 0.5) == i * 0.5]
q25()

def q26():
    # 7. Write a list comprehension to produce the list: [1, 2, 4, 8, 16 ... 4096].
    print([2**x for x in range(0, 13)])
q26()

def q27():
    # 8. What is the list comprehension equivalent for:
    # {x : x is a whole number less than 20, x is even} (including zero)
    print([x for x in range(0, 20) if x%2==0])
q27()

def q28():
    # 9. What is the output of the list comprehension shown below?
    print([j for i in range(2,8) for j in range(i*2, 50, i)])  # ?!?
q28()

def q29():
    # 10. What is the output of the code shown below?
    l = ["good", "oh!", "excellent!", "#450"]
    print([n for n in l if n.isalpha() or n.isdigit()])
q29()

def q30():
    # 2. What is the output of the snippet of code shown below?
    A = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]
    print(A[1])
q30()
