from functools import reduce
import pandas as pd


# src 1. https://blog.usejournal.com/how-not-to-give-programming-interviews-functional-programming-map-reduce-filter-47bb679eaf02
ages = [22, 66, 4, 21, 35, 43]
print(list(filter(lambda age: age > 25, ages)))

words = ["one", "two", "three", "four", "five"]
print(list(filter(lambda word: len(word) < 4, words)))

numbers = [3, 4, 243, 35, 3421, 265, 1432092]
print(list(map(lambda x: x * 2, numbers)))

one_to_five = range(1, 6)
product = reduce(lambda x, y: x * y, one_to_five)
sum = reduce(lambda x, y: x + y, one_to_five)
print(product, sum)

# src 2. https://www.python-course.eu/python3_lambda.php
fahrenheit = lambda temp: ((float(9)/5)*temp + 32)
celsius_temp_values = [-5, 0, 20.5, 42.5]
print(list(map(fahrenheit, celsius_temp_values)))

celsius_temp_values_2 = [-10, -9, -8, -7]
celsius_temp_values_3 = [7, 8, 9, 10, 11, 12, 13]
print_vals_two_lists = lambda x, y: print(x, y)
# Note: stop when the shortest list has been consumed
print(list(  
    map(print_vals_two_lists, celsius_temp_values_2, celsius_temp_values_3)
))

fibonacci = [0,1,1,2,3,5,8,13,21,34,55]
odd_numbers = list(filter(lambda num: num % 2 == 1, fibonacci))
print(odd_numbers)
even_numbers = list(filter(lambda num: num % 2 == 0, fibonacci))
print(even_numbers)

max_value_func = lambda prev, cur: prev if (prev > cur) else cur
print(reduce(max_value_func, fibonacci))
min_value_fync = lambda prev, cur: cur if (cur < prev) else prev
print(reduce(min_value_fync, fibonacci + [-1, -100]))

# exercise 1
print('\n')
sublist = {
    'order_number': ['34587', '98762', '77226', '88112'],
    'book_title_author': ['Learning Python, Mark Lutz', 'Programming Python, Mark Lutz', 'Head First Python, Paul Barry', 'Head First Python, Paul Barry'],
    'quantity': [4, 5, 3, 3],
    'price_per_item': [40.95, 56.80, 32.95, 24.99]
}
sublist_rows = list(zip(*sublist.values()))
print(sublist_rows, '\n')
ex1_answer = list(map(lambda row: (row[0], row[2] * row[3]), sublist_rows))
print(ex1_answer)

# invoice_totals = list(
#     map(
#         lambda x: x if x[1] >= min_order else (x[0], x[1] + 10),
#         map(lambda x: (x[0], x[2] * x[3]), orders)
#     )
# )
