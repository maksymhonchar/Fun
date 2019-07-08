# src:
# https://www.geeksforgeeks.org/practice-questions-time-complexity-analysis/


"""
Name	    Big O
Constant	O(c)
Linear	    O(n)
Quadratic	O(n^2)
Cubic	    O(n^3)
Exponential	O(2^n)
Logarithmic	O(log(n))
Log Linear	O(nlog(n))
"""

"""
1. What is the time, space complexity of following code:

int a = 0, b = 0; 
for (i = 0; i < N; i++) { 
    a = a + rand(); 
} 
for (j = 0; j < M; j++) { 
    b = b + rand(); 
} 

O(N+M) time;
O(1) space;
"""

"""
2. What is the time complexity of following code:

int a = 0; 
for (i = 0; i < N; i++) { 
    for (j = N; j > i; j--) { 
        a = a + i + j; 
    } 
} 

O(N*N) time
"""
# N = 10
# for i in range(10):
#     j = N
#     while j > i:
#         print(i, j)
#         j -= 1

"""
3. What is the time complexity of following code:

int i, j, k = 0; 
for (i = n / 2; i <= n; i++) { 
    for (j = 2; j <= n; j = j * 2) { 
        k = k + n / 2; 
    } 
} 

O(n * logn)
j keeps doubling till it is less than or equal to n.
Number of times, we can double a number till it is less than n would be log(n).
"""

"""
5. What is the time complexity of following code:

int a = 0, i = N; 
while (i > 0) { 
    a += i; 
    i /= 2; 
} 

O(logn)
"""
