def factorial(n):
    if n < 0:
        raise ValueError('n has to be greater than 0')
    elif n == 0:
        return 1
    else:
        return n * factorial(n-1)

def unique(seq):
    """Return True if there are no duplicate elements in sequence seq."""
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            if seq[i] == seq[j]:
                return False
    return True

def check_seq_uniqueness(seq, start, stop):
    """Note: this is a terribly inefficient use of recursion!!!"""
    if stop - start <= 1:
        return True  # at most one item in sequence.
    elif not check_seq_uniqueness(seq, start, stop-1):
        return False
    elif not check_seq_uniqueness(seq, start+1, stop):
        return False
    else:
        return seq[start] != seq[stop - 1]

def fibonacci(n):
    if n <= 1:
        return (n, 0)
    else:
        (a, b) = fibonacci(n-1)
        return (a + b, a)

def linear_seq_sum(seq, n):
    """Return the sum of the first n numbers of sequence S."""
    if n == 0:
        return 0
    else:
        return linear_seq_sum(seq, n-1) + seq[n-1]

def binary_seq_sum(seq, start, stop):
    if start >= stop:  # 0 elements in slice.
        return 0
    elif start == stop - 1:  # 1 element in slice.
        return seq[start]
    else:
        mid = (start + stop) // 2
        return binary_seq_sum(seq, start, mid) + \
                binary_seq_sum(seq, mid, stop)

def reverse_seq(seq, start, stop):
    """Reverse elements in implicit slice S[start:stop]."""
    if start < stop-1:
        seq[start], seq[stop-1] = seq[stop-1], seq[start]
        reverse_seq(seq, start+1, stop-1)

def reverse_iterative(seq):
    start, stop = 0, len(seq)
    while start < stop-1:
        seq[start], seq[stop-1] = seq[stop-1], seq[start]
        start, stop = start+1, stop-1

def computer_power(x, n):
    if n == 0:
        return 1
    else:
        return x * computer_power(x, n - 1)

def computer_power_v2(x, n):
    """Compute the value x**n for integer n."""
    if n == 0:
        return 1
    else:
        partial = computer_power_v2(x, n//2)
        result = partial * partial
        if n % 2 == 1:
            result *= x
        return result

if __name__ == "__main__":
    a = [1, 2, 3, 4, 4]
    print(check_seq_uniqueness(a, 0, len(a)))
    
    print(fibonacci(10))

    # To set recursive limit dynamically:
    import sys
    print(sys.getrecursionlimit())  # 1000
    sys.setrecursionlimit(100000)
    print(sys.getrecursionlimit())
