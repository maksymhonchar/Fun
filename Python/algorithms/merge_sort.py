"""
Merge sort algorithm.

Uses recursion in an algorithmic pattern divide-and-conquer.
The divide-and-conquer pattern consists of the following 3 steps:
    1. Divide: divide the input data into two or more disjoint subsets.
    2. Conquer: recursively solve the subproblems associated with the subsets.
    3. Combine: take the solutions to the subproblems and merge them into a solution to the original problem.
To sort a sequence S with n elements using divide-and-conquer steps algorithm proceeds as follows:
    1. Divide: Remove all the elements from S and put them into two sequences S1 and S2, each containing about half of the elements of S: S1 contains the firts [n/2] elements of S, and S2 contains the remaining [n/2] elements.
    2. Conquer: Recursively sort sequences S1 and S2.
    3. Combine: Put back the elements into S by merging the sorted sequences S1 and S2 into a sorted sequence.

We can visualize an execution of the merge-sort algorithm by means of a binary tree T, called the merge-sort tree.
The merge-sort tree associated with an execution of merge-sort on a sequence S of size n has height ceil( log_2(n) ).

Assuming two elements of S can be compared in O(1) time, algorithm merge-sort sorts a sequence S of size n in O(n*log_2(n)) time.
"""

from collections import deque
import math
import random


class ArrayBasedMergeSort(object):
    """Case when a sequence of items is represented as an (array-based) Python list."""
    def _merge(self, S1, S2, S):
        """Merge two sorted Python lists S1 and S2 into properly sized list S."""
        i = j = 0  # i: num of elements of S1 copied to S; j: same, about S2.
        while i + j < len(S):
            if j == len(S2) or (i < len(S1) and S1[i] < S2[j]):
                S[i+j] = S1[i]  # copy ith element of S1 as next item of S.
                i += 1
            else:
                S[i+j] = S2[j]  # copy jth elements of S2 as next item if S.
                j += 1   
    
    def merge_sort(self, S):
        """Sort the elements of Python list S using the merge-sort algorithm."""
        n = len(S)
        # 0. Check if S has zero or one element: situations when it is already sorted.
        if n < 2:
            return
        # 1. Divide.
        mid = n // 2
        S1 = S[0:mid]  # copy of first half.
        S2 = S[mid:n]  # copy of second half.
        # 2. Conquer (with recursion).
        self.merge_sort(S1)  # sort copy of first half.
        self.merge_sort(S2)  # sort copy of second half.
        # 3. Merge results.
        self._merge(S1, S2, S)  # merge sorted halves back into S.

    def print_example(self):
        array_tosort = random.sample(range(100), 10)
        print(array_tosort)
        self.merge_sort(array_tosort)
        print(array_tosort)


class LinkedQueueBasedMergeSort(object):
    """Case when a sequence of items is represented as an linked-based array."""
    def _merge(self, S1, S2, S):
        """Merge two sorted dequeue instances S1 and S2 into empty dequeue S."""
        while S1 and S2:  # same as [if not S1.is_empty() and not S2.is_empty()]
            if S1[-1] > S2[-1]:
                S.appendleft(S1.pop())
            else:
                S.appendleft(S2.pop())
        while S1:  # same as [while not S1.is_empty()]
            S.appendleft(S1.pop())  # move remaining elements of S1 to S.
        while S2:  # same as [while not S2.is_empty()]
            S.appendleft(S2.pop())  # move remaining elements of S2 to S.
    
    def merge_sort(self, S):
        """Sort the elements of dequeue S using the merge-sort algorithm."""
        n = len(S)
        # 0. Check if even need to sort the deque.
        if n < 2:
            return
        # 1. Divide.
        S1 = deque()
        S2 = deque()
        while len(S1) < (n // 2):
            S1.appendleft(S.pop())  # move the first n//2 elements to S1.
        while S:  # same as [while not S.is_empty()]
            S2.appendleft(S.pop())
        # 2. Conquer (with recursion).
        self.merge_sort(S1)  # sort first half.
        self.merge_sort(S2)  # sort second half.
        # 3. Merge results.
        self._merge(S1, S2, S)  # Merge sorted halves back into S.

    def print_example(self):
        deque_tosort = deque(random.sample(range(100), 10))
        # deque_tosort = deque(random.sample(range(10), 10) * 2)
        print(deque_tosort)
        self.merge_sort(deque_tosort)
        print(deque_tosort)


"""
The main idea is to perform merge-sort bottom-up, performing the merges level by level going up the merge-sort tree.

Nonrecursive bottom-up merge-sort is a bit faster than recursive merge-sort in practice, as it avoids the extra overheads of recursive calls and temporary memory at each level.
The algorithm runs in O(n*log_2(n)) time.
"""
class ArrayBasedBottomUpMergeSort(object):
    """Nonrecursive version of array-based merge-sort."""
    def _merge(self, src, result, start, inc):
        """Merge src[start:start + inc] and src[start+inc:start+2*inc] into result."""
        end1 = start + inc  # boundary for run 1.
        end2 = min(start + 2*inc, len(src))  # boundary for run 2.
        x, y, z = start, start+inc, start  # index into run1, run2, result.
        while x < end1 and y < end2:
            if src[x] < src[y]:
                result[z] = src[x]  # copy from run1
                x += 1  # increment x
            else:
                result[z] = src[y]  # copy from run2
                y += 1  # increment y
            z += 1  # increment z to reflect new result
        if x < end1:
            result[z:end2] = src[x:end1]  # Copy remainder of run1 to output.
        elif y < end2:
            result[z:end2] = src[y:end2]  # Copy remainder of run2 to output.

    def merge_sort(self, S):
        """Sort the elements of Python list S using the bottom-up merge-sort algorithm."""
        n = len(S)
        log_n = math.ceil(math.log(n, 2))
        src = S
        dest = [None] * n  # make temporary storage for dest.
        for i in (2**k for k in range(log_n)):  # pass i creates all runs of length 2i
            for j in range(0, n, 2*i):  # each pass merges two length i runs
                self._merge(src, dest, j, i)
            src, dest = dest, src  # Reverse roles of lists
        if S is not src:
            S[0:n] = src[0:n]  # Additional copy to get results to S

    def print_example(self):
        array_tosort = random.sample(range(100), 10)
        print(array_tosort)
        self.merge_sort(array_tosort)
        print(array_tosort)


if __name__ == "__main__":
    # Array-based merge-sort.
    _algo = ArrayBasedMergeSort()
    _algo.print_example()
    # Linked list based merge-sort.
    _algo = LinkedQueueBasedMergeSort()
    _algo.print_example()
    # Array-based merge-sort without recursion.
    _algo = ArrayBasedBottomUpMergeSort()
    _algo.print_example()
