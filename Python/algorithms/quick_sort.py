"""
Quick-sort algorithm.

Like merge-sort, this algorithm is also based on the divide-and-conquer paradigm.
However, it uses this technique in a different manner, as all the hard work is done before the recursive calls.
The main idea is to apply the divide-and-conquer technique: divide S into subsequences, sort each subsequence, and then combine the sorted subsequences by a simple concatenation.
Steps of the divide-and-conquer algorithm:
    1. Divide: Select a pivot: element x from S. As is common practice, choose the pivot x to be the last element in S. Remove all the elements from S and put them into 3 sequences:
        - L: storing the elements in S less than x
        - E: storing the elements in S equal to x
        - G: storing the elements in S greater than x
    If the elements of S are distinct, then E hodls just one element: pivot itself.
    2. Conquer: Recursively sort sequences L and G.
    3. Combine: Put back the elements into S in order by first inserting the elements of L, then thos of E, and finally those of G.

The visualized version of the algorithm can be represented by means of a binary recursion tree, called the quick-sort tree.

Running time of quick-sort is O(n*h), where h is height of the quick-sort tree T.
Worst case: n distinct elements in sequence, which are already sorted h=n. O(n^2)
Best case: L and G have roughly the same size. Time would be O(n*log_2(n))
Introducing randomization in the choice of pivot will makes quick-sort expected running time O(n*log_2(n)).
"""

from collections import deque
import random


class LinkedQueueBasedQuickSort(object):
    """Quick-sort algorithm implementation on Linked-queue data type."""
    def quick_sort(self, S):
        """Sort the elements of queue S using the quick-sort algorithm."""
        n = len(S)
        # 0. Check if we even need to run quick-sort on the list.
        if n < 2:
            return  # List of already sorted.
        # 1. Divide
        pivot = S[-1]
        L = deque()
        E = deque()
        G = deque()
        while S:  # Divide S into L, E and G. Same as [while not S.is_empty()]
            if S[-1] < pivot:
                L.appendleft(S.pop())
            elif S[-1] > pivot:
                G.appendleft(S.pop())
            else:
                E.appendleft(S.pop())  # S[-1] must equal pivot.
        # 2. Conquer (with recursion).
        self.quick_sort(L)  # Sort elements less than pivot.
        self.quick_sort(G)  # Sort elements greater than pivot.
        # 3. Combine: concatenate results.
        print('DEBUG: BEFORE CONCATENATION:\nL:{0}\nE:{1}\nG:{2}\n'.format(L, E, G))
        while L:  # Same as [while not L.is_empty()]
            S.appendleft(L.pop())
        while E:
            S.appendleft(E.pop())
        while G:
            S.appendleft(G.pop())
    
    def print_example(self):
        deque_tosort = deque(random.sample(range(100), 10))
        # deque_tosort = deque(random.sample(range(10), 10) * 3)
        print(deque_tosort)
        self.quick_sort(deque_tosort)
        deque_tosort.reverse()
        print(deque_tosort)


"""
An algorithm is in-place if it uses only a small amount of memory in addition to that needed for the original input.
The following class illustrates optimization to quick-sort.
Such implementation is used in most deployed implementations.
"""
class ArrayBasedInPlaceQuickSort(object):
    """Quick-sort algorithm based on in-place memory management used on array-based list."""
    def inplace_quick_sort(self, S, a, b):
        """Sort the list from S[a] to S[b] inclusive using the quick-sort algorithm."""
        if a >= b:
            return  # range is trivially sorted.
        pivot = S[b]  # last element of range is pivot.
        left = a  # will scan rightward.
        right = b-1  # will scan leftward.
        while left <= right:
            # scan until reaching value equal or larger than pivot (or right marker)
            while left <= right and S[left] < pivot:
                left += 1
            # scan until reaching value equal or smaller than pivot (or left marker)
            while left <= right and S[right] > pivot:
                right -= 1
            if left <= right:  # scans did not strictly cross
                S[left], S[right] = S[right], S[left]  # swap values.
                left, right = left+1, right-1  # shrink range.
        # Put pivot into its final place (currently marked by left index)
        S[left], S[b] = S[b], S[left]
        # Make recursive calls.
        self.inplace_quick_sort(S, a, left-1)
        self.inplace_quick_sort(S, left+1, b)

    def print_example(self):
        array_tosort = random.sample(range(100), 10)
        print(array_tosort)
        self.inplace_quick_sort(array_tosort, 0, len(array_tosort) - 1)
        print(array_tosort)       


# _algo = LinkedQueueBasedQuickSort()
# _algo.print_example()

_algo = ArrayBasedInPlaceQuickSort()
_algo.print_example()
