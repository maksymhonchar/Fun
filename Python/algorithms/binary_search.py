"""
Binary search is a classic recursive algorithm.

It is used to efficiently locate a target value within a sorted sequence of n elements.
"""

def binary_search(data, target, low, high):
    """Binary search implementation, inefficient, O(n)

    Return True if target is found in indicated portion of a list.
    
    The search only considers the portion from low to high inclusive.
    """
    if low > high:
        return False    # interval is empty - no match.
    else:
        mid = (low + high) // 2
        if target == data[mid]:
            return True
        elif target < data[mid]:
            return binary_search(data, target, 0, mid-1)
        else:
            return binary_search(data, target, mid+1, high)
