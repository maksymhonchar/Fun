class UnionField(object):
    """ Represents find union field """

    def __init__(self, n=10):
        # Initialize union-find data structure with N objects (0 to N-1)
        self.id = [i for i in range(n)]

    def union(self, p, q):
        # Add connection between p and q.
        # Second value (q) is more important.
        # Unfortunately, takes time O(N)
        pid = self.id[p]
        qid = self.id[q]
        for i in range(len(self.id)):
            if self.id[i] == pid:
                self.id[i] = qid

    def connected(self, p, q):
        # Are p and q in the same component
        return self.id[p] == self.id[q]

    def find(self, p):
        # Find component identifier for p (0 to N-1)
        try:
            return self.id[p]
        except IndexError as e:
            print('IndexError:', e)
            return -1

    def count(self):
        # Count number of components
        count = 0
        comp_id = [-1 for i in self.id]
        for i in self.id:
            print(self.id[i])
            comp_id[self.id[i]] = i
        for i in comp_id:
            if comp_id[i] != -1:
                count += 1
        return count

    def __str__(self):
        return str(self.id)

class QuickUnion(object):
    """ Represents quick-union union field """

    def __init__(self, n=10):
        # Initialize union-find data structure with N objects (0 to N-1)
        self.id = [i for i in range(n)]

    def root(self, i):
        while i != self.id[i]:
            i = self.id[i]
        # Return depth of i array access
        return i

    def connected(self, p, q):
        # Check if p and q have same root
        return self.root(p) == self.root(q)

    def union(self, p, q):
        # Takes only O(1) time.
        i = self.root(p)
        j = self.root(q)
        # Set a new root
        self.id[i] = j


def main():
    # Number of objects that are going to be processed
    N = int(input('>'))
    uf = UnionField(N)
    print(uf)
    p = int(input('>'))
    q = int(input('>'))
    while p != -1 or q != -1:
        if not uf.connected(p, q):
            uf.union(p, q)
            print('union between: [{0},{1}]'.format(p, q))
        print(uf)
        p = int(input('>'))
        q = int(input('>'))
    print('End of the program')


def test():
    t = UnionField()
    print(t.id)
    print(t.count())


if __name__ == '__main__':
    main()
    # test()
