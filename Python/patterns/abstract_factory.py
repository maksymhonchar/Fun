# Abstract factory.
class IToyFactory():
    def get_bear(self):
        pass
    def get_cat(self):
        pass
    def object_3(self):
        pass
    # etc.

# Concrete Factory 1
class TeddyToysFactory(IToyFactory):
    def get_cat(self, name):
        return TeddyCat(name)

# Concrete Factory 1
class WoodenToysFactory(IToyFactory):
    def get_cat(self, name):
        return WoodenCat(name)

# Abstract Product
class Cat():
    def __init__(self, name):
        self.name = name
    def pr(self):
        print('meow', self.name)

# Concrete Product
class TeddyCat(Cat):
    def pr(self):
        print('meow teddy cat', self.name)

# Concrete Product 2
class WoodenCat(Cat):
    def pr(self):
        print('meow wooden cat', self.name)


def main():
    def wooden():
        factory = WoodenToysFactory()
        cat = factory.get_cat('test1')
        cat.pr()
        print(cat.__class__)  # <class '__main__.WoodenCat'>
    def teddy():
        factory = TeddyToysFactory()
        cat = factory.get_cat('test2')
        cat.pr()
        print(cat.__class__)  # <class '__main__.TeddyCat'>
    wooden()
    teddy()


if __name__ == '__main__':
    main()
