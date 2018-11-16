# Shelve module
# Purpose: implements persistent storage for arbitrary
# Python objects which can be pickled, using a dictionary-like API

# The shelve module can be used as a simple persistent storage option
# for Python objects when a relational database is overkill.

# Shelve db is managed by anydbm

import shelve

# Create a new Shelf
s = shelve.open('test_shelf.db')
# Store some info in it
try:
    s['key1'] = {
        'int': 10,
        'float': 9.5,
        'string': 'Sample string'
    }
finally:
    s.close()
# To access data, open the shelf and get info
# like it's a dictionary:
s = shelve.open('test_shelf.db')
print(type(s))  # <class 'shelve.DbfilenameShelf'>
try:
    info = s['key1']
finally:
    s.close()
print(info)

# It is possible to open shelfe DB in read-only mode
s = shelve.open('test_shelf.db', flag='r')
try:
    s['key2'] = {
        'int': 0
    }
except Exception as e:
    print('Exception:', e)  # Reader can't store.
finally:
    s.close()


print()
# To catch changes in shelf, it should be opened with writeback enabled
s = shelve.open('test_shelf.db', writeback=True)
try:
    print(s['key1'])
    s['key1']['new_key'] = 'new value'
    print(s['key1'])
finally:
    s.close()

# Another example
d = shelve.open('test_shelf.db', writeback=True)
del d['key1']

data = 'hey'
# Store data at key.
d['key1'] = data
# Retrieve a copy of data at key
data = d['key1']
print(data)

flag = 'key1' in d
if flag:
    print(d['key1'])  # hey
else:
    print('nothing to do here.')

# Let's add a list to the shelf database.
d['xx'] = [0, 1, 2]
# The next line will append an item to the list!
d['xx'].append(3)
print(d['xx'])
# If [writeback] argument is False, then do it wouldn't be appended.
# You should do the following:
tmp = d['xx']
tmp.append(3)
d['xx'] = tmp  # So, simply override value at certain key.

