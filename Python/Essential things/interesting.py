# 1. Emulating [x ? y : z] construction
var = 5
z = False
y = True
result = [z, y][bool(var == 5)]  # note, that list is [False, True] !
print(result)  # True

# Another way to do x?y:x is (but ONLY if [y] is True) :
result = bool(var == 5) and y or z
print(result)  # True

# 2. Checking for the very old Python versions
# This block should be at the top of program, even before imports.
import sys
if not hasattr(sys, "hexversion") or sys.hexversion < 0x020300f0:
    sys.stderr.write('Sorry, your Python version is too old.\n')
    sys.stderr.write('Please, upgrade at least to 2.3.\n')
    sys.exit(1)

# 3. Parallel sorting of lists.
list1 = [3, 2, 1]
list2 = [6, 5, 4]
data = list(zip(list1, list2))
data.sort()
list1, list2 = map(lambda t: list(t), zip(*data))
print(list1, list2)

tuple1, tuple2 = zip(*data)
print(tuple1, tuple2)

# 4. Sorting IP addresses
ips = ['212.90.43.34',
       '127.255.255.255',
       '191.255.255.255']
for i in range(len(ips)):
    ips[i] = "%3s.%3s.%3s.%3s" % tuple(ips[i].split("."))
ips.sort()
for i in range(len(ips)):
    ips[i] = ips[i].replace(" ", "")
print(ips)
