from __future__ import print_function

a = 1000000000
b = 1e-6
for i in range(1000000):
    a += b
print(a - 1000000000)
