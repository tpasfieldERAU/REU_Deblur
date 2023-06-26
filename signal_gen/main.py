from matplotlib import pyplot as plt
from blur import blur
from tikhonov import tkv_reconstruct

# Open file and import row from CSV manually
FILENAME = r"C:\Users\TJ\DataspellProjects\REUTesting\standard_signals.csv"
FILE = open(FILENAME, 'r')
for i in range(7):FILE.readline()
x = FILE.readline().split(',')
x.pop()
x = [float(i) for i in x]

b, A = blur(16, x, 0.015)

diffs = []
aas = [1.0]
a = 1.0
it = 0.5
direct = -1.
t1, d1 = tkv_reconstruct(b, A, a)
a = a + it * direct
aas.append(a)
t2, d2 = tkv_reconstruct(b, A, a)
print("d2 bigger")
if d2 > d1:
    direct = 1.0

diffs.append(d1)
diffs.append(d2)

a = a + it * direct
aas.append(a)
for i in range(10):
    x_re, diff = tkv_reconstruct(b, A, a)
    if diff < d2:
        print("LESS")
        d2 = diff
    else:
        print("GREATER")
        d2 = diff
        direct *= -1.
        it = it / 2.0
    a = a + it * direct
    aas.append(a)
    diffs.append(diff)

plt.plot(x)
plt.plot(x_re)
plt.title(f"{d2}")
plt.show()

print(a)
print()
print(d1)
print(diff)

plt.plot(diffs)
plt.show()

plt.plot(aas)
plt.show()
