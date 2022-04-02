from sys import argv

fname = argv[1]
total = float(argv[2])
with open(fname) as f:
    lines = f.readlines()

s = 0
cnt = 0
for line in lines:
    c = int(line.split()[6])
    if c > 0:
        s += (c / 100)
        cnt += 1

print(s / cnt * total)



