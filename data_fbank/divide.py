"""
    Shuffled feats.scp and Divide it into N parts.
"""
import random

N = 100
scp = 'feats.scp'

file1 = open(scp, 'r')

# shuffle
lines = file1.readlines()
random.shuffle(lines)

# divide
num_in_division = len(lines) / N

for i in range(N):

    filename = 'feats_' + str(i) + '.scp'
    print filename
    f = open(filename, 'w')

    f.writelines(lines[num_in_division * i : num_in_division * (i+1)])
    f.close()

print('finished.')
