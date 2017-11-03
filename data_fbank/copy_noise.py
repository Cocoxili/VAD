"""
    Copy the noise data to six times the original
    for example:
        T2474G10981S0131_reverb_and_noise
        T2474G10981S0131_reverb_and_noise_c1
        T2474G10981S0131_reverb_and_noise_c2
        T2474G10981S0131_reverb_and_noise_c3
        T2474G10981S0131_reverb_and_noise_c4
        T2474G10981S0131_reverb_and_noise_c5
"""

import random

scp = 'cmvn.noise.scp.bak'

file1 = open(scp, 'r')
file2 = open('cmvn.noise.scp', 'w')
# shuffle
lines = file1.readlines()
lines_new = []

for line in lines:
    lines_new.append(line)

for i in range(1, 6):
    for line in lines:
        line = line.split(' ')
        new_key = line[0] + '_c' + str(i)
        new_line = ' '.join([new_key, line[1]])
        lines_new.append(new_line)

file2.writelines(lines_new)

print 'finished.'
