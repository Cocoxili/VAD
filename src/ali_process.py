"""
1. Transform ali form:
    82 phoneId means silence
2. Down-sampling:

example 1:
part-03ffc602-dba0-4750-bb81-92c856c659a5 82 34 ; 78 6 ; 37 17 ; 65 17 ; 119 7 ; 69 7 ; 66 20
>
part-03ffc602-dba0-4750-bb81-92c856c659a5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
part-03ffc602-dba0-4750-bb81-92c856c659a5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

example 2:
part-fa33941f-8e85-4b27-ba03-6ccfae6afb51 82 23 ; 9 24 ; 61 20 ; 58 16 ; 83 15 ; 2 12 ; 66 10
>
part-fa33941f-8e85-4b27-ba03-6ccfae6afb51 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1

"""

from __future__ import division
import os

f0 = open('../ali.phone.length.txt', 'r')
f1 = open('../ali.phone.txt', 'w')
ali_old = f0.readlines()
ali_new = []


# print ali_old
for idx, line in enumerate(ali_old):
    line = line.strip().split(' ')
    key = line[0]
    line.pop(0)

    new = []

    for i in range(0, int(len(line) + 1), 3):
        if line[i] != '82':
            for j in range(int(line[i+1])):
                new.append('1')
        if line[i] == '82':
            for j in range(int(line[i+1])):
                new.append('0')

    # down sampling
    new_ds = [key]
    for i in range(0, len(new), 2):
        new_ds.append(new[i])

    # save txt
    line_new = ' '.join(str(i) for i in new_ds)
    line_new = line_new + '\n'
    ali_new.append(line_new)

    # if idx == 1000:
    #     break


f1.writelines(ali_new)
