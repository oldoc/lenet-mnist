import numpy as np
import os
import shutil

seven = np.loadtxt('results.csv', delimiter=',')

trh = 0.2
seven[abs(seven) < trh] = 0

s = list()
s.append(seven.shape[0])
s.append(seven.shape[1]-1)

mask = np.empty(s, dtype=int)

for i in range(seven.shape[0]):
    for j in range(1, seven.shape[1]):
        if seven[i][j] > 0:
            mask[i][j-1] = 1
        else:
            if seven[i][j] < 0:
                mask[i][j-1] = -1
            else:
                mask[i][j-1] = 0

np.savetxt("mask.csv", mask, delimiter=",")
groups = list()

for i in range(mask.shape[0]):
    group = list()

    if mask[i][0] != 2:
        group.append(str(int(seven[i][0])) + '.png')
        for j in range(i + 1, mask.shape[0]):
            if (mask[i] == mask[j]).all():
                group.append(str(int(seven[j][0]))+'.png')
                mask[j] = 2
        #print(len(group))
    groups.append(group)



for i in range(len(groups)):

    if (len(groups[i]) > 0):
        if not os.path.exists('./7/'+groups[i][0]):
            os.mkdir('./7/'+groups[i][0])
        for j in range(len(groups[i])):
            if not(os.path.exists('./7/'+groups[i][0]+'/'+groups[i][j])):
                shutil.copyfile('./data/testing/7/'+groups[i][j], './7/'+groups[i][0]+'/'+groups[i][j])

for i in range(len(groups)):
    if len(groups[i]) > 0:
        print('%s, %d' % (groups[i][0],len(groups[i])))
