import numpy as np
import os
import shutil

#load feature vector
seven = np.loadtxt('results.csv', delimiter=',')

'''
trh = 0.2
seven[abs(seven) < trh] = 0
'''

#calculate mask vector positive num:+; zero:0; negitive num:-1
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

#save mask to file
np.savetxt("mask.csv", mask, delimiter=",")
groups = list()

#delete the column that almost all lines(sum * rate) are the same
mask_sum = np.zeros(mask.shape[1], dtype=int)
sum = 0
for i in range(mask.shape[0]):
    mask_sum += mask[i]
    sum += 1

rate = 0.8
trhd = int(sum * rate)
col2del = list()

for j in range(mask.shape[1]):
    if abs(mask_sum[j]) > trhd or abs(mask_sum[j]) < (sum - trhd):
        col2del.append(j)
mask = np.delete(mask, col2del, axis=1)

for i in range(len(col2del)):
    col2del[i] = col2del[i] + 1
seven = np.delete(seven, col2del, axis=1)

#find the similar lines
for i in range(mask.shape[0]):
    group = list()

    if mask[i][0] != 2:
        #group.append(str(int(seven[i][0])) + '.png')
        group.append(i)
        for j in range(i + 1, mask.shape[0]):
            if (mask[i] == mask[j]).all():
                #group.append(str(int(seven[j][0]))+'.png')
                group.append(j)
                mask[j] = 2
        #print(len(group))
    groups.append(group)

#calculate the cluster center
#note: the index 0 of cluster_center has no means
cluster = 6
total = list()
for i in range(len(groups)):
    total.append(len(groups[i]))

groups_total = np.array(total)
top_k_idx=groups_total.argsort()[::-1][0:cluster]

cluster_center = np.zeros((cluster, seven.shape[1]))
for i in range(cluster):
    for j in range(len(groups[top_k_idx[i]])):
        cluster_center[i] += seven[groups[top_k_idx[i]][j]]
    cluster_center[i] /= len(groups[top_k_idx[i]])

#calculate the distance of two vector
#note: index 0 of both vector are not take part in the calculation
def calc_dist(a, b):
    result = np.linalg.norm(a[1:-1] - b[1:-1])
    return result

#clustering
new_gruops = [[] for i in range(cluster)]
for i in range(seven.shape[0]):
    distance = [0] * cluster
    for j in range(cluster):
        distance[j] = calc_dist(seven[i],cluster_center[j])
    new_gruops[distance.index(min(distance))].append(str(int(seven[i][0]))+'.png')

#copy grouped images
def copy_img(groups):
    if os.path.exists('./7/'):
        shutil.rmtree('./7/')
    if not os.path.exists('./7/'):
        os.mkdir('./7/')
    for i in range(len(groups)):
        if (len(groups[i]) > 0):
            for j in range(len(groups[i])):
                if not(os.path.exists('./7/'+groups[i][0]+'/'+groups[i][j])):
                    shutil.copyfile('./data/testing/7/'+groups[i][j], './7/'+groups[i][0]+'-'+groups[i][j])

    #write group imformation to file
    with open('gruop.csv', 'w') as f:
        for i in range(len(groups)):
            if len(groups[i]) > 0:
                f.write('%s, %d\n' % (groups[i][0], len(groups[i])))
        f.close()

copy_img(new_gruops)

