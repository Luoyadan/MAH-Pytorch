'''
import os
import numpy as np
label = np.zeros((25000, 38), dtype=int)
path = '/home/yadan/github/CIKM2018-Code/ADSH-AAAI2018/ADSH_pytorch/data/Flickr-25/tag'

j = 0
for root, dirs, files in os.walk(path):
    for name in files:
        if '.txt' in name:
            print name
            with open(root+'/'+name, 'r') as f:
                for line in f:
                    s = line.split()
                    label[int(s[0])-1, j] = 1
            j = j + 1

            print label






print("Enter in Saving Stage...")
with open('../data/Flickr-25/unique_label.txt','w') as f:
    for j in label:
        temp = str(j).replace('\n', '').strip('[').strip(']')
        f.write(temp+'\n')

'''
'''
import random
label = []
import os
# Obtain the name of all images

path = '/home/yadan/github/CIKM2018-Code/ADSH-AAAI2018/ADSH_pytorch/data/Flickr-25/mirflickr/'

image = []
for i in range(25000):
    image.append(path+'im'+str(i+1)+'.jpg')




with open('../data/Flickr-25/unique_label.txt','r') as f:
    for line in f:
        s = line.split()
        label.append(s)
random.seed(0)
all_idx = range(25000)
test_idx = random.sample(all_idx, 1700)
train_idx = [x for x in all_idx if x not in test_idx]
database_label = [label[i] for i in train_idx]
test_label = [label[i] for i in test_idx]
database_image = [image[i] for i in train_idx]
test_image = [image[i] for i in test_idx]

with open('../data/Flickr-25/database_image.txt','w') as f:
    for j in database_image:
        f.write(j+'\n')

with open('../data/Flickr-25/database_label.txt','w') as f:
    for j in database_label:
        temp = str(j).replace('\n', '').replace('\'', '').replace(',', '').strip('[').strip(']')
        f.write(temp + '\n')

with open('../data/Flickr-25/test_image.txt','w') as f:
    for j in test_image:
        f.write(j+'\n')

with open('../data/Flickr-25/test_label.txt','w') as f:
    for j in test_label:
        temp = str(j).replace('\n', '').replace('\'', '').replace(',', '').strip('[').strip(']')
        f.write(temp + '\n')
'''

label = []
with open('../data/Flickr-25/test_label.txt','r') as f:
    for line in f:
        s = line.split()
        label.append(" ".join(str(y) for y in s))
