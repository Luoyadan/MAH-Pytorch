import os
label = []
image = []
re_image = []
re_label = []
path = '/home/yadan/github/CIKM2018-Code/ADSH-AAAI2018/ADSH_pytorch/data/NUS-WIDE/images'

with open('../data/NUS-WIDE/database_img.txt','r') as f:
    for line in f:
        s = line.split()
        temp = " ".join(str(x[7:]) for x in s)
        image.append(temp)

with open('../data/NUS-WIDE/database_label.txt','r') as f:
    for line in f:
        s = line.split()
        label.append(" ".join(str(y) for y in s))


for root, dirs, files in os.walk(path):
    for name in files:
        if name in image:
            re_image.append(str(root) +'/' +str(name))
            ix = image.index(name)
            re_label.append(label[ix])
        if len(re_label) % 100 == 0:
            print('dealing with the ' + str(len(re_label)) +'th file')



with open('../data/NUS-WIDE/database_image_new.txt','w') as f:
    for j in re_image:
        f.write(j+'\n')

with open('../data/NUS-WIDE/database_label_new.txt','w') as f:
    for j in re_label:
        f.write(j+'\n')
'''
label = []
with open('../data/NUS-WIDE/test_image_new.txt','r') as f:
    for line in f:
        s = line.split()
        temp = " ".join(str(x)[:-19]+'/'+str(x)[-19:] for x in s)
        label.append(temp)

with open('../data/NUS-WIDE/test_image_new1.txt','w') as f:
    for j in label:
        f.write(j+'\n')

'''