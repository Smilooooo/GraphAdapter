import pickle

data = pickle.load(open('E:/BachelorThesis/Data/data(3)/data/LungHist700/split_fewshot/shot_8-seed_1.pkl', 'rb'))
train = data["train"]
print(f'Total train samples: {len(train)}')
print('\nPer class counts:')
for i in range(7):
    count = sum(1 for x in train if x.label == i)
    classname = train[0].classname if train else "Unknown"
    for x in train:
        if x.label == i:
            classname = x.classname
            break
    print(f'  Class {i} ({classname}): {count} samples')
