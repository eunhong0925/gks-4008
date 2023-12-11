import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
categories = data.target_names
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
vocab_set = set()
for dstr in train.data:
    words = dstr.lower().split()
    vocab_set.update(words)
new_vocab= list(vocab_set)


# collect data strings only for 'space'
data_space = []
for i in range(len(train.data)):
    if train.target_names[train.target[i]] == "sci.space":
        # print (i, train.target_names[train.target[i]])
        data_space.append(train.data[i])
#
# print(data_space[2])
len(data_space)

# let's count!
# vocabcount_space = np.zeros( (len(vocab)), dtype='int')
vocabcount_space = { v: 0 for v in vocab}

total_w = 0
for dstr in data_space:
    words = dstr.lower().split()
    # print(words)
    for w in words:
        if w in vocab:
            # print(w)
            total_w += 1
            vocabcount_space[w] += 1
#
print(f"total words in vocab = {total_w}") 

total = 0
for k, v in vocabcount_space.items():
    total += v 
print(total)

# Add 1 to every count = Laplace Smoothing
for k in vocabcount_space:
    vocabcount_space[k] += 1 

total = 0
for k, v in vocabcount_space.items():
    total += v 
print("total after lapace smoothing: ", total)

# convert counts to prob
for k in vocabcount_space:
    vocabcount_space[k] /= total


# compute log-probability

vocab_space_logp = { key: np.log(value) for key, value in vocabcount_space.items() }