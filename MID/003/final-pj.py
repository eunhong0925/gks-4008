import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
import string
#
data = fetch_20newsgroups()
categories = data.target_names
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
#
vocab_set = set()
for dstr in train.data:
    words = dstr.lower().split()
    vocab_set.update(words)
    
vocab_list = list(vocab_set)
#
def clean_list(input_list):
    cleaned_list = [''.join(char for char in element if char.isalpha()) for element in input_list]
    cleaned_list = list(filter(None, cleaned_list))
    return cleaned_list
#
def prior_prob(categories):
    category_counts = np.zeros(len(categories))

    for i in range(len(train.data)):
        GT_category = train.target_names[train.target[i]]
        
        for j, s in enumerate(categories):
            if GT_category == s:
                category_counts[j] += 1

    total_samples = len(train.data)
    category_probabilities = category_counts / total_samples
    
    return category_probabilities.tolist()  # Convert to Python list

category_probabilities = prior_prob(categories)
print(category_probabilities)
#

