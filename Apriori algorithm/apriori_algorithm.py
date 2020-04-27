# Apriori algorithm developed and used on a sample dataset
# This code was written just for demonstration and learning purposes

from collections import Counter
from itertools import combinations
import numpy as np

# transactions = {
#     1: ["a", "c", "d"],
#     2: ["b", "c", "e"],
#     3: ["a", "b", "c", "e"],
#     5: ["b", "e"],
#     6: ["a", "c", "e"]
# }


transactions = {
    1: ["1", "3", "4"],
    2: ["2", "3", "5"],
    3: ["1", "2", "3", "5"],
    5: ["2", "5"],
    6: ["1", "3", "5"]
}

min_support_count = 2
min_confidence_value = 0.6


# apriori pruning concept
def _pruning(current, previous, size):
    final_keys = []
    previous = [tuple(i) for i in previous]
    for key in current:
        FLAG = False
        current_comb = list(combinations(key, size))
        for i in current_comb:
            if i in previous or i[::-1] in previous:
                FLAG = True
            else:
                FLAG = False
                break

        if FLAG:
            final_keys.append(key)

    return final_keys


def support_value(itemset_keys_, transactions):
    itemset = {key: 0 for key in itemset_keys_}

    for keys in itemset_keys_:
        for val in transactions.values():
            if set(keys) & set(val) == set(keys):
                itemset[keys] += 1
    return itemset


# creating frequent itemset
def get_frequent_itemset(size=None, transactions=None, itemset=None):
    if size == 1:
        itemset = Counter()

        for val in transactions.values():
            itemset.update(val)

    else:

        prev_itemset_keys = list(itemset.keys())
        prev_itemset = itemset.copy()

        valid_keys = list(set(itemset.keys()))
        # flatten list of tuple -> keys: [(), ()] -> []
        # useful for running a combination of all the chosen features
        l = []
        for row in valid_keys:
            l.extend(row)

        valid_keys = set(l)

        # candidate itemset keys
        itemset_keys_ = list(combinations(valid_keys, size))

        # Apriori algorithm is based on theconcept that a subset
        # of a frequent itemset must also be a frequent itemset
        # so we are pruning away those features whose subset are not present
        # in the previous frequent itemset
        if size >= 2:
            itemset_keys_ = _pruning(
                itemset_keys_, prev_itemset_keys, size - 1)

        # finding support value for each of the selected itemset feature combination
        itemset = support_value(itemset_keys_, transactions)

        # defaulting back to th previous frequent itemset if
        # the iteration doesn't find any itemset which has the theshold required
        if itemset == {}:
            itemset = prev_itemset

    # getting frequent itemset from itemset
    # Frequent Itemset is an itemset whose support
    # value is greater than a threshold value(support).

    frequent_itemset = {}
    for key, val in itemset.items():
        if val >= min_support_count:
            frequent_itemset[key] = val

    return frequent_itemset


def finding_subsets(frequent_set):
    item_list = []
    size = len(list(frequent_set.keys())[0])
    for key in frequent_set.keys():
        subsets = []
        for i in range(1, size):
            subsets.append(list(combinations(key, i)))

        subsets = list(np.array(subsets).flatten())
        subsets.insert(0, key)
        item_list.append(subsets)

    return item_list


def finding_rules(itemset_sub):
    print("Antecedents -->  Consequents --- Confidence")
    for i in range(1, len(itemset_sub)):

        # passing as list as we have designed support_value function as
        # a function that takes an iteratable list of itemsets
        x = support_value([itemset_sub[0], ], transactions)
        y = support_value([itemset_sub[i], ], transactions)
        confidence = list(x.values())[0] / list(y.values())[0]
        if confidence >= min_confidence_value:
            print(
                f"{itemset_sub[i]} --> {itemset_sub[0]} --- {round(confidence, 2)}")


print("""
    ITEMS
    1: Banana
    2: Eggs
    3: Milk
    4: Tea
    5: Bread

""")

f = {}

for i in range(1, 5):
    f = get_frequent_itemset(size=i, transactions=transactions,
                             itemset=f)

# frequent_itemsets

print("Frequent Itemsets...")
for key, val in f.items():
    print(f"Itemset: {key}, support value: {val}")


subset = finding_subsets(f)

for i in subset:
    print(f"Rules for itemset - {i[0]}")
    finding_rules(i)
    print()
