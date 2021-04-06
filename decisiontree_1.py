####################################################################################################
# decisiontree_1.py
# AUTHOR:           NGOC TRAN
# CREATED:          05 Apr 2021
# DESCRIPTION:      An implementation of Decision Tree with Gini index as the split criterion for
#                   a classification problem
####################################################################################################
import pandas as pd
import numpy as np

data = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/balloons/adult-stretch.data',
    names=['Color', 'Size', 'Act', 'Age', 'Inflated'])

two_way_tbl = pd.crosstab(index=data['Age'], columns=data['Inflated'])


def gini_impurity(two_way_tbl):
    """
    Calculate gini index for a feature with 2 categories p & q.
    Gini impurity = 1 - (Pr(p))^2 - (Pr(q))^2
    NOTE:   Gini index only works with 2-class features.
    """
    nrow = len(two_way_tbl)
    ncol = len(two_way_tbl.columns)

    # Return False if there are more than 2 categories
    if nrow > 2:
        return False

    # Calculate gini index for each category, i.e. leaf, of the feature
    gini_leaf = []
    for r in range(nrow):     # Loop each category
        row = two_way_tbl.iloc[r, :]
        size = sum(row)  # total items of the category
        gini_value = 1
        for c in range(ncol):
            gini_value -= pow(row[c] / size, 2)
        gini_leaf.append(gini_value)

    # Count all items in the table
    total_items = 0
    for r in range(nrow):
        total_items += sum(two_way_tbl.iloc[r, :])

    # Calculate the weighted gini index for the feature, i.e. root of the sub-tree
    weighted_gini = 0
    for r in range(nrow):
        weighted_gini += gini_leaf[r] * (sum(two_way_tbl.iloc[r, :]) / total_items)

    return weighted_gini


gini = gini_impurity(two_way_tbl)
print(gini)
