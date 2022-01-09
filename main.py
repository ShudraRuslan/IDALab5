import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules

warnings.filterwarnings('ignore')


def getProcessedData():
    df = pd.read_csv('Var_8_Market_Basket_Analysis_2.csv', sep='\t')
    values_list = df.values.tolist()
    for i in range(len(values_list)):
        values_list[i] = values_list[i][0].split(sep=',')
        values_list[i].pop()

    te = TransactionEncoder()
    te_ary = te.fit_transform(values_list)
    transactions = pd.DataFrame(te_ary, columns=te.columns_)
    return transactions


def aprioriAlgo(min_support, transactions):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print('Max number of items in sets: ', max(frequent_itemsets['length']))
    print(frequent_itemsets)
    return frequent_itemsets


def fpGrowthAlgo(min_support, transactions):
    frequent_itemsets = fpgrowth(transactions, min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print('Max number of items in sets: ', max(frequent_itemsets['length']))
    print(frequent_itemsets.head())
    return frequent_itemsets


def associationRules(frequent_itemsets):
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)
    rules = rules[(rules['lift'] >= 1)]
    print(rules[['antecedents', 'consequents', 'lift']])
    return rules


transactions = getProcessedData()
# print(transactions)

min_support = 0.1
# frequent_itemsets = aprioriAlgo(min_support, transactions)
frequent_itemsets = fpGrowthAlgo(min_support, transactions)
rules = associationRules(frequent_itemsets)

ant = [list(val)[0] for val in rules['antecedents'].values]
con = [list(val)[0] for val in rules['consequents'].values]
res = pd.DataFrame(list(zip(ant, con)), columns=['antecedents', 'consequents'])
print(res)
