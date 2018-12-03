import numpy as np
import pandas as pd
from bffs.bf import BF
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

def name(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Veriscolour'
    else:
        return 'Virginica'


if __name__ == "__main__":
    selector = BF(verbose=True)
    data = datasets.load_iris()
    dataX = pd.DataFrame(data=data.data,columns=data.feature_names)
    dataY = pd.DataFrame(data=data.target)
    dataY = dataY.rename(columns={0: 'Species'})
    # dataY['Species'] = dataY['Species'].apply(name)
    X = dataX.values
    y = dataY.values.squeeze(1)

    print()
    model1 = LogisticRegression()
    model1.fit(X, y)
    acc1 = model1.score(X, y)
    print("All Feature Acc", acc1)
    print()

    print("Apply Feature Selection by MIP")
    newX = selector.fit_transform(X, y)
    print(f"Selected {newX.shape[1]} features")
    print(selector.selected())
    print(selector.coef())
    model2 = LogisticRegression()
    model2.fit(newX, y)
    acc2 = model2.score(newX, y)
    print("Selected Feature Acc", acc2)
