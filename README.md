```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
```


```python
data = pd.read_csv('a.csv')
x = data.iloc[:,:]
data = pd.read_csv('train.csv')
y = data.iloc[:,-1]
data = pd.read_csv('t1.csv')
x_test = data.iloc[:,:]
data2 = pd.read_csv('gender_submission.csv')
y_test = data2.iloc[:,-1]
print(x)
print(y)
print(x_test)
print(y_test)
```

         Unnamed: 0  0  1  2  3  4  5          6  7  8        9
    0             0  0  0  1  0  1  3  22.000000  1  0   7.2500
    1             1  1  0  0  1  0  1  38.000000  1  0  71.2833
    2             2  0  0  1  1  0  3  26.000000  0  0   7.9250
    3             3  0  0  1  1  0  1  35.000000  1  0  53.1000
    4             4  0  0  1  0  1  3  35.000000  0  0   8.0500
    ..          ... .. .. .. .. .. ..        ... .. ..      ...
    886         886  0  0  1  0  1  2  27.000000  0  0  13.0000
    887         887  0  0  1  1  0  1  19.000000  0  0  30.0000
    888         888  0  0  1  1  0  3  29.699118  1  2  23.4500
    889         889  1  0  0  0  1  1  26.000000  0  0  30.0000
    890         890  0  1  0  0  1  3  32.000000  0  0   7.7500
    
    [891 rows x 11 columns]
    0      0
    1      1
    2      1
    3      1
    4      0
          ..
    886    0
    887    1
    888    0
    889    1
    890    0
    Name: Survived, Length: 891, dtype: int64
         Unnamed: 0  0  1  2  3  4  5         6  7  8         9
    0             0  0  1  0  0  1  3  34.50000  0  0    7.8292
    1             1  0  0  1  1  0  3  47.00000  1  0    7.0000
    2             2  0  1  0  0  1  2  62.00000  0  0    9.6875
    3             3  0  0  1  0  1  3  27.00000  0  0    8.6625
    4             4  0  0  1  1  0  3  22.00000  1  1   12.2875
    ..          ... .. .. .. .. .. ..       ... .. ..       ...
    413         413  0  0  1  0  1  3  30.27259  0  0    8.0500
    414         414  1  0  0  1  0  1  39.00000  0  0  108.9000
    415         415  0  0  1  0  1  3  38.50000  0  0    7.2500
    416         416  0  0  1  0  1  3  30.27259  0  0    8.0500
    417         417  1  0  0  0  1  3  30.27259  1  1   22.3583
    
    [418 rows x 11 columns]
    0      0
    1      1
    2      0
    3      0
    4      1
          ..
    413    0
    414    1
    415    0
    416    0
    417    0
    Name: Survived, Length: 418, dtype: int64
    


```python
from xgboost import XGBClassifier
classifier = XGBClassifier(use_label_encoder=False)
classifier.fit(x,y)
```

    [22:34:15] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.4.0/src/learner.cc:1095: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=4, num_parallel_tree=1, random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', use_label_encoder=False,
                  validate_parameters=1, verbosity=None)




```python
y_pred = classifier.predict(x_test)
print(y_pred)
```

    [0 0 0 0 1 0 1 0 1 0 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 0 1 0 0 0 0 0 0 0 1 0 1
     0 0 1 0 1 0 1 1 0 0 0 1 1 0 0 1 1 0 0 0 0 0 1 0 0 0 1 0 1 1 0 0 1 1 0 1 0
     1 0 0 1 1 1 1 0 0 0 0 0 1 1 1 1 1 0 1 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 0
     1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
     0 0 1 0 0 0 0 0 1 1 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 0 1 1 0 0 1 0 1
     0 1 0 0 0 0 0 1 0 1 0 1 1 0 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 1 0 1 0 1 1
     1 0 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 0 1 1 0 0 0 0 0 0 1
     0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 1 1 1 0 0 0 0 1 0 1 0 0 1 0
     1 0 0 0 1 0 0 0 1 1 0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 0 0
     1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 1 0 1 1 0 1 0 0 0 1 0
     0 1 0 0 1 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0
     0 0 1 0 1 0 0 1 0 0 0]
    


```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
m=accuracy_score(y_test, y_pred)
print(m)
```

    [[247  19]
     [ 28 124]]
    0.8875598086124402
    


```python
df = pd.DataFrame(y_pred)
df.to_excel('Titanic.xlsx')
```
