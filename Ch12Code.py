import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score

data = pd.DataFrame({'x0': [1, 2, 3, 4, 5],
                     'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                     'y': [-1.5, 0., 3.6, 1.3, -2.]})

# # 12.1: Interfacing Between pandas and Model Code
# print(data)
# print(data.columns)
# print(data.to_numpy())
# df2 = pd.DataFrame(data.to_numpy(), columns=['one', 'two', 'three'])
# print(df2)
# df3 = data.copy()
# df3['strings'] = ['a', 'b', 'c', 'd', 'e']
# print(df3)
# print(df3.to_numpy())
# model_cols = ['x0', 'x1']
# print(data.loc[:, model_cols].to_numpy())
# data['category'] = pd.Categorical(['a', 'b', 'a', 'a', 'b'], categories=['a', 'b'])
# print(data)
# dummies = pd.get_dummies(data.category, prefix='category')
# data_with_dummies = data.drop('category', axis=1).join(dummies)
# print(data_with_dummies)

# # 12.2: Creating Model Descriptions with Patsy
# print(data)
# y, X = patsy.dmatrices('y ~ x0 + x1', data)
# print(y)
# print(X)
# print(np.asarray(y))
# print(np.asarray(X))
# print(patsy.dmatrices('y ~ x0 + x1 + 0', data))
# print(patsy.dmatrices('y ~ x0 + x1 + 0', data)[1])
# coef, resid, _, _ = np.linalg.lstsq(X, y)
# print(coef)
# coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)
# print(coef)

# # Data Transformations in Patsy Formulas
# y, X = patsy.dmatrices('y ~ x0 + np.log(np.abs(x1) + 1)', data)
# print(X)
# y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
# print(X)
# new_data = pd.DataFrame({'x0': [6, 7, 8, 9],
#                          'x1': [3.1, -0.5, 0, 2.3],
#                          'y': [1, 2, 3, 4]})
# new_X = patsy.build_design_matrices([X.design_info], new_data)
# print(new_X)
# y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)
# print(X)

# # Categorical Data and Patsy
# data2 = pd.DataFrame({'key1': ['a', 'a', 'b', 'b', 'a', 'b', 'a', 'b'],
#                       'key2': [0, 1, 0, 1, 0, 1, 0, 0],
#                       'v1': [1, 2, 3, 4, 5, 6, 7, 8],
#                       'v2': [-1, 0, 2.5, -0.5, 4.0, -1.2, 0.2, -1.7]})
# print(data2)
# y, X = patsy.dmatrices('v2 ~ key1', data2)
# print(y)
# print(X)
# y, X = patsy.dmatrices('v2 ~ key1 + 0', data2)
# print(y)
# print(X)
# y, X = patsy.dmatrices('v2 ~ C(key2)', data2)
# print(y)
# print(X)
# data2['key2'] = data2['key2'].map({0: 'zero', 1: 'one'})
# print(data2)
# y, X = patsy.dmatrices('v2 ~ key1 + key2', data2)
# print(X)
# y, X = patsy.dmatrices('v2 ~ key1 + key2 + key1:key2', data2)
# print(X)

# # 12.3: Introduction to statsmodels
# rng = np.random.default_rng(seed=12345)
# def dnorm(mean, variance, size=1):
#     if isinstance(size, int):
#         size = size,
#     return mean + np.sqrt(variance) * rng.standard_normal(*size)
# N = 100
# X = np.c_[dnorm(0, 0.4, size=N),
#  dnorm(0, 0.6, size=N),
#  dnorm(0, 0.2, size=N)]
# eps = dnorm(0, 0.1, size=N)
# beta = [0.1, 0.3, 0.5]
# y = np.dot(X, beta) + eps
# print(X[:5])
# print(y[:5])
# X_model = sm.add_constant(X)
# print(X_model[:5])
# model = sm.OLS(y, X)
# results = model.fit()
# print(results.params)
# print(results.summary())
# data2 = pd.DataFrame(X, columns = ['col0', 'col1', 'col2'])
# data2['y'] = y
# print(data2[:5])
# results = smf.ols('y ~ col0 + col1 + col2', data=data2).fit()
# print(results.params)
# print(results.tvalues)
# print(data2[:5])
# print(results.predict(data2[:5]))

# # Estimating Time Series Processes
# rng = np.random.default_rng(seed=12345)
# def dnorm(mean, variance, size=1):
#     if isinstance(size, int):
#         size = size,
#     return mean + np.sqrt(variance) * rng.standard_normal(*size)
# init_x = 4
# values = [init_x, init_x]
# N = 1000
# b0 = 0.8
# b1 = -0.4
# noise = dnorm(0, 0.1, N)
# for i in range(N):
#     new_x = values[-1] * b0 + values[-2] + noise[i]
#     values.append(new_x)
# MAXLAGS = 5
# model = AutoReg(values, MAXLAGS)
# results = model.fit()
# print(results)
# print(results.params)

# # 12.4: Introduction to scikit-learn
# train = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/titanic/train.csv")
# test = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/titanic/test.csv")
# print(train.head())
# # Check if any missing data
# print(train.isna().sum()) 
# print(test.isna().sum())
# # Libraries like statsmodels and scikit-learn generally cannot be fed missing data
# impute_value = train['Age'].median()
# train['Age'] = train['Age'].fillna(impute_value)
# test['Age'] = test['Age'].fillna(impute_value)
# train['IsFemale'] = (train['Sex'] == 'female').astype(int)
# test['IsFemale'] = (test['Sex'] == 'female').astype(int)
# predictors = ['Pclass', 'IsFemale', 'Age']
# X_train = train[predictors].to_numpy()
# X_test = test[predictors].to_numpy()
# y_train = train['Survived'].to_numpy()
# print(X_train[:5])
# print(y_train[:5])
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_predict = model.predict(X_test)
# print(y_predict[:10])
# # If you had the true values for the test dataset, you could compute an accuracy percentage or some other error metric:
# # (y_true == y_predict).mean()
# model_cv = LogisticRegressionCV(Cs=10)
# model_cv.fit(X_train, y_train)
# model2 = LogisticRegression(C=10)
# scores = cross_val_score(model, X_train, y_train, cv=4)
# print(scores)

# # 12.5: Conclusion
# # This book is focused especially on data wrangling, but there are many others dedicated to modeling and data science tools. 
# # Some excellent ones are:
# # • Introduction to Machine Learning with Python by Andreas Müller and Sarah Guido (O’Reilly)
# # • Python Data Science Handbook by Jake VanderPlas (O’Reilly)
# # • Data Science from Scratch: First Principles with Python by Joel Grus (O’Reilly)
# # • Python Machine Learning by Sebastian Raschka and Vahid Mirjalili (Packt Publishing)
# # • Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron (O’Reilly)