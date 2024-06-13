import numpy as np
import pandas as pd

# The biggest difference between the series and dataframe is that
# Series: 1D, homogeneous, single column (exluding index), single index, single column of data or a single time series
# DataFrame: 2D, heterogeneous, multiple columns, two indices (rows and columns), full datasets with multiple columns and rows

# obj = pd.Series([4, 7, -5, 3])
# print(obj, obj.array, obj.index)
# obj.index = ["Bob", "Steve", "Jeff", "Ryan"]
# print(obj)

# obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
# print(obj2, obj2.array, obj2.index)

# print(obj2["a"])
# print(obj2[["c", "a", "d"]])
# print(obj2[1:3])
# print(obj[1:3]) # This still seems to work even though the index are technically names now

# sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
# obj3 = pd.Series(sdata)
# print(obj3)
# print(obj3.to_dict())
# states = ["California", "Ohio", "Oregon", "Texas"]
# obj4 = pd.Series(sdata, index=states)
# print(obj4)
# print(obj3 + obj4)
# obj4.name = "population"
# obj4.index.name = "state"
# print(obj4)

# data = {"state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
#         "year": [2000, 2001, 2002, 2001, 2002, 2003],
#         "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
# frame = pd.DataFrame(data)
# print(data)
# print(frame) # Can be frame.head() and frame.tail() for the top/bottom 5, respectively
# print(pd.DataFrame(data, columns=["year", "state", "pop"]))
# frame2 = pd.DataFrame(data, columns=["year", "state", "pop", "debt"])
# print(frame2, "\n", frame2.columns)

# print(frame2["state"])
# print(frame2.year)
# # 2 Ways to get a column in a DataFrame as a Series

# print(frame2.loc[1])
# frame2["debt"] = 16.5
# print(frame2)
# frame2["debt"] = np.arange(6.) # I guess doing #. makes it a float, good to know
# print(frame2)

# val = pd.Series([-1.2, -1.5, -1.7], index=["two", "four", "five"])
# print(val)
# frame2["debt"] = val
# print(frame2)
# frame2["eastern"] = frame2["state"] == "Ohio"
# del frame2["eastern"]
# print(frame2.columns)

# The column returned from indexing a DataFrame is a view on the
# underlying data, not a copy. Thus, any in-place modifications to
# the Series will be reflected in the DataFrame. The column can be
# explicitly copied with the Series’s copy method.

# # Dictionary  of dictionaries
# populations = {"Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
#                "Nevada": {2001: 2.4, 2002: 2.9}}
# frame3 = pd.DataFrame(populations)
# print(frame3)
# # The outer dictionary keys are the columns, the inner dictionary keys are the rows
# # print(frame3.T)

# # Note that transposing discards the column data types if the col‐
# # umns do not all have the same data type, so transposing and then
# # transposing back may lose the previous type information. The col‐
# # umns become arrays of pure Python objects in this case.

# # print(pd.DataFrame(populations, index=[2001, 2002, 2003]))

# # # Dictionaries of Series
# # pdata = {"Ohio": frame3["Ohio"][:-1],
# #          "Nevada": frame3["Nevada"][:2]}
# # print(pd.DataFrame(pdata))
# frame3.index.name = "year"
# frame3.columns.name = "state"
# print(frame3)
# print(frame3.to_numpy())
# print(frame2.to_numpy())

# obj5 = pd.Series(np.arange(3), index=["a", "b", "c"])
# index = obj5.index
# print(index)
# print(index[1:])
# # Index objects are immutable and thus cannot be modified by the user

# labels = pd.Index(np.arange(3))
# print(labels)
# obj6 = pd.Series([1.5, -2.5, 0], index=labels)
# print(obj6)

# print(obj6.index is labels)

# To access rows, it seems like you do .index, while columns is .columns

# frame4 = pd.DataFrame(np.arange(9).reshape((3, 3)),
#                      index=["a", "c", "d"],
#                      columns=["Ohio", "Texas", "California"])
# print(frame4)
# frame5 = frame4.reindex(index=["a", "b", "c", "d"])
# print(frame5)
# states = ["Texas", "Utah", "California"]
# print(frame4.reindex(columns=states)) # OR
# print(frame4.reindex(states, axis="columns"))
# print(frame4.loc[["a", "d", "c"], ["California", "Texas"]]) # .loc: "Access a group of rows and columns by label(s) or a boolean array."

# obj7 = pd.Series(np.arange(5.), index=["a", "b", "c", "d", "e"])
# print(obj7)
# new_obj7 = obj7.drop("c")
# print(new_obj7)

# data2 = pd.DataFrame(np.arange(16).reshape((4, 4)),
#                      index=["Ohio", "Colorado", "Utah", "New York"],
#                      columns=["one", "two", "three", "four"])
# print(data2)
# print(data2.drop(index=["Colorado", "Ohio"]))
# print(data2.drop(columns=["two"]))
# print(data2.drop("two", axis=1))
# print(data2.drop(["two", "four"], axis="columns"))

# When trying to deal with multiple things (like .loc-ing multiple things), you usually use two brackets [[]]
# Using .loc is prefered because:
#    The reason to prefer loc is because of the different treatment of integers when
#    indexing with []. Regular []-based indexing will treat integers as labels if the index
#    contains integers, so the behavior differs depending on the data type of the index.

# With .loc, you have to be specific what you're trying to get, can't just say say number if the index is a letter(s), using .iloc disregards this I think

# You can also slice with labels, but it works differently from normal Python slicing in that the endpoint is inclusive

# print(data2["two"])
# print(data2[["three", "one"]])

# print(data2[:2])
# print(data2[data2["three"] > 5])

# print(data2 < 5)
# data3 = pd.DataFrame(data2.copy())
# data3[data3 < 5] = 0
# print(data2)
# print(data3)

# modified_data2 = data2.applymap(lambda x: 0 if x < 5 else x)
# print(modified_data2) # Another way to do the above without changing data2, using lamba
# Why use lambda? It allows one to concisely write a function without making a function with def
# lambda functions are often used for short, simple operations where defining a full function with def would be unnecessarily verbose.
# If we didn't use lambda, we would need to make a def function something like this:
# def replace_less_than_5(x):
#     if x < 5:
#         return 0
#     else:
#         return x
# modified_data2 = data2.applyman(replace_less_than_5)
# Clearly, using lambda is much easier

# Note, sometimes its a little hard to figure out when to use "=" or "==" with pandas

# .loc is used to select rows and columns by their labels
# .iloc is used to select rows and columns by their integer positions

# print(data2.loc["Colorado", ["two", "three"]])
# print(data2.loc[["Colorado", "Ohio"], ["two", "three"]])
# print(data2.iloc[[1, 2], [3, 0, 1]])

# Boolean arrays can be used with .loc but not .iloc
# print(data2.loc[data2.three >= 2])

# print(data2.iloc[:, :3][data2.three > 5])

# It is better to always try to use .loc and .iloc, for example, do data.iloc[-1] instead of data[-1] (if data[-1] even works, see page 149 (pdf page 167))
# Slicing with integers is always integer oriented

# data2.loc[:, "one"] = 1
# print(data2)
# data2.iloc[2] = 5
# print(data2)
# data2.iloc[:, 2] = 10
# print(data2)
# data2.loc[data2["four"] > 5, "four"] = 3 # This code makes any value in the "four" column that is above 5 to 3
# print(data2)
# data2.loc[data2["four"] > 4] = 9 # This code makes any row where the value in the "four" column that is above 4 to 9
# print(data2)

# mask = data2["four"] > 5
# print(mask)
# if mask.any():
#     data2["four"] = 4
# print(data2)
# This code sees if any value in the "four" column is above 5, and if there is, make the whole "four" column 4

# One way to make the columns list faster
# pd.DataFrame(np.arange(9.).reshape((3, 3)), columns=list("bcd"),
#              index=["Ohio", "Texas", "Colorado"])

# For both Series and DataFrames, if it try to add two Series/DataFrames and there is a value within the index/column that is not in both datasets, it will return NaN
# Do dataset1.add(dataset2, fill_value=0) to get around this
# When doing the whoel dataset minus a row of a dataset (called x), it'll do each row of the dataset minus the row selected (x)

# frame6 = pd.DataFrame(np.arange(12.).reshape((4, 3)),
#                      columns=list("bde"),
#                      index=["Utah", "Ohio", "Texas", "Oregon"])
# series = frame6.iloc[0]
# print(frame6 - series)
# series3 = frame6["d"]
# print(frame6.sub(series3, axis="index"))
# It looks like if you want to instead to what line 207 described by column, you use .sub

# frame7 = pd.DataFrame(np.random.standard_normal((4, 3)),
#                       columns=list("bde"),
#                       index=["Utah", "Ohio", "Texas", "Oregon"])
# print(frame7)
# print(np.abs(frame7))

# def f1(x):
#     return x.max() -x.min()

# print(frame7.apply(f1))
# print(frame7.apply(f1, axis="columns"))
# .apply is how to apply functions on one-dimensional arrays to each column or row

# def f2(x):
#     return pd.Series([x.min(), x.max()], index=["min", "max"])

# print(frame7.apply(f2))
# print(frame7.apply(f2, axis="columns"))
# How to use functions and .apply to return a Series with multiple values

# def my_format(x):
#     return f"{x:.2f}"

# print(frame7.applymap(my_format))
# How to do element-wise Python functions with .applymap

# print(frame7["e"].map(my_format))
# I think you use applymap because it is a Series we want to end up with, "Series has a map method for applying an element-wise function"

# obj8 = pd.Series(np.arange(4), index=["d", "a", "b", "c"])
# print(obj8)
# print(obj8.sort_index())

# frame8 = pd.DataFrame(np.arange(8).reshape((2, 4)),
#                       index=["three", "one"],
#                       columns=["d", "a", "b", "c"])
# print(frame8)
# print(frame8.sort_index(ascending=False))
# print(frame8.sort_index(axis="columns"))

# Two ways to sort by both columns and rows (indexes) (acsending part just for show, look at order of sort_index)
# print(frame8.sort_index().sort_index(axis="columns"))
# print(frame8.sort_index(axis="columns", ascending=False).sort_index())

# obj9 = pd.Series([4, 7, -3, 2])
# print(obj9)
# print(obj9.sort_values())

# obj10 = pd.Series([4, np.nan, 7, np.nan, -3, 2])
# print(obj10.sort_values())

# How to put missing values sorted at the start
# print(obj10.sort_values(na_position="first"))

# frame9 = pd.DataFrame({"b": [4, 7, -3, 2], "a": [0, 1, 0, 1]})
# print(frame9)
# print(frame9.sort_values("b"))
# print(frame9.sort_values(["a", "b"]))

# The rank() command is a little confusing. It basically first sorts the datapoints by its value, then does
# the order of value (starting at 1), cumulative (so if a value is in the first and second place in the sorted list, 1 + 2 = 3)
# divided by the number of times that value appears. Then it returns this value and replaces the original value in the given rank Series or DataFrame
# values that are the same will also have the same rank value I believe

# Duplicates of the same index/column value makes things complicated, if you try to deal with a certain row or column it'll give you everything related to that specific index/column value you're filtering by
# df = pd.DataFrame(np.random.standard_normal((5, 3)),
#                   index=["a", "a", "b", "b", "c"])
# print(df)
# print(df.loc["b"])
# print(df.loc["c"])

# df2 = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],
#                    [np.nan, np.nan], [0.75, -1.3]],
#                    index=["a", "b", "c", "d"],
#                    columns=["one", "two"])
# print(df2)
# print(df2.sum())
# print(df2.sum(skipna=False))
# print(df2.sum(axis="columns"))
# print(df2.sum(axis="columns", skipna=False))
# print(df2.idxmax())
# print(df2.idxmax(axis="columns"))
# print(df2.cumsum())
# print(df2.describe())
# print(df2.T.describe()) # This is how to use describe() by the columns

# obj11 = pd.Series(["a", "a", "b", "c"] * 4)
# print(obj11)
# print(obj11.describe())

# This is about correlation and covariance, page 169 in the book, page 187 of the pdf
# price = pd.read_pickle("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/yahoo_price.pkl")
# volume = pd.read_pickle("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/yahoo_volume.pkl")
# print(price)
# returns = price.pct_change()
# print(returns.tail())
# print(returns["MSFT"].corr(returns["IBM"]))
# print(returns["MSFT"].cov(returns["IBM"]))
# print(returns["MSFT"].corr(returns["IBM"]))
# print(returns.corr())
# print(returns.cov())
# print(returns.corrwith(returns["IBM"]))
# print(returns.corrwith(volume))

# obj12 = pd.Series(["c", "a", "d", "a", "a", "b", "b", "c", "c"])
# print(obj12)
# uniques = obj12.unique()
# print(uniques)
# print(obj12.value_counts())
# print(pd.value_counts(obj12.to_numpy(), sort=False))

# Two ways (kind of) to do the same thing
# mask2 = obj12.isin(["b", "c"])
# print(mask2)
# print(obj12.isin(["b", "c"]))
# print(obj12[mask2])
# print(obj12[obj12.isin(["b", "c"])])

# to_match = pd.Series(["c", "a", "b", "b", "c", "a"])
# unique_vals = pd.Series(["c", "b", "a"])
# indices = pd.Index(unique_vals).get_indexer(to_match)
# print(indices)

# data3 = pd.DataFrame({"Qu1": [1, 3, 4, 3, 4],
#                       "Qu2": [2, 3, 1, 2, 3],
#                       "Qu3": [1, 5, 2, 4, 4]})
# print(data3)
# print(data3["Qu1"].value_counts().sort_index())

# Need to use pd.value_counts to compute value counts for all columns using the apply method
# For some reason value_counts does not need "()" after it in the line of code below
# result = data3.apply(pd.value_counts).fillna(0)
# print(result)

# There is also a DataFrame.value_counts method, but it computes counts considering
# each row of the DataFrame as a tuple to determine the number of occurrences of each
# distinct row:
# data4 = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": [0, 0, 1, 0, 0]})
# print(data4)
# print(data4.value_counts())
# For some reason value_counts does not need "()" after it in the line of code below
# print(data4.apply(pd.value_counts))