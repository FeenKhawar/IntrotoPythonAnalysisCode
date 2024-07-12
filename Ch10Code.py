import numpy as np
import pandas as pd
import statsmodels.api as sm
from io import StringIO

# DataFrame below used many times in chapter
df = pd.DataFrame({"key1" : ["a", "a", None, "b", "b", "a", None],
                    "key2" : pd.Series([1, 2, 1, 2, 1, None, 1], dtype="Int64"),
                    "data1" : np.random.standard_normal(7),
                    "data2" : np.random.standard_normal(7)})

# # To understand the line of code below more (making own aggregation functions),
# # Look at page 330 in the book, page 348 of pdf
def peak_to_peak(arr):
    return arr.max() - arr.min()

tips = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/tips.csv")
tips["tip_pct"] = tips["tip"] / tips["total_bill"]

# # Data Aggregation and Group Operations

# # 10.1: How to Think About Group Operations
# print(df) 
# grouped1 = df["data1"].groupby(df["key1"])
# print(grouped1)
# print(grouped1.mean())
# means = df["data1"].groupby([df["key1"], df["key2"]]).mean()
# print(means)
# print(means.unstack())
# states = np.array(["OH", "CA", "CA", "OH", "OH", "CA", "OH"])
# years = [2005, 2005, 2006, 2005, 2006, 2005, 2006]
# print(df["data1"].groupby([states, years]).mean())
# print(df.groupby("key1").mean())
# # print(df.groupby("key2").mean()) # This line of code does not work for some reason
# print(df.groupby(["key1", "key2"]).mean())
# print(df.groupby(["key1", "key2"]).size())
# print(df.groupby("key1", dropna=False).size())
# print(df.groupby(["key1", "key2"], dropna=False).size())
# print(df.groupby(["key1", "key2"], dropna=False).count())
# print(df.groupby("key1").count())
# print(df.groupby("key1").size())
# # .count() gives the number of nonvalue values in each group
# # .size() gives the number of distinct values of the thing being "groupby"-ed

# # Iterating over Groups
# print(df)
# for name, group in df.groupby("key1"):
#     print(name)
#     print(group)
# for (k1, k2), group in df.groupby(["key1", "key2"]):
#     print((k1, k2))
#     print(group)
# pieces = {name: group for name, group in df.groupby("key1")}
# print(pieces["b"])
# grouped2 = df.groupby({"key1": "key", "key2": "key",
#                         "data1": "data", "data2": "data"}, axis="columns")
# print(grouped2)
# for group_key, group_values in grouped2:
#     print(group_key)
#     print(group_values)

# # Selecting a Column or Subset of Columns
# # df.groupby("key1")["data1"]
# # df.groupby("key1")[["data2"]]
# #       are conveniences for:
# # df["data1"].groupby(df["key1"])
# # df[["data2"]].groupby(df["key1"])
# print(df.groupby(["key1", "key2"])[["data2"]].mean())
# print(df["data2"].groupby([df["key1"], df["key2"]]).mean())
# # Line 65 and 64 give the same output
# s_grouped1 = df.groupby(["key1", "key2"])["data2"]
# s_grouped2 = df["data2"].groupby([df["key1"], df["key2"]])
# print(s_grouped1)
# print(s_grouped2)
# print(s_grouped1.mean())
# print(s_grouped2.mean())
# # Line 67 and 68 also do the same thing (making line 69 and 70, line 71 and 72 do the same thing)

# # Grouping with Dictionaries and Series
# people = pd.DataFrame(np.random.standard_normal((5, 5)),
#                       columns=["a", "b", "c", "d", "e"],
#                       index=["Joe", "Steve", "Wanda", "Jill", "Trey"])
# print(people)
# people.iloc[2:3, [1,2]] = np.nan
# print(people)
# mapping = {"a": "red", "b": "red", "c": "blue", "d": "blue", "e": "red", "f" : "orange"}
# print(mapping)
# by_column = people.groupby(mapping, axis="columns")
# print(by_column)
# print(by_column.sum())
# map_series = pd.Series(mapping)
# print(map_series)
# print(people.groupby(map_series, axis="columns").count())

# # Grouping with Functions
# people = pd.DataFrame(np.random.standard_normal((5, 5)),
#                       columns=["a", "b", "c", "d", "e"],
#                       index=["Joe", "Steve", "Wanda", "Jill", "Trey"])
# print(people)
# print(people.groupby(len).sum())

# # Grouping by Index Levels
# columns = pd.MultiIndex.from_arrays([["US", "US", "US", "JP", "JP"],
#                                      [1, 3, 5, 1, 3]],
#                                      names=["cty", "tenor"])
# print(columns)
# hief_df = pd.DataFrame(np.random.standard_normal((4, 5)), columns=columns)
# print(hief_df)
# print(hief_df.groupby(level="cty", axis="columns").count())

# # 10.2: Data Aggregation
# print(df)
# grouped = df.groupby("key1")
# print(grouped)
# print(grouped["data1"].nsmallest(2))
# print(grouped.agg(peak_to_peak)) 
# # To understand the 3 lines of code above more (making own aggregation functions),
# # Look at page 330 in the book, page 348 of pdf
# print(grouped.describe())

# # Column-Wise and Multiple Function Application
# print(tips.head())
# print(tips.head())
# grouped = tips.groupby(["day", "smoker"])
# print(grouped)
# grouped_pct = grouped["tip_pct"]
# print(grouped_pct)
# print(grouped_pct.mean)
# print(grouped_pct.agg("mean"))
# print(grouped_pct.agg(["mean", "std", peak_to_peak]))
# # Looks like why using .groupby, we need to use .agg to find stuff like mean and standard deviation
# print(grouped_pct.agg([("average", "mean"), ("stdev", np.std)]))
# # The line of code above does essentially the same as line 127
# # Line 129 is simply giving names to the columns, it is passing a list of (name, function) tuples
# functions = ["count", "mean", "max"]
# result = grouped[["tip_pct", "total_bill"]].agg(functions)
# print(result)
# print(result["tip_pct"])
# ftuples = [("Average", "mean"), ("Variance", np.var)]
# print(grouped[["tip_pct", "total_bill"]].agg(ftuples))
# print(grouped.agg({"tip" : np.max, "size" : "sum"}))
# print(grouped.agg({"tip_pct" : ["min", "max", "mean", "std"], "size" : "sum"}))

# # Returning Aggregated Data Without Row Indexes
# print(tips.head())
# print(tips.groupby(["day", "smoker"], as_index=False).mean())
# # Line of code above does not work, not sure why

# # 10.3: Apply: General split-apply-combine
# print(tips.head())
# def top(df, n=5, column="tip_pct"):
#     return df.sort_values(column, ascending=False)[:n]
# print(top(tips, n=6))
# print(tips.groupby("smoker").apply(top))
# print(tips.groupby(["smoker", "day"]).apply(top, n=1, column="total_bill"))
# result = tips.groupby("smoker")["tip_pct"].describe()
# print(result)
# print(result.unstack("smoker"))
# # Inside GroupBy, when you invoke a method like describe, it is actually just a shortcut for:
# # def f(group):
# #     return group.describe()
# # grouped.apply(f)

# # Suppressing the Group Keys
# def top(df, n=5, column="tip_pct"):
#     return df.sort_values(column, ascending=False)[:n]
# print(tips.groupby("smoker").apply(top))
# print(tips.groupby("smoker", group_keys=False).apply(top))

# # Quantile and Bucket Analysis
# frame = pd.DataFrame({"data1": np.random.standard_normal(1000),
#                       "data2": np.random.standard_normal(1000)})
# print(frame.head())
# quartiles = pd.cut(frame["data1"], 4)
# print(quartiles.head(10))
# def get_stats(group):
#     return pd.DataFrame({"min": group.min(), "max": group.max(),
#                          "count": group.count(), "mean": group.mean()})
# grouped = frame.groupby(quartiles)
# print(grouped.apply(get_stats))
# # The near-same result from line 178 can be done more simply with the line of code below:
# print(grouped.agg(["min", "max", "count", "mean"]))
# quartiles_samp = pd.qcut(frame["data1"], 4, labels=False)
# print(quartiles_samp.head())
# grouped = frame.groupby(quartiles_samp)
# print(grouped.apply(get_stats))

# # Example: Filling Missing Values with Group-Specific Values
# s = pd.Series(np.random.standard_normal(6))
# s[::2] = np.nan
#  # The [::2] selects every second element of the Series starting at the first element
# print(s)
# s = s.fillna(s.mean())
# print(s)
# states = ["Ohio", "New York", "Vermont", "Florida",
#           "Oregon", "Nevada", "California", "Idaho"]
# group_key = ["East", "East", "East", "East",
#              "West", "West", "West", "West"]
# data = pd.Series(np.random.standard_normal(8), index=states)
# print(data)
# data[["Vermont", "Nevada", "Idaho"]] = np.nan
# print(data)
# print(data.groupby(group_key).size())
# print(data.groupby(group_key).count())
# print(data.groupby(group_key).mean())
# def fill_mean(group):
#     return group.fillna(group.mean())
# print(data.groupby(group_key).apply(fill_mean))
# fill_values = {"East": 0.5, "West": -1}
# def fill_func(group):
#     return group.fillna(fill_values[group.name])
# print(data.groupby(group_key).apply(fill_func))

# # Example: Random Sampling and Permutation
# suits = ["H", "S", "C", "D"] # Hearts, Spades, Clubs, Diamonds
# card_val = (list(range(1, 11)) + [10] * 3) * 4
# base_names = ["A"] + list(range(2, 11)) + ["J", "K", "Q"]
# cards = []
# for suit in suits:
#  cards.extend(str(num) + suit for num in base_names)
# deck = pd.Series(card_val, index=cards)
# print(deck.head(13))
# def draw(deck, n=5):
#    return deck.sample(n)
# print(draw(deck))
# def get_suit(card):
#    return card[-1] # Last letter is suit
# print(deck.groupby(get_suit).apply(draw, n=2)) 
# print(deck.groupby(get_suit, group_keys=False).apply(draw, n=2))
# # This was a fairly difficult example and it was difficult to grasp all the code within it
# # Likely need to come back to this as random sampling will be important
# # Page 344 in book, page 362 in pdf

# # Example: Group Weighted Average and Correlation
# df2 = pd.DataFrame({"category": ["a", "a", "a", "a",
#                                 "b", "b", "b", "b"],
#                                 "data": np.random.standard_normal(8),
#                                 "weights": np.random.uniform(size=8)})
# print(df2)
# grouped = df2.groupby("category")
# def get_wavg(group):
#     return np.average(group["data"], weights=group["weights"])
# print(grouped.apply(get_wavg))
# close_px = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/stock_px.csv", 
#                        parse_dates=True, index_col=0)
# print(close_px.tail(4))
# print(close_px.info())
# def spx_corr(group):
#     return group.corrwith(group["SPX"])
# rets = close_px.pct_change().dropna()
# def get_year(x):
#     return x.year
# by_year = rets.groupby(get_year)
# print(by_year.apply(spx_corr))
# def corr_aapl_msft(group):
#     return group["AAPL"].corr(group["MSFT"])
# print(by_year.apply(corr_aapl_msft))

# # Example: Group-Wise Linear Regression
# close_px = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/stock_px.csv", 
#                        parse_dates=True, index_col=0)
# rets = close_px.pct_change().dropna()
# def get_year(x):
#     return x.year
# by_year = rets.groupby(get_year)
# def regress(data, yvar=None, xvars=None):
#     Y = data[yvar]
#     X = data[xvars]
#     X["intercept"] = 1.
#     result = sm.OLS(Y, X).fit()
#     return result.params
# print(by_year.apply(regress, yvar="AAPL", xvars=["SPX"]))

# # 10.4: Group Transforms and "Unwrapped" GroupBys
# df3 = pd.DataFrame({'key': ['a', 'b', 'c'] * 4, 'value': np.arange(12.)})
# print(df3)
# g = df3.groupby('key')['value']
# print(g.mean())
# def get_mean(group):
#     return group.mean()
# print(g.transform(get_mean))
# print(g.transform('mean'))
# def times_two(group):
#     return group * 2
# print(df3.transform(times_two))
# print(g.transform(times_two))
# def get_ranks(group):
#     return group.rank(ascending=False)
# print(g.transform(get_ranks))
# def normalize(x):
#     return (x - x.mean()) / x.std()
# print(g.transform(normalize)) # OR
# print(g.apply(normalize)) # To achieve the same thing (I think)
# print(g.transform('mean'))
# normalized = (df3["value"] - g.transform('mean')) / g.transform('std')
# print(normalized)

# # Pivot Tables and Cross-Tabulation
# print(tips.head())
# # print(tips.pivot_table(index=["day", "smoker"]))
# # print(tips.groupby(["day","smoker"]).mean())
# # The 2 lines of code above does not work, cannot convert string to numeric (the "time" column)
# print(tips.pivot_table(index=["time", "day"], columns="smoker", values=["tip_pct", "size"]))
# print(tips.pivot_table(index=["time", "day"], columns="smoker", values=["tip_pct", "size"], margins=True))
# # Here, the All values are means without taking into account smoker versus non
# # -smoker (the All columns) or any of the two levels of grouping on the rows (the All row)
# print(tips.pivot_table(index=["time", "smoker"], columns="day", values="tip_pct", aggfunc=len, margins=True))
# print(tips.pivot_table(index=["time", "size", "smoker"], columns="day", values="tip_pct", fill_value=0))

# # Cross-Tabulations: Crosstab
# data = """Sample Nationality Handedness
# 1 USA Right-handed
# 2 Japan Left-handed
# 3 USA Right-handed
# 4 Japan Right-handed
# 5 Japan Left-handed
# 6 Japan Right-handed
# 7 USA Right-handed
# 8 USA Left-handed
# 9 Japan Right-handed
# 10 USA Right-handed"""
# data = pd.read_table(StringIO(data), sep="\s+")
# print(data)
# print(pd.crosstab(data["Nationality"], data["Handedness"]))
# print(pd.crosstab(data["Nationality"], data["Handedness"], margins=True))
# print(pd.crosstab([tips["time"], tips["day"]], tips["smoker"]))
# print(pd.crosstab([tips["time"], tips["day"]], tips["smoker"], margins=True))