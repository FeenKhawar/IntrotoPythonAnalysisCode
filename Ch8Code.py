import pandas as pd
import numpy as np

# # 8.1: Heirarchial Indexing
# data1 = pd.Series(np.random.uniform(size=9),
#                  index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"],
#                         [1, 2, 3, 1, 3, 1, 2, 2, 3]])
# print(data1)
# print(data1.index)
# print(data1["b"])
# print(data1["b":"c"])
# print(data1.loc[["b", "d"]])
# print(data1.loc[:, 2]) # Remember this is loc not iloc, so its finding any value that has a index of 2
# print(data1)
# print(data1.unstack()) # How to make this data into a DataFrame
# print(data1.unstack().stack()) # Possibly how to make a DataFrame a series, it is the inverse of unstack()

# frame1 = pd.DataFrame(np.arange(12).reshape((4, 3)),
#                      index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
#                      columns=[["Ohio", "Ohio", "Colorado"],
#                               ["Green", "Red", "Green"]])
# # Note that for frame1, it only says Ohio once because it shares the name with Green and Red
# # If Red had a different state (or name other than Ohio) attached to it, it that name would show up
# print(frame1)
# frame1.index.names = ["key1", "key2"]
# frame1.columns.names = ["state", "color"]
# print(frame1)
# # “Be careful to note that the index names "state" and "color" are not part of the row labels (the frame.index values).”
# print(frame1.index.nlevels)
# print(frame1["Ohio"]) 
# # A MultiIndex can be created by itself and then reused; the columns in the preceding DataFrame with level names could also be created like this:
# pd.MultiIndex.from_arrays([["Ohio", "Ohio", "Colorado"],
#                            ["Green", "Red", "Green"]],
#                            names=["state", "color"])

# # Reordering and Sorting Levels
# frame1 = pd.DataFrame(np.arange(12).reshape((4, 3)),
#                      index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
#                      columns=[["Ohio", "Ohio", "Colorado"],
#                               ["Green", "Red", "Green"]])
# print(frame1)
# frame1.index.names = ["key1", "key2"]
# frame1.columns.names = ["state", "color"]
# print(frame1)
# print(frame1.swaplevel("key1", "key2"))
# print(frame1.sort_index(level=1))
# print(frame1.swaplevel(0, 1).sort_index(level=0)) # 0 and 1 refer to "key1" and "key2"
# # “Data selection performance is much better on hierarchically
# # indexed objects if the index is lexicographically sorted starting with
# # the outermost level—that is, the result of calling sort_index(level=0) or sort_index().”

# # Summary Statistics by Level
# frame1 = pd.DataFrame(np.arange(12).reshape((4, 3)),
#                      index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
#                      columns=[["Ohio", "Ohio", "Colorado"],
#                               ["Green", "Red", "Green"]])
# print(frame1)
# frame1.index.names = ["key1", "key2"]
# frame1.columns.names = ["state", "color"]
# print(frame1)
# print(frame1.groupby(level="key2").sum())
# # print(frame1.groupby(level="color", axis="columns").sum()) # Something is wrong with this code, not sure what

# # Indexing with a DataFrame's columns
# frame2 = pd.DataFrame({"a": range(7), "b": range(7, 0, -1),
#                        "c": ["one", "one", "one", "two", "two",
#                              "two", "two"],
#                              "d": [0, 1, 2, 0, 1, 2, 3]})
# print(frame2)
# frame2.set_index(["c", "d"], drop=False)
# print(frame2)
# frame2 = frame2.set_index(["c", "d"])
# print(frame2)
# print(frame2.reset_index())

# # 8.2: Combining and merging Datasets

# # Database-Style DataFrame Joins
# df1 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "a", "b"],
#                     "data1": pd.Series(range(7), dtype="Int64")})
# df2 = pd.DataFrame({"key": ["a", "b", "d"],
#                     "data2": pd.Series(range(3), dtype="Int64")})
# print(df1, "\n" * 2, df2)
# # Take note that he uses "pandas's Int64 extension type for nullable integers, discussed in Section 7.3, 'Extension DataTypes'"
# print(pd.merge(df1, df2))
# print(pd.merge(df1, df2, on="key"))

# df3 = pd.DataFrame({"lkey": ["b", "b", "a", "c", "a", "a", "b"],
#                     "data1": pd.Series(range(7), dtype="Int64")})
# df4 = pd.DataFrame({"rkey": ["a", "b", "d"],
#                     "data2": pd.Series(range(3), dtype="Int64")})
# print(df3, "\n" * 2, df4)
# print(pd.merge(df3, df4, left_on="lkey", right_on="rkey"))
# print(pd.merge(df1, df2, how="outer"))
# print(pd.merge(df3, df4, left_on="lkey", right_on="rkey", how="outer"))
# # Using "outer" shows the values that have keys that appear in only one dataset (not both)
# # Without "outer" or some sort of equivalent, the pd.merge will only give values of keys that are found in both datasets

# df5 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"],
#                     "data1": pd.Series(range(6), dtype="Int64")})
# df6 = pd.DataFrame({"key": ["a", "b", "a", "b", "d"],
#                     "data2": pd.Series(range(5), dtype="Int64")})
# print(df5, "\n" * 2, df6)
# print(pd.merge(df5, df6, on="key", how="left"))
# print(pd.merge(df5, df6, on="key", how="right"))
# print(pd.merge(df5, df6, how="inner"))

# left1 = pd.DataFrame({"key1": ["foo", "foo", "bar"],
#                      "key2": ["one", "two", "one"],
#                      "lval": pd.Series([1, 2, 3], dtype='Int64')})
# right1 = pd.DataFrame({"key1": ["foo", "foo", "bar", "bar"],
#                       "key2": ["one", "one", "one", "two"],
#                       "rval": pd.Series([4, 5, 6, 7], dtype='Int64')})
# print(left1, "\n" * 2, right1)
# print(pd.merge(left1, right1, on=["key1", "key2"], how="outer"))
# # When you’re joining columns on columns, the indexes on the passed
# # DataFrame objects are discarded. If you 
# # need to preserve the index values, you can use reset_index to append the index to the columns.”
# print(pd.merge(left1, right1, on="key1", how="outer"))
# print(pd.merge(left1, right1, on="key1"))
# print(pd.merge(left1, right1, on="key1", suffixes=("_left", "_right")))

# left2 = pd.DataFrame({"key": ["a", "b", "a", "a", "b", "c"],
#                       "value": pd.Series(range(6), dtype="Int64")})
# right2 = pd.DataFrame({"group_val": [3.5, 7]}, index=["a", "b"])
# print(left2, "\n" * 2, right2)
# print(pd.merge(left2, right2, left_on="key", right_index=True))
# print(pd.merge(left2, right2, left_on="key", right_index=True, how="outer")) # OR
# print(left2.join(right2, on="key"))

# lefth = pd.DataFrame({"key1": ["Ohio", "Ohio", "Ohio",
#                                "Nevada", "Nevada"],
#                                "key2": [2000, 2001, 2002, 2001, 2002],
#                                "data": pd.Series(range(5), dtype="Int64")})
# righth_index = pd.MultiIndex.from_arrays(
#     [
#         ["Nevada", "Nevada", "Ohio", "Ohio", "Ohio", "Ohio"],
#         [2001, 2000, 2000, 2000, 2001, 2002]
#     ]
# )
# print(lefth, "\n" * 2, righth_index)
# righth = pd.DataFrame({"event1": pd.Series([0, 2, 4, 6, 8, 10], dtype="Int64", index=righth_index), 
#                        "event2": pd.Series([1, 3, 5, 7, 9, 11], dtype="Int64", index=righth_index)})
# print(righth)
# print(pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True))
# print(pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True, how="outer"))

# left3 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],
#                      index=["a", "c", "e"],
#                      columns=["Ohio", "Nevada"]).astype("Int64")
# right3 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],
#                       index=["b", "c", "d", "e"],
#                       columns=["Missouri", "Alabama"]).astype("Int64")
# print(left3, "\n" * 2, right3)
# print(pd.merge(left3, right3, how="outer", left_index=True, right_index=True))
# # DataFrame has a join instance
# # method to simplify merging by index. It can also be used to combine many DataFrame objects having the same or similar indexes but
# # nonoverlapping columns. In the prior example, we could have
# # written:
# print(left3.join(right3, how="outer"))

# another = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
#                        index=["a", "c", "e", "f"],
#                        columns=["New York", "Oregon"])
# print(another)
# print(left3.join([right3, another]))
# print(left3.join([right3, another], how="outer"))

# # Concatenating Along an Axis
# arr1 = np.arange(12).reshape((3, 4))
# print(arr1)
# print(np.concatenate([arr1, arr1], axis=1))
# print(np.concatenate([arr1, arr1], axis=0))

# s1 = pd.Series([0, 1], index=["a", "b"], dtype="Int64")
# s2 = pd.Series([2, 3, 4], index=["c", "d", "e"], dtype="Int64")
# s3 = pd.Series([5, 6], index=["f", "g"], dtype="Int64")
# print(str(s1) + "\n" + str(s2) + "\n" + str(s3))
# print(pd.concat([s1, s2, s3]))
# print(pd.concat([s1, s2, s3], axis="columns"))
# s4 = pd.concat([s1, s3])
# print(s4)
# print(pd.concat([s1, s4], axis="index"))
# print(pd.concat([s1, s4], axis="columns"))
# print(pd.concat([s1, s4], axis="columns", join="inner"))

# result1 = pd.concat([s1, s1, s3], keys=["one", "two", "three"])
# print(result1)
# print(result1.unstack())
# print(pd.concat([s1, s2, s3], axis="columns", keys=["one", "two", "three"]))
# print(pd.concat([s1, s2, s3], keys=["one", "two", "three"]))

# df1 = pd.DataFrame(np.arange(6).reshape(3, 2), index=["a", "b", "c"],
#                    columns=["one", "two"])
# df2 = pd.DataFrame(5 + np.arange(4).reshape(2, 2), index=["a", "c"],
#                    columns=["three", "four"])
# print(df1, "\n" * 2, df2)
# print(pd.concat([df1, df2], axis="columns", keys=["level1", "level2"]))
# print(pd.concat([df1, df2], keys=["level1", "level2"]))
# # In 2 lines of code above, notice what they keys are doing and what axis="columns" does
# # “Here the keys argument is used to create a hierarchical index where the first
# # level can be used to identify each of the concatenated DataFrame objects.”
# print(pd.concat({"level1": df1, "level2": df2}, axis="columns"))
# print(pd.concat({"level1": df1, "level2": df2}))
# print(pd.concat([df1, df2], axis="columns", keys=["level1", "level2"], names=["upper", "lower"]))

# df3 = pd.DataFrame(np.random.standard_normal((3, 4)),
#                    columns=["a", "b", "c", "d"])
# df4 = pd.DataFrame(np.random.standard_normal((2, 3)),
#                    columns=["b", "d", "a"])
# print(df3, "\n" * 2, df4)
# print(pd.concat([df3, df4])) # ignore_index=False by default
# print(pd.concat([df3, df4], ignore_index=True))

# # Combining Data with Overlap
# a = pd.Series([np.nan, 2.5, 0.0, 3.5, 4.5, np.nan],
#               index=["f", "e", "d", "c", "b", "a"])
# b = pd.Series([0., np.nan, 2., np.nan, np.nan, 5.],
#               index=["a", "b", "c", "d", "e", "f"])
# print(a, "\n" * 2, b)
# print(np.where(pd.isna(a), b, a))
# print(a.combine_first(b))

# df5 = pd.DataFrame({"a": [1., np.nan, 5., np.nan],
#                     "b": [np.nan, 2., np.nan, 6.],
#                     "c": range(2, 18, 4)})
# df6 = pd.DataFrame({"a": [5., 4., np.nan, 3., 7.],
#                     "b": [np.nan, 3., 4., 6., 8.]})
# print(df5, "\n" * 2, df6)
# print(df5.combine_first(df6))
# # Note that combine_first does not add values, but simply replaces NaN with a possible actual value between two datasets

# # 8.3: Reshaping and Pivoting

# # Reshaping with Hierarchial Indexing
# data2 = pd.DataFrame(np.arange(6).reshape((2, 3)),
#                     index=pd.Index(["Ohio", "Colorado"], name="state"),
#                     columns=pd.Index(["one", "two", "three"],
#                                      name="number"))
# print(data2)
# result2 = data2.stack()
# print(result2)
# print(result2.unstack()) # This gives the original data2 layout
# print(result2.unstack(level=0)) # Flips the unstack between index and columns
# print(result2.unstack(level="state")) # Also flips the unstack between index and columns
# print(result2.unstack(level=1)) # Gives original data2 layout
# print(result2.unstack(level="number")) # Gives original data2 layout
# df7 = pd.DataFrame({"left": result2, "right": result2 + 5},
#                    columns=pd.Index(["left", "right"], name="side"))
# print(df7)
# print(df7.unstack(level="state"))
# print(df7.unstack(level="state").stack(level="side"))

# # s5 = pd.Series([0, 1, 2, 3], index=["a", "b", "c", "d"], dtype="Int64")
# # s6 = pd.Series([4, 5, 6], index=["c", "d", "e"], dtype="Int64")
# # print(s5, "\n" * 2, s6)
# # data3 = pd.concat([s5, s6], keys=["one", "two"])
# # print(data3)
# # print(data3.unstack())
# # print(data3.unstack().stack())
# # print(data3.unstack().stack(dropna=False)) # This keeps the NA values/indexes

# # Pivoting "Long" to "Wide" Format
# data4 = pd.read_csv("/Users/faheemkhawar/Downloads/pydata-book-3rd-edition/examples/macrodata.csv")
# # The pf.read_csv is reading a file for a MacBook, not Windows, change as needed for code to work
# data4 = data4.loc[:, ["year", "quarter", "realgdp", "infl", "unemp"]]
# print(data4.head())
# periods = pd.PeriodIndex(year=data4.pop("year"),
#                          quarter=data4.pop("quarter"),
#                          name="date")
# # Given a warning to use PeriodIndex.from_fields instead as constructing PeriodIndex
# # from fields is deprecated, not sure what this means, will ignore it
# # The .pop "returns a column while deleteing it from the DataFrame at the same time"
# print(periods)
# data4.index = periods.to_timestamp("D")
# print(data4.head())
# data4 = data4.reindex(columns=["realgdp", "infl", "unemp"])
# data4.columns.name = "item"
# print(data4.head())
# print(data4.columns)
# long_data = (data4.stack().reset_index().rename(columns={0: "value"}))
# print(long_data[:10])
# pivoted = long_data.pivot(index="date", columns="item", values="value")
# print(pivoted.head())
# long_data["value2"] = np.random.standard_normal(len(long_data))
# print(long_data[:10])
# pivoted = long_data.pivot(index="date", columns="item")
# print(pivoted.head())
# print(pivoted["value"].head())
# unstacked = long_data.set_index(["date", "item"]).unstack(level="item")
# print(unstacked.head())

# # Pivoting "Wide" to "Long" Format
# df8 = pd.DataFrame({"key": ["foo", "bar", "baz"],
#                     "A": [1, 2, 3],
#                     "B": [4, 5, 6],
#                     "C": [7, 8, 9]})
# print(df8)
# melted = pd.melt(df8, id_vars="key")
# print(melted)
# reshaped = melted.pivot(index="key", columns="variable", values="value")
# print(reshaped)
# print(reshaped.reset_index())
# print(pd.melt(df8, id_vars="key", value_vars=["A", "B"]))
# print(pd.melt(df8, id_vars="key", value_vars=["A", "B", "C"]))
# print(pd.melt(df8, value_vars=["A", "B", "C"]))
# print(pd.melt(df8, value_vars=["key", "A", "B"]))