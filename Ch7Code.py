import numpy as np
import pandas as pd
import re

# float_data1 = pd.Series([1.2, -3.5, np.nan, 0])
# print(float_data1)
# print(float_data1.isna())
# string_data1 = pd.Series(["aardvark", np.nan, None, "avocado"])
# print(string_data1)
# print(string_data1.isna())
# # NaN and None will be treated as essentially the same by pandas

# data1 = pd.Series([1, np.nan, 3.5, np.nan, 7])
# print(data1)
# print(data1.dropna())
# print(data1[data1.notna()])
# # Lines 14 ad 13 will show the same thing

# data2 = pd.DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
# print(data2)
# print(data2.dropna()) # This will drop an row that has NaN
# print(data2.dropna(how="all")) # Will only drop a row if that row is entirely NaN
# data2[4] = np.nan
# print(data2)
# print(data2.dropna(axis="columns", how="all")) # Will only drop a column if that column is entirely NaN

# df1 = pd.DataFrame(np.random.standard_normal((7, 3)))
# print(df1)
# df1.iloc[:4, 1] = np.nan
# df1.iloc[:2, 2] = np.nan
# print(df1)
# print(df1.dropna())
# print(df1.dropna(thresh=2)) # If a row has 2 (due to thresh=2) NaN's, drop that row

# # Filling in Missing Data
# print(df1.fillna(0))
# print(df1.fillna({1: 0.5, 2: 0}))

# df2 = pd.DataFrame(np.random.standard_normal((6, 3)))
# print(df2)
# df2.iloc[2:, 1] = np.nan
# df2.iloc[4:, 2] = np.nan
# print(df2)
# print(df2.fillna(method="ffill"))
# print(df2.fillna(method="ffill", limit=2))

# data2 = pd.Series([1., np.nan, 3.5, np.nan, 7])
# print(data2)
# print(data2.fillna(data2.mean()))

# # Removing Duplicates
# data3 = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"], "k2": [1, 1, 2, 3, 3, 4, 4]})
# print(data3)
# print(data3.duplicated())
# print(data3.drop_duplicates())
# data3["v1"] = range(7)
# print(data3)
# print(data3.drop_duplicates(subset=["k1"]))
# print(data3.drop_duplicates(subset=["k1"], keep="last"))
# print(data3.drop_duplicates(["k1", "k2"], keep="last"))

# # Transforming Data Using a Function or Mapping
# data4 = pd.DataFrame({"food": ["bacon", "pulled pork", "bacon",
#                                "pastrami", "corned beef", "bacon",
#                                "pastrami", "honey ham", "nova lox"],
#                                "ounces": [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
# print(data4)
# meat_to_animal = {
#  "bacon": "pig",
#  "pulled pork": "pig",
#  "pastrami": "cow",
#  "corned beef": "cow",
#  "honey ham": "pig",
#  "nova lox": "salmon"
# }
# data4["animal"] = data4["food"].map(meat_to_animal)
# print(data4)

# def get_animal(x):
#     return meat_to_animal[x]

# print(data4["food"].map(get_animal))
# print(data4["food"].apply(get_animal))
# Both will do the same thing, don't think that they always do the same thing though
# .applymap is for when you want to apply a defined function to a whole dataset (not a part of it) I think

# # Replacing Values
# data5 = pd.Series([1., -999., 2., -999., -1000., 3.])
# print(data5)
# print(data5.replace(-999, np.nan))
# print(data5.replace([-999 -1000], np.nan))
# print(data5.replace([-999, -1000], [np.nan, 0]))
# print(data5.replace({-999: np.nan, -1000: 0}))
# # The data.replace method is distinct from data.str.replace,
# # which performs element-wise string substitution. We look at these
# # string methods on Series later in the chapter.

# # Renaming Axis Indexes
# data6 = pd.DataFrame(np.arange(12).reshape((3, 4)),
#                      index=["Ohio", "Colorado", "New York"],
#                      columns=["one", "two", "three", "four"])
# print(data6)

# def transform(x):
#     return x[:4].upper()

# print(data6.index.map(transform))
# print(data6)
# data6.index = data6.index.map(transform) # This is to actually modify the DataFrame in place
# print(data6)
# print(data6.rename(index=str.title, columns=str.upper)) # Creating a transformed version without modifiying the original
# # str. gives the existing names, .title is maybe like .lower?
# data6.rename(index={"OHIO": "INDIANA"},
#              columns={"three": "peekaboo"})
# print(data6)

# # Discretization and Binning
# ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# bins = [18, 25, 35, 60, 100]
# age_categories = pd.cut(ages, bins)
# print(age_categories)
# print(age_categories.codes)
# print(age_categories.categories)
# print(age_categories.categories[0])
# print(pd.value_counts(age_categories))
# print(pd.cut(ages, bins, right=False))

# group_names = ["Youth", "YoungAdult", "MiddleAged", "Senior"] 
# print(pd.cut(ages, bins, labels=group_names))
# # Code above basically does age_categories, but now the categories are according to the group_names
# data7 = np.random.uniform(size=20)
# print(data7)
# print(pd.cut(data7, 4, precision=2)) # precision=2 sets the precision to two decimal points, 4 means 4 categories

# data8 = np.random.standard_normal(1000)
# quartiles = pd.qcut(data8, 4, precision=2) # pd.qcut ensures equally sized bins, something pd.cut does not do
# print(quartiles)
# print(pd.value_counts(quartiles))
# print(pd.qcut(data8, [0, 0.1, 0.5, 0.9, 1.]).value_counts())
# # The array in the code above are quartiles, one to 0%-10%, 10%-50% 50%-90%, and 90%-100%

# # Detecing and Filtering Outliers
# data9 = pd.DataFrame(np.random.standard_normal((1000, 4)))
# print(data9)
# print(data9.describe())
# print(pd.Series(data9.values.ravel()).describe()) # How to get one set of describe statistics for the whole dataset
# col = data9[2]
# print(col[col.abs() > 3])
# print(data9[(data9.abs() > 3).any(axis="columns")])
# data9[data9.abs() > 3] = np.sign(data9) * 3
# print(data9.describe())
# print(np.sign(data9).head())
# # The statement np.sign(data) produces 1 and –1 values based on whether the values in data are positive or negative

# # Permutation and Random Sampling
# df3 = pd.DataFrame(np.arange(5 * 7).reshape((5, 7)))
# print(df3)
# sampler = np.random.permutation(5)
# print(sampler)
# print(df3.take(sampler))
# print(df3.iloc[sampler])
# # Two lines of code above seem to do the same thing
# print(df3.take(sampler, axis=1)) # How to do it by the columns
# # I noticed sometimes it's hard to know if to do axis="columns" or axis = 1
# print(np.random.permutation(df3))
# column_sampler = np.random.permutation(7)
# print(column_sampler)
# print(df3.take(column_sampler, axis="columns"))
# print(df3.sample(n=3))
# choices = pd.Series([5, 7, -1, 6, 4])
# print(choices.sample(n=10, replace=True))

# # How to take 5 randomly selected values from a dataset
# flattened_df3 = df3.values.ravel() # Flatten the DataFrame to a 1D array
# random_values_of_df3 = np.random.choice(flattened_df3, size=5, replace=False)
# # "replace=False" ensures that the same value is not selected more than once.
# print(random_values_of_df3)

# # Computing Indicator/Dummy Variables
# df4 = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"],
#                     "data1": range(6)})
# print(df4)
# print(pd.get_dummies(df4["key"])) # True stands for 1, False stands for 0
# dummies1 = pd.get_dummies(df4["key"], prefix="key")
# df_with_dummy = df4[["data1"]].join(dummies1)
# print(dummies1)
# print(df_with_dummy)

# mnames = ["movie_id", "title", "genres"]
# movies = pd.read_table("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/movielens/movies.dat", sep="::",
#                        header=None, names=mnames, engine="python")

# print(movies[:10])
# dummies2 = movies["genres"].str.get_dummies("|")
# print(dummies2.iloc[:10, :6])
# movies_windic = movies.join(dummies2.add_prefix("Genre_"))
# print(movies_windic.iloc[0])

# # For much larger data, this method of constructing indicator vari‐
# # ables with multiple membership is not especially speedy. It would
# # be better to write a lower-level function that writes directly to a
# # NumPy array, and then wrap the result in a DataFrame.

# np.random.seed(12345) # to make the example repeatable
# values1 = np.random.uniform(size=10)
# print(values1)
# bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
# print(pd.cut(values1, bins))
# print(pd.get_dummies(pd.cut(values1, bins)))

# # 7.3: Extension Data Types
# s = pd.Series([1, 2, 3, None])
# print(s, "\n", s.isna(), "\n", s.dtype)
# print(s[3], s[3] is pd.NA)
# s = pd.Series([1, 2, 3, None], dtype=pd.Int64Dtype())
# print(s, s.isna())
# print(s[3], s[3] is pd.NA) 
# # Notice that in the second version of "s", "s[3] is pd.NA" is now True, while before it was False even if we expected it to be True
# # Could have also said "Int64" instead of "pd.Int64DType", capitalization is necessary
# # s = pd.Series([1, 2, 3, None], dtype="Int64")
# s = pd.Series(['one', 'two', None, 'three'], dtype=pd.StringDtype())
# print(s)
# # "These string arrays generally use much less memory and are frequently computation‐
# # ally more efficient for doing operations on large datasets.""

# df5 = pd.DataFrame({"A": [1, 2, None, 4],
#                     "B": ["one", "two", "three", None],
#                     "C": [False, None, False, True]})
# print(df5, "\n" ,df5.dtypes)
# df5["A"] = df5["A"].astype("Int64")
# df5["B"] = df5["B"].astype("string")
# df5["C"] = df5["C"].astype("boolean")
# print(df5, "\n", df5.dtypes)

# # 7.4: String Manipulation

# # Python Built-In String Object Methods
# val = "a,b, guido"
# print(val, "\n", val.split(","))
# print(val, "\n", str(val.split(",")))
# print(val + "\n" + str(val.split(",")))
# # The line above is how to get the output to have not spaces at the beginning after the first line of output
# pieces = [x.strip() for x in val.split(",")]
# print(pieces)
# pieces2 = []
# for x in val.split(","):
#     pieces2.append(x.strip())
# print(pieces2)
# # How to do line 243 in a different way
# first, second, third = pieces
# print(first + "::" + second + "::" + third)
# print("::".join(pieces))
# # Line 251 and 252 do the same thing, but line 252 is faster and a more "Pythonic" way
# print("guido" in val)
# print(val.index(","))
# print(val.index("b"))
# print(val.index("guido")) # Notice it gives the end point, to get range, do index, index plus length (len)
# print(val.count(","))
# print(val.replace(",", "::"))
# print(val.replace(",", ""))

# # Regular Expressions
# text1 = "foo bar\t baz \tqux"
# print(re.split(r"\s+", text1))
# regex1 = re.compile(r"\s+")
# print(regex1)
# print(regex1.split(text1))
# # The re.split(r"\s+", text) compiles (line 266) and then splits (line 268) all in one
# print(regex1.findall(text1))

# text2 = """Dave dave@google.com
# Steve steve@gmail.com
# Rob rob@gmail.com
# Ryan ryan@yahoo.com"""
# pattern = r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}"
# # re.IGNORECASE makes the regex case insensitive
# regex2 = re.compile(pattern, flags=re.IGNORECASE)
# print(regex2.findall(text2))
# m = regex2.search(text2)
# print(m)
# print(text2[m.start():m.end()])
# print(regex2.match(text2)) # You get "None" as the output because "it will match only if the pattern occurs at the start of the string"
# print(regex2.sub("REDACTED", text2))

# pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
# regex3 = re.compile(pattern, flags=re.IGNORECASE)
# n = regex3.match("wesm@bright.net")
# print(n)
# print(n.groups())
# print(regex3.findall(text2))
# print(regex3.sub(r"Username: \1, Domain: \2, Suffix: \3", text2))
# # Regex was completely confusing, did not really understand this

# # String Function in pandas
# data10 = {"Dave": "dave@google.com", "Steve": "steve@gmail.com", "Rob": "rob@gmail.com", "Wes": np.nan}
# print(data10)
# data10 = pd.Series(data10)
# print(data10)
# print(data10.isna())
# print(data10.str.contains("gmail"))
# # "String and regular expression methods can be applied (passing a lambda or other
# # function) to each value using data.map, but it will fail on the NA (null) values.
# # To cope with this, Series has array-oriented methods for string operations that skip
# # over and propagate NA values. These are accessed through Series’s str attribute"
# data_as_string_ext = data10.astype('string')
# print(data_as_string_ext)
# print(data_as_string_ext.str.contains("gmail"))
# pattern2 = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
# print(data10.str.findall(pattern2, flags=re.IGNORECASE))

# matches = data10.str.findall(pattern2, flags=re.IGNORECASE).str[0]
# print(matches)
# print(matches.str.get(1))
# print(data10.str[:5])
# print(data10.str.extract(pattern2, flags=re.IGNORECASE))
# # Line of code above is using: "The str.extract method will return the captured groups of a regular expression as a DataFrame"

# # 7.5: Categorical Data
# values = pd.Series(['apple', 'orange', 'apple', 'apple'] * 2)
# print(values)
# print(pd.unique(values))
# print(pd.value_counts(values))
# values = pd.Series([0, 1, 0, 0] * 2)
# dim = pd.Series(['apple', 'orange'])
# print(values)
# print(dim)
# print(dim.take(values))

# # Categorical Extension Type in pandas
# fruits = ['apple', 'orange', 'apple', 'apple'] * 2
# N = len(fruits)
# print(fruits, "\n", N)
# print(str(fruits) + "\n" + str(N)) # Do this to not get extra space at beginning of second line of output
# rng = np.random.default_rng(seed=12345)
# df6 = pd.DataFrame({'fruit': fruits,
#                    'basket_id': np.arange(N),
#                    'count': rng.integers(3, 15, size=N),
#                    'weight': rng.uniform(0, 4, size=N)},
#                    columns=['basket_id', 'fruit', 'count', 'weight'])
# # rng.integers gives whole integers, rng.uniform gives decimal (float) numbers
# print(df6)
# fruit_cat = df6['fruit'].astype('category')
# c = fruit_cat.array
# print(fruit_cat, "\n", c, "\n", type(c))
# print(c.categories)
# print(c.codes)
# print(dict(enumerate(c.categories)))
# print(df6["fruit"]) # dtype is object
# df6["fruit"] = df6["fruit"].astype('category')
# print(df6["fruit"]) # dtype is category

# my_categories = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
# print(my_categories)
# categories = ['foo', 'bar', 'baz']
# codes = [0, 1, 2, 0, 0, 1]
# my_cats_2 = pd.Categorical.from_codes(codes, categories)
# print(my_cats_2)
# ordered_cat = pd.Categorical.from_codes(codes, categories, ordered=True)
# print(ordered_cat)
# print(my_cats_2.as_ordered())

# # Computations with Categoricals
# rng = np.random.default_rng(seed=12345)
# draws = rng.standard_normal(1000)
# print(draws[:5])
# bins = pd.qcut(draws, 4)
# print(bins)
# print(pd.value_counts(bins))
# bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
# print(bins)
# print(pd.value_counts(bins))
# print(bins.codes[:10])
# bins = pd.Series(bins, name='quartile')
# print(bins)
# results = (pd.Series(draws)
#            .groupby(bins)
#            .agg(['count', 'min', 'max'])
#            .reset_index())
# # Look carefully at how results is done, useful to know for the future
# print(results)
# print(results['quartile'])

# # Better performance with categoricals
# N = 10_000_000
# labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))
# print(labels)
# categories = labels.astype('category')
# print(labels.memory_usage(deep=True))
# print(categories.memory_usage(deep=True))

# # Categorical Methods
# s = pd.Series(['a', 'b', 'c', 'd'] * 2)
# cat_s = s.astype('category')
# print(cat_s)
# print(cat_s.cat.codes)
# print(cat_s.cat.categories)
# # I'm pretty sure we use .cat. now because this is a Series, not a DataFrame
# actual_categories = ['a', 'b', 'c', 'd', 'e']
# cat_s2 = cat_s.cat.set_categories(actual_categories)
# print(cat_s2)
# print(cat_s.value_counts)
# print(cat_s2.value_counts)
# cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
# print(cat_s3)
# print(cat_s3.cat.remove_unused_categories())

# # Creating Dummy Variables for Modeling
# cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')
# print(cat_s)
# print(pd.get_dummies(cat_s))