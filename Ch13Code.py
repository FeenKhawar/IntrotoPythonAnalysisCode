import numpy as np
import pandas as pd
import json
from collections import defaultdict
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

path = "D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/bitly_usagov/example.txt"
with open(path) as f:
    records = [json.loads(line) for line in f]
time_zones = [rec["tz"] for rec in records if "tz" in rec]

# # 13.1: Bitly Data from 1.USA.gov
# with open(path) as f:
#     print(f.readline())
# with open(path) as f:
#     records = [json.loads(line) for line in f]
# print(records[0])

# # Counting Time Zones in Pure Python
# print(time_zones[:10])
# def get_counts(sequence):
#     counts = {}
#     for x in sequence:
#         if x in counts:
#             counts[x] += 1
#         else:
#             counts[x] = 1
#     return counts
# def get_counts2(sequence):
#     counts = defaultdict(int) # values will initialize to 0
#     for x in sequence:
#         counts[x] += 1
#     return counts
# # get_counts2 is a more advanced way of doing get_counts
# counts = get_counts(time_zones)
# print(counts["America/New_York"])
# len(time_zones)
# def top_counts(count_dict, n=10):
#     value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
#     value_key_pairs.sort()
#     return value_key_pairs[-n:]
# print(top_counts(counts))
# # Simpler way to do top_counts:
# counts = Counter(time_zones)
# print(counts.most_common(10))

# # Counting Time Zones with pandas
# frame = pd.DataFrame(records)
# print(frame.info())
# print(frame.head())
# print(frame["tz"].head())
# tz_counts = frame["tz"].value_counts()
# print(tz_counts.head())
# # Code above is a much easier and faster way to find the top-most used time-zones,
# # Compared with all the code in the section above this one
# clean_tz = frame["tz"].fillna("Missing")
# clean_tz[clean_tz == ""] = "Unknown"
# tz_counts = clean_tz.value_counts()
# print(tz_counts.head())
# subset = tz_counts.head()
# sns.barplot(y=subset.index, x=subset.to_numpy())
# plt.show()
# print(frame["a"][1])
# print(frame["a"][50])
# print(frame["a"][51][:50])
# results = pd.Series([x.split()[0] for x in frame["a"].dropna()])
# print(results.head())
# print(results.value_counts().head(8))
# cframe = frame[frame["a"].notna()].copy()
# cframe["os"] = np.where(cframe["a"].str.contains("Windows"), "Windows", "Not Windows")
# print(cframe["os"].head(5))
# by_tz_os = cframe.groupby(["tz", "os"])
# agg_counts = by_tz_os.size().unstack().fillna(0)
# print(agg_counts.head())
# indexer = agg_counts.sum("columns").argsort()
# print(indexer.values[:10])
# count_subset = agg_counts.take(indexer[-10:])
# print(count_subset)
# print(agg_counts.sum("columns").nlargest(10))
# # Line of code above does the same that count_subset shows, but easier and faster to do
# count_subset = count_subset.stack()
# count_subset.name = "total"
# print(count_subset.head(10))
# count_subset = count_subset.reset_index()
# print(count_subset.head(10))
# sns.barplot(x="total", y="tz", hue="os", data=count_subset)
# plt.show()
# def norm_total(group):
#     group["normed_total"] = group["total"] / group["total"].sum()
#     return group
# results = count_subset.groupby("tz").apply(norm_total)
# sns.barplot(x="normed_total", y="tz", hue="os", data=results)
# plt.show()
# # We could have computed the normalized sum more efficiently by using the trans form method with groupby:
# g = count_subset.groupby("tz")
# results2 = count_subset["total"] / g["total"].transform("sum")

# # 13.2: MovieLens 1M Dataset
# unames = ["user_id", "gender", "age", "occupation", "zip"]
# users = pd.read_table("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/movielens/users.dat",
#                       sep="::", header=None, names=unames, engine="python")
# rnames = ["user_id", "movie_id", "rating", "timestamp"]
# ratings = pd.read_table("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/movielens/ratings.dat",
#                         sep="::", header=None, names=rnames, engine="python")
# mnames = ["movie_id", "title", "genres"]
# movies = pd.read_table("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/movielens/movies.dat",
#                        sep="::",header=None, names=mnames, engine="python")
# print(users.head(5))
# print(ratings.head(5))
# print(movies.head(5))
# print(ratings)
# data = pd.merge(pd.merge(ratings, users), movies)
# print(data)
# print(data.iloc[0])
# mean_ratings = data.pivot_table("rating", index="title", columns="gender", aggfunc="mean")
# print(mean_ratings.head(5))
# ratings_by_title = data.groupby("title").size()
# print(ratings_by_title.head())
# active_titles = ratings_by_title.index[ratings_by_title >= 250]
# print(active_titles)
# mean_ratings = mean_ratings.loc[active_titles]
# print(mean_ratings)
# top_female_ratings = mean_ratings.sort_values("F", ascending=False)
# print(top_female_ratings.head())

# # Measuring Rating Disagreement
# unames = ["user_id", "gender", "age", "occupation", "zip"]
# users = pd.read_table("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/movielens/users.dat",
#                       sep="::", header=None, names=unames, engine="python")
# rnames = ["user_id", "movie_id", "rating", "timestamp"]
# ratings = pd.read_table("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/movielens/ratings.dat",
#                         sep="::", header=None, names=rnames, engine="python")
# mnames = ["movie_id", "title", "genres"]
# movies = pd.read_table("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/movielens/movies.dat",
#                        sep="::",header=None, names=mnames, engine="python")
# data = pd.merge(pd.merge(ratings, users), movies)
# mean_ratings = data.pivot_table("rating", index="title", columns="gender", aggfunc="mean")
# ratings_by_title = data.groupby("title").size()
# active_titles = ratings_by_title.index[ratings_by_title >= 250]
# mean_ratings = mean_ratings.loc[active_titles]
# mean_ratings["diff"] = mean_ratings["M"] - mean_ratings["F"]
# sorted_by_diff = mean_ratings.sort_values("diff")
# print(sorted_by_diff.head())
# print(sorted_by_diff[::-1].head())
# rating_std_by_title = data.groupby("title")["rating"].std()
# rating_std_by_title = rating_std_by_title.loc[active_titles]
# print(rating_std_by_title.head())
# print(rating_std_by_title.sort_values(ascending=False)[:10])
# print(movies["genres"].head())
# print(movies["genres"].head().str.split("|"))
# movies["genre"] = movies.pop("genres").str.split("|")
# # The ".pop" essential removes the old "genres" column, and allows for a new column called "genre"
# print(movies.head())
# movies_exploded = movies.explode("genre")
# print(movies_exploded[:10])
# ratings_with_genre = pd.merge(pd.merge(movies_exploded, ratings), users)
# print(ratings_with_genre.iloc[0])
# print(ratings_with_genre.iloc[1])
# genre_ratings = (ratings_with_genre.groupby(["genre", "age"])["rating"].mean().unstack("age"))
# print(genre_ratings[:10])

# # 13.3: US Baby Names 1880-2010
# names1880 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/babynames/yob1880.txt",
#                         names=["name", "sex", "births"])
# print(names1880)
# print(names1880.groupby("sex")["births"].sum())
# pieces = []
# for year in range(1880, 2011):
#     path = f"D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/babynames/yob{year}.txt"
#     frame = pd.read_csv(path, names=["name", "sex", "births"])
#     frame["year"] = year # "Add a column for the year"
#     pieces.append(frame)
# names = pd.concat(pieces, ignore_index=True) # "Concatenate everything into a single DataFrame"
# # "Remember that concat combines the DataFrame objects by row by default"
# # "You have to pass ignore_index=True because 
# # we’re not interested in preserving the original row numbers returned from pandas.read_csv"
# print(names)
# total_births = names.pivot_table("births", index="year", columns="sex", aggfunc=sum)
# print(total_births.tail())
# total_births.plot(title="Total births by sex and year")
# plt.show()
# def add_prop(group):
#     group["prop"] = group["births"] / group["births"].sum()
#     return group
# names = names.groupby(["year", "sex"]).apply(add_prop)
# print(names)
# names.set_index(['year', 'sex'], inplace=True)
# names_reset = names.reset_index()
# grouped = names_reset.groupby(["year", "sex"])["prop"].sum()
# print(names.groupby(["year", "sex"])["prop"].sum())
# # Line 189 - 191 were needed to deal with the fact that both "year" and "sex" were an index 
# # level and a column label (which is ambiguous)
# def get_top1000(group):
#     return group.sort_values("births", ascending=False)[:1000]
# grouped = names.groupby(["year", "sex"])
# top1000 = grouped.apply(get_top1000)
# print(top1000.head())
# # "We can drop the group index since we don’t need it for our analysis:"
# top1000 = top1000.reset_index(drop=True)
# print(top1000.head())

# # Analyzing Naming Trends
# # Note: Something was happening where I had to reset and alter the index due to the fact that there being
# # "sex" and "year" in both the index and column was messing with other functions (couldn't distinguish between them)
# # After fixing it, for some reason the "sex" of the columns also dropped, and the later functions couldn't
# # work properly without it, I tried fixing this but I could not. I ultimately ended up skipping everything
# # Up until 13.4, page 457 of the book, page 475 of the pdf
# names1880 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/babynames/yob1880.txt",
#                         names=["name", "sex", "births"])
# pieces = []
# for year in range(1880, 2011):
#     path = f"D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/babynames/yob{year}.txt"
#     frame = pd.read_csv(path, names=["name", "sex", "births"])
#     frame["year"] = year # "Add a column for the year"
#     pieces.append(frame)
# names = pd.concat(pieces, ignore_index=True) # "Concatenate everything into a single DataFrame"
# total_births = names.pivot_table("births", index="year", columns="sex", aggfunc=sum)
# def add_prop(group):
#     group["prop"] = group["births"] / group["births"].sum()
#     return group
# names = names.groupby(["year", "sex"]).apply(add_prop)
# # names.set_index(['year', 'sex'], inplace=True)
# # names_reset = names.reset_index()
# abc = {"year": "year2", "sex": "sex2"}
# names = names.rename(columns=abc, copy=False)
# names.rename(index={"year": "year2", "sex": "sex2"}, inplace=True)

# print(names.head())
# names.groupby(["year2", "sex2"])["prop"].sum()
# # names.set_index(['year', 'sex'], inplace=True)
# # names_reset = names.reset_index()
# # grouped = names_reset.groupby(["year", "sex"])["prop"].sum()
# def get_top1000(group):
#     return group.sort_values("births", ascending=False)[:1000]
# grouped = names.groupby(["year2", "sex2"])
# top1000 = grouped.apply(get_top1000)
# top1000 = top1000.reset_index(drop=True)
# # top1000 = top1000.drop(top1000.columns[[0, 1]], axis=1)
# print(top1000.head())
# boys = top1000[top1000["sex2"] == "M"]
# girls = top1000[top1000["sex2"] == "F"]
# total_births = top1000.pivot_table("births", index="year2", columns="name", aggfunc=sum)
# print(total_births.head(10))
# print(total_births.info())

# # 13.4: USDA Food Database
# db = json.load(open("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/usda_food/database.json"))
# print(len(db))
# print(db[0].keys())
# print(db[0]["nutrients"][0])
# nutrients = pd.DataFrame(db[0]["nutrients"])
# print(nutrients.head(7))
# info_keys = ["description", "group", "id", "manufacturer"]
# info = pd.DataFrame(db, columns=info_keys)
# print(info.head())
# print(info.info())
# # print(pd.value_counts(info["group"])[:10]) # Old way to do this, newer and better way below
# print(pd.Series(info["group"]).value_counts()[:10])
# nutrients = []
# for rec in db:
#     fnuts = pd.DataFrame(rec["nutrients"])
#     fnuts["id"] = rec["id"]
#     nutrients.append(fnuts)
# nutrients = pd.concat(nutrients, ignore_index=True)
# print(nutrients)
# print(nutrients.duplicated().sum()) # "Number of duplicates"
# nutrients = nutrients.drop_duplicates()
# col_mapping = {"description" : "food", "group" : "fgroup"}
# info = info.rename(columns=col_mapping, copy=False)
# print(info)
# col_mapping = {"description" : "nutrient", "group" : "nutgroup"}
# nutrients = nutrients.rename(columns=col_mapping, copy=False)
# print(nutrients)
# ndata = pd.merge(nutrients, info, on="id")
# print(ndata.info())
# print(ndata)
# print(ndata.iloc[30000])
# result = ndata.groupby(["nutrient", "fgroup"])["value"].quantile(0.5)
# result["Zinc, Zn"].sort_values().plot(kind="barh")
# plt.show()
# by_nutrient = ndata.groupby(["nutgroup", "nutrient"])
# def get_maximum(x):
#     return x.loc[x.value.idxmax()]
# max_foods = by_nutrient.apply(get_maximum)[["value", "food"]]
# max_foods["food"] = max_foods["food"].str[:50]
# print(max_foods.loc["Amino Acids"]["food"])

# # 13.5 Federal Election Commission Database
# fec = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/fec/P00000001-ALL.csv",
#                   low_memory=False)
# print(fec.info())
# print(fec.iloc[123456])
# unique_cands = fec["cand_nm"].unique()
# print(unique_cands)
# print(unique_cands[2])
# parties = {"Bachmann, Michelle": "Republican",
#     "Cain, Herman": "Republican",
#     "Gingrich, Newt": "Republican",
#     "Huntsman, Jon": "Republican",
#     "Johnson, Gary Earl": "Republican",
#     "McCotter, Thaddeus G": "Republican",
#     "Obama, Barack": "Democrat",
#     "Paul, Ron": "Republican",
#     "Pawlenty, Timothy": "Republican",
#     "Perry, Rick": "Republican",
#     "Roemer, Charles E. 'Buddy' III": "Republican",
#     "Romney, Mitt": "Republican",
#     "Santorum, Rick": "Republican"}
# print(fec["cand_nm"][123456:123461])
# print(fec["cand_nm"][123456:123461].map(parties))
# fec["party"] = fec["cand_nm"].map(parties) # "Add it as a column"
# print(fec["party"].value_counts())
# print((fec["contb_receipt_amt"] > 0).value_counts())
# fec = fec[fec["contb_receipt_amt"] > 0]
# fec_mrbo = fec[fec["cand_nm"].isin(["Obama, Barack", "Romney, Mitt"])]
# # Donation Statistics by Occupation and Employer
# print(fec["contbr_occupation"].value_counts()[:10])
# occ_mapping = {"INFORMATION REQUESTED PER BEST EFFORTS" : "NOT PROVIDED",
#     "INFORMATION REQUESTED" : "NOT PROVIDED",
#     "INFORMATION REQUESTED (BEST EFFORTS)" : "NOT PROVIDED",
#     "C.E.O.": "CEO"}
# def get_occ(x):
#     # "If no mapping provied, return x"
#     return occ_mapping.get(x, x)
# fec["contbr_occupation"] = fec["contbr_occupation"].map(get_occ)
# emp_mapping = {"INFORMATION REQUESTED PER BEST EFFORTS" : "NOT PROVIDED",
#     "INFORMATION REQUESTED" : "NOT PROVIDED",
#     "SELF" : "SELF-EMPLOYED",
#     "SELF EMPLOYED" : "SELF-EMPLOYED"}
# def get_emp(x):
#     # "If no mapping provided, return x"
#     return emp_mapping.get(x, x)
# fec["contbr_employer"] = fec["contbr_employer"].map(get_emp)
# by_occupation = fec.pivot_table("contb_receipt_amt",
#                                 index="contbr_occupation",
#                                 columns="party", aggfunc="sum")
# over_2mm = by_occupation[by_occupation.sum(axis="columns") > 2000000]
# print(over_2mm)
# over_2mm.plot(kind="barh")
# plt.show()
# def get_top_amounts(group, key, n=5):
#     totals = group.groupby(key)["contb_receipt_amt"].sum()
#     return totals.nlargest(n)
# grouped = fec_mrbo.groupby("cand_nm")
# print(grouped.apply(get_top_amounts, "contbr_occupation", n=7))
# print(grouped.apply(get_top_amounts, "contbr_employer", n=10))
# # Bucketing Donation Amounts
# bins = np.array([0, 1, 10, 100, 1000, 10000, 100_000, 1_000_000, 10_000_000])
# labels = pd.cut(fec_mrbo["contb_receipt_amt"], bins)
# print(labels)
# grouped = fec_mrbo.groupby(["cand_nm", labels])
# print(grouped.size().unstack(level=0))
# bucket_sums = grouped["contb_receipt_amt"].sum().unstack(level=0)
# normed_sums = bucket_sums.div(bucket_sums.sum(axis="columns"), axis="index")
# print(normed_sums)
# normed_sums[:-2].plot(kind="barh")
# plt.show()
# # Donation Statistics by State
# grouped = fec_mrbo.groupby(["cand_nm", "contbr_st"])
# totals = grouped["contb_receipt_amt"].sum().unstack(level=0).fillna(0)
# totals = totals[totals.sum(axis="columns") > 100000]
# print(totals.head())
# percent = totals.div(totals.sum(axis="columns"), axis="index")
# print(percent.head(10))