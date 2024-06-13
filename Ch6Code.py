import pandas as pd
import numpy as np
import sys
import csv
import json
from lxml import objectify
import requests
import sqlite3
import sqlalchemy as sqla

# df1 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex1.csv")
# print(df1)

# df2 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex2.csv")
# print(df2)
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex2.csv", header=None))
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex2.csv", names=["a", "b", "c", "d", "message"]))
# names = ["a", "b", "c", "d", "message"]
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex2.csv", names=names, index_col="message"))
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex2.csv", names=names))
# A little weird, so far the index column has been all the way to the right, but by doing index_col="message", the index column moves all the way to the left

# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/csv_mindex.csv"))
# parsed = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/csv_mindex.csv", index_col=["key1", "key2"])
# print(parsed)

# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex3.txt"))
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex3.txt", sep="\s+"))
# Because there was one fewer column name than the number of data rows,
# pandas.read_csv infers that the first column should be the DataFrame’s index in
# this special case

# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex4.csv"))
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex4.csv", skiprows=[0, 2, 3]))

# result = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex5.csv")
# print(result)
# print(pd.isna(result))
# result = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex5.csv", na_values=["NULL"])
# result2 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex5.csv", keep_default_na=False)
# # result still gives the same dataset
# print(result)
# print(result2)
# print(result2.isna())
# result3 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex5.csv", keep_default_na=False, na_values=["NA"])
# print(result3)
# print(result3.isna())
# sentinels = {"message": ["foo", "NA"], "something": ["two"]}
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex5.csv", na_values=sentinels, keep_default_na=False))

# result4 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex6.csv")
# print(result4)
# print(pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex6.csv", nrows=20))

# chunker = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex6.csv", chunksize=1000)
# print(type(chunker))

# tot = pd.Series([], dtype='int64')
# for piece in chunker:
#     tot = tot.add(piece["key"].value_counts(), fill_value=0)
# tot = tot.sort_values(ascending=False)
# print(tot[:10])

# data1 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex5.csv")
# print(data1)
# print(data1.to_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/out.csv"))
# print(data1.to_csv(sys.stdout, sep="|"))
# print(data1.to_csv(sys.stdout, sep="|", na_rep="NULL"))
# print(data1.to_csv(sys.stdout, sep="|", na_rep="NULL", index=False, header=False))
# print(data1.to_csv(sys.stdout, sep="|", na_rep="NULL", index=False, columns=["a", "b", "c"]))

# f = open("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex7.csv")
# reader = csv.reader(f)
# for line in reader:
#     print(line)
# f.close()

# with open("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex7.csv") as f:
#     lines = list(csv.reader(f))
# header, values = lines[0], lines[1:]
# data_dict = {h: v for h, v in zip(header, zip(*values))}
# print(data_dict)

# A little confusing stuff about dealing with difficult datasets, look for page 187 in book, page 205 in pdf

# obj1 = """
# {"name": "Wes",
#  "cities_lived": ["Akron", "Nashville", "New York", "San Francisco"],
#  "pet": null,
#  "siblings": [{"name": "Scott", "age": 34, "hobbies": ["guitars", "soccer"]},
# {"name": "Katie", "age": 42, "hobbies": ["diving", "art"]}]
# }
# """

# print(obj1)
# result = json.loads(obj1)
# print(result)

# asjson = json.dumps(result)
# print(asjson)

# siblings = pd.DataFrame(result["siblings"], columns=["name", "age"])
# print(siblings)

# data2 = pd.read_json("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/example.json")
# print(data2)

# print(data2.to_json(sys.stdout))
# print(data2.to_json(sys.stdout, orient="records"))

# tables = pd.read_html("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/fdic_failed_bank_list.html")
# len(tables) # This checks how many tables pd.read_html obtained
# failures = tables[0] # This is NOT referring to the first row of the tables, but telling the program 
# #                      to make failures equal to the first (in this case, only) table obtained from pd.read_html
# print(failures.head())

# close_timestamps = pd.to_datetime(failures["Closing Date"])
# print(close_timestamps.dt.year.value_counts())

# Parsing XML with lxml.objectify
# path = "D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/mta_perf/Performance_MNR.xml"
# with open(path) as f:
#     parsed = objectify.parse(f)
# root = parsed.getroot()
# print(root)
# data3 = []
# skip_fields = ["PARENT_SEQ", "INDICATOR_SEQ",
#  "DESIRED_CHANGE", "DECIMAL_PLACES"]
# for elt in root.INDICATOR:
#     el_data3 = {}
#     for child in elt.getchildren():
#         if child.tag in skip_fields:
#             continue
#         el_data3[child.tag] = child.pyval
#     data3.append(el_data3)
# perf = pd.DataFrame(data3)
# print(perf.head())
# perf2 = pd.read_xml(path)
# print(perf2.head())
# # Fairly confusing to figure out compared to other data formats

# frame1 = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex1.csv")
# print(frame1)
# frame1.to_pickle("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/frame_pickle")
# print(pd.read_pickle("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/frame_pickle"))
# fec = pd.read_parquet('D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/datasets/fec/fec.parquet')
# print(fec)

# xlsx = pd.ExcelFile("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex1.xlsx")
# print(xlsx) # This doesn't work 
# print(xlsx.sheet_names)
# print(xlsx.parse(sheet_name="Sheet1"))
# print(xlsx.parse(sheet_name="Sheet1", index_col=0))

# frame2 = pd.read_excel("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex1.xlsx", sheet_name="Sheet1")
# print(frame2)

# writer = pd.ExcelWriter("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex2.xlsx")
# frame2.to_excel(writer, "Sheet1")
# writer._save() # You could also do frame2.to_excel("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/ex2.xlsx") to avoid ExcelWriter
# I think you use ._save

# HDF5 Format
# frame3 = pd.DataFrame({"a": np.random.standard_normal(100)})
# store = pd.HDFStore("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/mydata.h5")
# store["obj2"] = frame3
# store["obj2_col"] = frame3["a"]
# print(store)
# print(store["obj2_col"])
# print(store["obj2"])
# store.put("obj3", frame3, format="table")
# print(store.select("obj3", where=["index >= 10 and index <= 15"]))
# store.close()
# frame3.to_hdf("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/mydata.h5", "obj3", format="table")
# print(pd.read_hdf("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/mydata.h5", "obj3", where=["index < 5"]))
# I can delete HDF5 files I created like so:
# import os
# os.remove("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/mydata.h5")

# Interacting with Web APIs
# url = "https://api.github.com/repos/pandas-dev/pandas/issues"
# resp = requests.get(url)
# resp.raise_for_status()
# print(resp)
# # This apparently checks for HTTP errors
# data4 = resp.json()
# print(data4[0]["title"])
# issues = pd.DataFrame(data4, columns=["number", "title", "labels", "state"])
# print(issues)
# A little lost on what this is exactly, I guess this is for getting datasets from website APIs
# Website API: "A software intermediary that allows a web browser and web server to communicate with each other"

# # Interacting with Databases
# # SQL and SQL-relational databases are talked about in this section
# # Page 199 in the book, page 219 in the pdf
# query = """
# CREATE TABLE test
# (a VARCHAR(20), b VARCHAR(20),
# c REAL, d INTEGER
# );"""
# con = sqlite3.connect("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/mydata.sqlite")
# print(con.execute(query))
# con.commit()

# data5 = [("Atlanta", "Georgia", 1.25, 6),
#     ("Tallahassee", "Florida", 2.6, 3),
#     ("Sacramento", "California", 1.7, 5)]
# stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
# print(con.executemany(stmt, data5))
# con.commit()
# cursor = con.execute("SELECT * FROM test")
# rows = cursor.fetchall()
# print(rows)

# print(cursor.description)
# print(pd.DataFrame(rows, columns=[x[0] for x in cursor.description]))

# db = sqla.create_engine("sqlite:///mydata.sqlite")
# print(pd.read_sql("SELECT * FROM test", db))
# # Line of code above doesn't really work