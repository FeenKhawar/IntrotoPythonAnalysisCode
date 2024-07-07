import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# # It appears that to do this in Python (and not something like Jupyter Notebook)
# # You need to type plt.show() after writing something like plt.plot
# # It appears that with plt.show(), if there are multiple lines of code using it, it'll produce
# # them all in the sequence they are called, but the plots/graphs must be deleted/closed before viewing
# # the next plot/graph in the sequence 
# # When running plt.show(), the code used to make the plot/graph are blocked off, so you will need to rewrite the code to get the same plot/graph
# # It shouldn't matter if you use the same names for plt.figure() as any code is blocked off when doing plt.show()

# # 9.1: A Brief matplotlib API Primer

# data1 = np.arange(10)
# print(data1)
# plt.plot(data1)
# plt.show()

# # Figures and Subplots
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# plt.show()

# fig2 = plt.figure()
# ax1 = fig2.add_subplot(2, 2, 1)
# ax2 = fig2.add_subplot(2, 2, 2)
# ax3 = fig2.add_subplot(2, 2, 3)
# ax3.plot(np.random.standard_normal(50).cumsum(), color="black", linestyle="dashed")
# plt.show()

# fig3 = plt.figure()
# ax1 = fig3.add_subplot(2, 2, 1)
# ax2 = fig3.add_subplot(2, 2, 2)
# ax3 = fig3.add_subplot(2, 2, 3)
# ax1.hist(np.random.standard_normal(100), bins=20, color="black", alpha=0.3)
# ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30))
# ax3.plot(np.random.standard_normal(50).cumsum(), color="black", linestyle="dashed")
# plt.show()

# fig, axes = plt.subplots(2, 3)
# print(axes)
# plt.show()
# # This is how to create a gird of subplots faster, it "creates a new figure and returns a NumPy array containing the created subplot objects"

# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
# for i in range(2):
#     for j in range(2):
#         axes[i, j].hist(np.random.standard_normal(500), bins=50,
#                         color="black", alpha=0.5)
# fig.subplots_adjust(wspace=0, hspace=0)
# plt.show()

# # Colors, Markers, and Line Styles
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(np.random.standard_normal(30).cumsum(),color="black", linestyle="dashed", marker="o")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot()
# data2 = np.random.standard_normal(30).cumsum()
# ax.plot(data2, color="black", linestyle="dashed", label="Default")
# ax.plot(data2, color="black", linestyle="dashed", drawstyle="steps-post", label="steps-post")
# ax.legend()
# plt.show()
# # “You must call ax.legend to create the legend, whether or not you passed the label options when plotting the data.”

# # Ticks, Labels, and Legends
# fig, ax = plt.subplots()
# ax.plot(np.random.standard_normal(1000).cumsum())
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(np.random.standard_normal(1000).cumsum())
# ticks = ax.set_xticks([0, 250, 500, 750, 1000])
# labels = ax.set_xticklabels(["one", "two", "three", "four", "five"], rotation=30, fontsize=8)
# print(ax.set_xlabel("Stages")) # It seems I can say print to make it give me some output and still have the thing within the print work
# ax.set_title("My first matplotlib plot")
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(np.random.standard_normal(1000).cumsum())
# ticks = ax.set_xticks([0, 250, 500, 750, 1000])
# labels = ax.set_xticklabels(["one", "two", "three", "four", "five"], rotation=30, fontsize=8)
# ax.set(title="My first matplotlib plot", xlabel="Stages")
# # Line 88 does the same as both line 80 and 81 (excluding the print() part in line 80)
# plt.show()
# # To modify the y-axis, substitute the y for x

# fig, ax = plt.subplots()
# ax.plot(np.random.standard_normal(1000).cumsum())
# ticks = ax.set_xticks([0, 250, 500, 750, 1000])
# ticks = ax.set_yticks([-40, -30, -20, -10, 0, 10, 20, 30, 40])
# labels = ax.set_xticklabels(["one", "two", "three", "four", "five"], rotation=30, fontsize=8)
# ax.set(title="My first matplotlib plot", xlabel="Stages", ylabel="Range")
# plt.show()
# # Not sure how to change something like fontdixr without having to name all the labels

# # Adding legends
# fig, ax = plt.subplots()
# ticks = ax.set_xticks([0, 200, 400, 600, 800, 1000])
# ax.plot(np.random.randn(1000).cumsum(), color="black", label="one")
# ax.plot(np.random.randn(1000).cumsum(), color="black", linestyle="dashed", label="two")
# ax.plot(np.random.randn(1000).cumsum(), color="black", linestyle="dotted", label="three")
# ax.legend()
# plt.show()

# # Annotations and Drawing on a Subplot
# fig, ax = plt.subplots()
# data3 = pd.read_csv("/Users/faheemkhawar/Downloads/pydata-book-3rd-edition/examples/spx.csv", index_col=0, parse_dates=True)
# # The pf.read_csv is reading a file for a MacBook, not Windows, change as needed for code to work
# spx = data3["SPX"]
# spx.plot(ax=ax, color="black")
# crisis_data = [
#     (datetime(2007, 10, 11), "Peak of bull market"),
#     (datetime(2008, 3, 12), "Bear Stearns Fails"),
#     (datetime(2008, 9, 15), "Lehman Bankruptcy")
# ]
# for date, label in crisis_data:
#     ax.annotate(label, xy=(date, spx.asof(date) + 75),
#                 xytext=(date, spx.asof(date) + 225),
#                 arrowprops=dict(facecolor="black", headwidth=4, width=2,
#                                 headlength=4),
#                 horizontalalignment="left", verticalalignment="top")
# # Zoom in on 2007-2010
# ax.set_xlim(["1/1/2007", "1/1/2011"])
# ax.set_ylim([600, 1800])
# ax.set_title("Important dates in the 2008-2009 financial crisis")
# fig.savefig("FinancialCrisisPlot.svg") 
# fig.savefig("FinancialCrisisPlot.png", dpi=400) 
# fig.savefig("FinancialCrisisPlot.pdf", dpi=400) 
# plt.show()

# fig, ax = plt.subplots()
# rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3)
# circ = plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3)
# pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]],
#                    color="green", alpha=0.5)
# ax.add_patch(rect)
# ax.add_patch(circ)
# ax.add_patch(pgon)
# plt.show()
# fig.savefig("ShapesPlot")

# # Saving Plots to File
# # See book for description, already incorporated this in two examples above

# # matplotlib Configuration
# # See book for description and two one-line code examples

# # 9.2: Plotting with pandas and seaborn

# # Line Plots
# s = pd.Series(np.random.standard_normal(10).cumsum(), index=np.arange(10))
# s.plot()
# plt.show()

# s = pd.Series(np.random.standard_normal(10).cumsum(), index=np.arange(0, 100, 10))
# s.plot()
# plt.show()

# df1 = pd.DataFrame(np.random.standard_normal((10, 4)).cumsum(0), columns=["A", "B", "C", "D"], index=np.arange(0, 100, 10))
# plt.style.use('grayscale')
# df1.plot()
# plt.show()

# # Bar Plots
# fig, axes = plt.subplots(2, 1)
# # The plt.subplots(2, 1) means to give me 2 rows, 1 column of plots (giving 2 plots)
# # If it was instead plt.subplots(1, 2), it woulld give 1 row, 2 columns of plots (giving 2 plots)
# data4 = pd.Series(np.random.uniform(size=16), index=list("abcdefghijklmnop"))
# data4.plot.bar(ax=axes[0], color="black", alpha=0.7)
# data4.plot.barh(ax=axes[1], color="black", alpha=0.7)
# plt.show()

# df2 = pd.DataFrame(np.random.uniform(size=(6, 4)),
#                    index=["one", "two", "three", "four", "five", "six"],
#                    columns=pd.Index(["A", "B", "C", "D"], name="Genus"))
# print(df2)
# df2.plot.bar()
# plt.show()
# df2.plot.barh(stacked=True, alpha=0.5)
# plt.show()
# df2.value_counts().plot.bar()
# plt.show()
# # “A useful recipe for bar plots is to visualize a Series’s value frequency using value_counts: s.value_counts().plot.bar().”
# # Line 188 is just me messing around with the note described in line 190

# tips = pd.read_csv("/Users/faheemkhawar/Downloads/pydata-book-3rd-edition/examples/tips.csv")
# print(tips.head())
# party_counts = pd.crosstab(tips["day"], tips["size"])
# print(party_counts)
# party_counts = party_counts.reindex(index=["Thur", "Fri", "Sat", "Sun"])
# print(party_counts)
# party_counts = party_counts.loc[:, 2:5] # Notice how 5 is inclusive here
# print(party_counts)
# party_pcts = party_counts.div(party_counts.sum(axis="columns"), axis="index")
# print(party_pcts)
# party_pcts.plot.bar(stacked=True)
# plt.show()
# tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
# print(tips.head())
# sns.barplot(x="tip_pct", y="day", data=tips, orient="h")
# plt.show()
# sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")
# plt.show()
# sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")
# sns.set_style("whitegrid")
# plt.show()
# sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")
# sns.set_palette("Greys_r")
# plt.show()

# # Histogram and Density Plots
# tips = pd.read_csv("/Users/faheemkhawar/Downloads/pydata-book-3rd-edition/examples/tips.csv")
# tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
# tips["tip_pct"].plot.hist(bins=50)
# plt.show()
# tips["tip_pct"].plot.density()
# plt.show()

# comp1 = np.random.standard_normal(200)
# comp2 = 10+ 2 * np.random.standard_normal(200)
# values = pd.Series(np.concatenate([comp1, comp2]))
# # np.concatenate is adding the comp2 list to the comp1 list
# sns.histplot(values, bins=100, color="black")
# plt.show()

# # Scatter or Point Plots
# marco = pd.read_csv("/Users/faheemkhawar/Downloads/pydata-book-3rd-edition/examples/macrodata.csv")
# data5 = marco[["cpi", "m1", "tbilrate", "unemp"]]
# trans_data = np.log(data5).diff().dropna()
# print(trans_data.tail())
# ax = sns.regplot(x="m1", y="unemp", data=trans_data)
# ax.set_title("Changes in log(m1) versus log(unemp)")
# plt.show()
# ax = sns.regplot(x="m1", y="unemp", data=trans_data)
# ax.set(title="Changes in log(m1) versus log(unemp)")
# plt.show()
# # Looks like two identical ways to set the title
# sns.pairplot(trans_data, diag_kind="kde", plot_kws={"alpha": 0.2})
# # Not sure what the kde stands for, if you change it, you look the scatter plots where both axes are the same lists
# plt.show()

# # Facet Grids and Categorical Data
# tips = pd.read_csv("/Users/faheemkhawar/Downloads/pydata-book-3rd-edition/examples/tips.csv")
# tips["tip_pct"] = tips["tip"] / (tips["total_bill"] - tips["tip"])
# print(tips.head())
# sns.catplot(x="day", y="tip_pct", hue="time", col="smoker", kind="bar", data=tips[tips.tip_pct<1])
# plt.show()
# sns.catplot(x="day", y="tip_pct", hue="time", col="smoker", kind="bar", data=tips)
# plt.show()
# sns.catplot(x="day", y="tip_pct", row="time", col="smoker", kind="bar", data=tips[tips.tip_pct < 1])
# plt.show()
# sns.catplot(x="tip_pct", y="day", kind="box", data=tips[tips.tip_pct < 0.5])
# plt.show()
# print(tips["tip_pct"].max())
# x = (tips['tip'] > (tips['total_bill'] - tips['tip'])).sum()
# y = (tips['tip'] > (tips['total_bill'] - tips['tip'])).value_counts()
# print(x)
# print(y)
# z = (tips['tip_pct'] > 1).sum()
# print(z)
# print(tips["tip_pct"].count())