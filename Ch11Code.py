import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Hour, Minute
from pandas.tseries.offsets import Day, MonthEnd
from pandas.tseries.frequencies import to_offset
from scipy.stats import percentileofscore
import pytz 

dates = [datetime(2011, 1, 2), datetime(2011, 1, 5),
         datetime(2011, 1, 7), datetime(2011, 1, 8),
         datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = pd.Series(np.random.standard_normal(6), index=dates)

close_px_all = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/stock_px.csv",
                            parse_dates=True, index_col=0)
close_px = close_px_all[["AAPL", "MSFT", "XOM"]]
close_px = close_px.resample("B").ffill()

# # 11.1: Date and TIme Data Types and Tools
# now = datetime.now()
# print(now)
# print(now.year, now.month, now.day)
# delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
# print(delta)
# print(delta.days, delta.seconds)
# start = datetime(2011, 1, 7)
# print(start + timedelta(12))
# print(start - 2 * timedelta(12))

# # Converting Between String and Datetime
# stamp = datetime(2011, 1, 3)
# print(stamp)
# print(str(stamp))
# print(stamp.strftime("%Y-%m-%d"))
# value = "2011-01-03"
# print(datetime.strptime(value, "%Y-%m-%d"))
# datestrs = ["7/6/2011", "8/6/2011"]
# print([datetime.strptime(x, "%m/%d/%Y") for x in datestrs])
# datestrs = ["2011-07-06 12:00:00", "2011-08-06 00:00:00"]
# print(pd.to_datetime(datestrs))
# idx = pd.to_datetime(datestrs + [None])
# print(idx)
# print(idx[2])
# print(pd.isna(idx))
# # "NaT" stands for "Not a Time"

# # 11.2: Time Series Basics
# print(dates)
# print(ts)
# print(ts.index)
# print(ts + ts[::2])
# print(ts.index.dtype)
# stamp = ts.index[0]
# print(stamp)
# # pandas.Timestamp like a datetime object but a little better (for example, goes to nanoseconds, not microseconds)
# # More on pandas.Timestamp later

# # Indexing, Selection, Subsetting
# stamp = ts.index[2]
# print(ts)
# print(ts[stamp])
# print(ts["2011-01-10"])
# longer_ts = pd.Series(np.random.standard_normal(1000), index=pd.date_range("2000-01-01", periods=1000))
# print(longer_ts)
# print(longer_ts.count())
# print(longer_ts["2001"])
# print(longer_ts["2001-05"])
# print(ts[datetime(2011, 1, 7):])
# print(ts[datetime(2011, 1, 7):datetime(2011, 1, 10)])
# print(ts)
# print(ts["2011-01-06":"2011-01-11"])
# # Slicing changes the original time series, no data is copied, modifications to the time series reflected in original data
# print(ts.truncate(after="2011-01-09"))
# dates = pd.date_range("2000-01-01", periods=100, freq="W-WED")
# long_df = pd.DataFrame(np.random.standard_normal((100, 4)), index=dates,
#                        columns=["Colorado", "Texas", "New York", "Ohio"])
# print(dates)
# print(long_df.loc["2001-05"])

# # Time Series with DUplicate Indices
# dates = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-02", "2000-01-02", "2000-01-03"])
# dup_ts = pd.Series(np.arange(5), index=dates)
# print(dup_ts)
# print(dup_ts.index.is_unique)
# print(dup_ts["2000-01-03"])
# print(dup_ts["2000-01-02"])
# grouped = dup_ts.groupby(level=0)
# print(grouped.mean())
# print(grouped.count())

# # 11.3: Date Ranges, Frequencies, and Shifting
# print(ts)
# resampler = ts.resample("D")
# print(resampler)

# # Generating Date Ranges
# index = pd.date_range("2012-04-01", "2012-06-01")
# print(index)
# print(pd.date_range(start="2012-04-01", periods=20))
# print(pd.date_range(end="2012-06-01", periods=20))
# print(pd.date_range("2000-01-01", "2000-12-01", freq="BM"))
# print(pd.date_range("2012-05-02 12:56:31", periods=5))
# print(pd.date_range("2012-05-02 12:56:31", periods=5, normalize=True))
# # "normalize=" normalizes a set of timestamps to midnight

# # Frequencies and Date Offsets
# hour = Hour()
# print(hour)
# four_hours = Hour(4)
# print(four_hours)
# print(pd.date_range("2000-01-01", "2000-01-03 23:59", freq="4H"))
# print(Hour(2) + Minute(30))
# print(pd.date_range("2000-01-01", periods=10, freq="1h30min"))
# # Week of month dates
# monthly_dates = pd.date_range("2012-01-01", "2012-09-01", freq="WOM-3FRI")
# print(monthly_dates)
# print(list(monthly_dates))

# # Shifting (Leading and Lagging) Data
# ts2 = pd.Series(pd.Series(np.random.standard_normal(4), index=pd.date_range("2000-01-01", periods=4, freq="M")))
# print(ts2)
# print(ts2.shift(2))
# print(ts2.shift(-2))
# # A common use of shift is computing consecutive percent changes in a time series or
# # multiple time series as DataFrame columns. This is expressed as:
# # ts2 / ts2.shift(1) - 1
# print(ts2.shift(2, freq="M"))
# print(ts2.shift(3, freq="D"))
# print(ts2.shift(1, freq="90T"))
# # "T" stands for minutes
# # Shifting dates with offsets
# now = datetime(2011, 11, 17)
# print(now + 3 * Day())
# print(now + MonthEnd())
# print(now + MonthEnd(2))
# offset = MonthEnd()
# print(offset.rollforward(now))
# print(offset.rollback(now))
# ts3 = pd.Series(np.random.standard_normal(20),
#                 index=pd.date_range("2000-01-15", periods=20, freq="4D"))
# print(ts3)
# print(ts3.groupby(MonthEnd().rollforward).mean())
# print(ts3.resample("M").mean())

# # 11.4: Time Zone Handling
# print(pytz.common_timezones[-5:])
# tz = pytz.timezone("America/New_York")
# print(tz)

# # Time Zone Localization and Conversion
# dates = pd.date_range("2012-03-09 09:30", periods=6)
# print(dates)
# ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
# print(ts)
# print(ts.index.tz)
# print(pd.date_range("2012-03-09 09:30", periods=10, tz="UTC"))
# ts_utc = ts.tz_localize("UTC")
# print(ts_utc)
# print(ts_utc.index)
# print(ts_utc.tz_convert("America/New_York"))
# ts_eastern = ts.tz_localize("America/New_York")
# print(ts_eastern.tz_convert("UTC"))
# print(ts_eastern.tz_convert("Europe/Berlin"))
# print(ts.index.tz_localize("Asia/Shanghai"))

# # Operations with Time Zone-Aware TImestamp Objects
# stamp = pd.Timestamp("2011-03-12 04:00")
# stamp_utc = stamp.tz_localize("utc")
# print(stamp_utc.tz_convert("America/New_York"))
# stamp_moscow = pd.Timestamp("2011-03-12 04:00", tz="Europe/Moscow")
# print(stamp_moscow)
# print(stamp_utc.value)
# print(stamp_utc.tz_convert("America/New_York").value)
# stamp = pd.Timestamp("2012-03-11 01:30", tz="US/Eastern")
# print(stamp)
# print(stamp + Hour())
# stamp = pd.Timestamp("2012-11-04 00:30", tz="US/Eastern")
# print(stamp)
# print(stamp + 2 * Hour())

# # Operations Between Different Time Zones
# # If two time series with different time zones are combined, the result will be UTC.
# # Since the timestamps are stored under the hood in UTC, this is a straightforward
# # operation and requires no conversion:
# dates = pd.date_range("2012-03-07 09:30", periods=10, freq="B")
# ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
# print(ts)
# ts1 = ts[:7].tz_localize("Europe/London")
# ts2 = ts1[2:].tz_convert("Europe/Moscow")
# result = ts1 + ts2
# print(result.index)

# # 11.5: Periods and Period Arithmetic
# p = pd.Period("2011", freq="A-DEC")
# print(p)
# print(p + 5)
# print(p - 2)
# print(pd.Period("2014", freq="A-DEC") - p)
# periods1 = pd.date_range("2000-01-01", "2000-06-30", freq="M")
# print(periods1)
# periods2 = pd.period_range("2000-01-01", "2000-06-30", freq="M")
# print(periods2)
# # .period_range seems to make the dates only year-month (due to freq="M"), 
# # while .date_range tries to include the last day of a month
# print(pd.Series(np.random.standard_normal(6), index=periods2))
# values = ["2001Q3", "2002Q2", "2003Q1"]
# index = pd.PeriodIndex(values, freq="Q-DEC")
# print(index)

# # Period Frequency Conversion
# p = pd.Period("2012Q4", freq="Q-JAN")
# print(p)
# print(p.asfreq("D", how="start"))
# print(p.asfreq("D", how="end"))
# p4pm = (p.asfreq("B", how="end") - 1).asfreq("T", how="start") + 16 * 6
# # p4pm = time‐stamp at 4 P.M. on the second-to-last business day of the quarter 
# print(p4pm)
# print(p4pm.to_timestamp())
# periods = pd.period_range("2011Q3", "2012Q4", freq="Q-JAN")
# ts = pd.Series(np.arange(len(periods)), index=periods)
# print(ts)
# new_periods = (periods.asfreq("B", "end") - 1).asfreq("H", "start") + 1
# ts.index = new_periods.to_timestamp()
# print(ts)

# # Converting Timestamps to Periods (and Back)
# dates = pd.date_range("2000-01-01", periods=3, freq="M")
# ts = pd.Series(np.random.standard_normal(3), index=dates)
# print(ts)
# pts = ts.to_period()
# print(pts)
# dates = pd.date_range("2000-01-29", periods=6)
# ts2 = pd.Series(np.random.standard_normal(6), index=dates)
# print(ts2)
# print(ts2.to_period("M"))
# pts = ts2.to_period()
# print(pts)
# print(pts.to_timestamp(how="start"))
# print(pts.to_timestamp(how="end"))

# # Creating a PeriodIndex from Arrays
# data = pd.read_csv("D:/IntrotoPyAnalysisDatasets&Examples/pydata-book-3rd-edition/examples/macrodata.csv")
# print(data.head(5))
# print(data["year"])
# print(data["quarter"])
# index = pd.PeriodIndex(year=data["year"], quarter=data["quarter"], freq="Q-DEC")
# print(index)
# data.index = index
# print(data["infl"])

# # 11.6: Resampling and Frequency Conversion
# dates = pd.date_range("2000-01-01", periods=100)
# ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
# print(ts)
# print(ts.resample("M").mean())
# print(ts.resample("M", kind="period").mean())

# # Downsampling
# dates = pd.date_range("2000-01-01", periods=12, freq="min") # Used "min" instead of "t" since it's better/updated version
# ts = pd.Series(np.arange(len(dates)), index=dates)
# print(ts)
# print(ts.resample("5min").sum()) # right bin edge not inclusive
# print(ts.resample("5min", closed="right").sum()) # right bin edge inclusive
# print(ts.resample("5min", closed="right", label="right").sum())
# result = ts.resample("5min", closed="right", label="right").sum()
# print(result)
# result.index = result.index + to_offset("-1s")
# print(result)
# # Open-high-low-close (OHLC) resampling
# ts = pd.Series(np.random.permutation(np.arange(len(dates))), index=dates)
# print(ts)
# print(ts.resample("5min").ohlc())

# # Upsampling and Interpolation
# frame = pd.DataFrame(np.random.standard_normal((2, 4)),
#                      index=pd.date_range("2000-01-01", periods=2,
#                                          freq="W-WED"),
#                                          columns=["Colorado", "Texas", "New York", "Ohio"])
# print(frame)
# df_daily = frame.resample("D").asfreq()
# print(df_daily)
# print(frame.resample("D").ffill())
# print(frame.resample("D").ffill(limit=2))
# print(frame.resample("W-THU").ffill())

# # Resampling with Periods
# frame = pd.DataFrame(np.random.standard_normal((24, 4)),
#                      index=pd.period_range("1-2000", "12-2001", freq="M"),
#                      columns=["Colorado", "Texas", "New York", "Ohio"])
# print(frame.head())
# annual_frame = frame.resample("A-DEC").mean()
# print(annual_frame)
# print(annual_frame.resample("Q-DEC").ffill())
# print(annual_frame.resample("Q-DEC", convention="end").asfreq())
# print(annual_frame.resample("Q-MAR").ffill())

# # Grouped Time Resampling
# N = 15
# times = pd.date_range("2017-05-20 00:00", freq="1min", periods=N)
# df = pd.DataFrame({"time": times, "value": np.arange(N)})
# print(df)
# print(df.set_index("time").resample("5min").count())
# df2 = pd.DataFrame({"time": times.repeat(3),
#                     "key": np.tile(["a", "b", "c"], N),
#                     "value": np.arange(N * 3.)})
# print(df2.head(7))
# time_key = pd.Grouper(freq="5min")
# resampled = (df2.set_index("time").groupby(["key", time_key]).sum())
# print(resampled)
# print(resampled.reset_index)
# # One constraint with using pandas.Grouper is that the time must be the index of the Series or DataFrame

# # 11.7: Moving Window Functions
# print(close_px["AAPL"].plot())
# close_px["AAPL"].rolling(250).mean().plot()
# plt.show()
# print(plt.figure())
# std250 = close_px["AAPL"].pct_change().rolling(250, min_periods=10).std()
# print(std250[5:12])
# std250.plot()
# plt.show()
# expanding_mean = std250.expanding().mean()
# # Line of code above is what writing a expanding window mean would look like
# plt.style.use('grayscale')
# close_px.rolling(60).mean().plot(logy=True)
# plt.show()
# print(close_px.rolling("20D").mean())

# # Exponentially Weighted Functions
# aapl_px = close_px["AAPL"]["2006":"2007"]
# ma30 = aapl_px.rolling(30, min_periods=20).mean()
# ewma30 = aapl_px.ewm(span=30).mean()
# aapl_px.plot(style="k-", label="Price")
# ma30.plot(style="k--", label="Simple Moving Avg")
# ewma30.plot(style="k-", label="EW MA")
# plt.legend()
# plt.show()

# # Binary Moving Window Functions
# spx_px = close_px_all["SPX"]
# spx_rets = spx_px.pct_change()
# returns = close_px.pct_change()
# corr =returns["AAPL"].rolling(125, min_periods=100).corr(spx_rets)
# corr.plot()
# plt.show()
# corr = returns.rolling(125, min_periods=100).corr(spx_rets)
# corr.plot()
# plt.show()

# # User-Defined Moving Window Functions
# def score_at_2percent(x):
#     return percentileofscore(x, 0.02)
# returns = close_px.pct_change()
# result = returns["AAPL"].rolling(250).apply(score_at_2percent)
# result.plot()
# plt.show()