# Step 1: Prepare the data
#
# - Read the 60 monthly CAMS files (2020-2024, one value every 3 hours).
# - Turn them into daily averages (one value per day).
# - Change the units to the usual ones.
# - Save two files:
#     data/pm25_daily.nc   the daily data on the map grid (used for the maps)
#     data/pm25_daily.csv  a table with one row per (day, grid cell in India)
#
# Run: conda run -n pm25 python step1_prepare_data.py

import glob

import xarray as xr

# 1. Read all monthly files and put them together along time.
files = sorted(glob.glob("data/raw/eac4/*.nc"))
print("Number of files:", len(files))
data = xr.concat([xr.open_dataset(f) for f in files], dim="valid_time")

# 2. Short names, and latitude from south to north.
data = data.rename({"valid_time": "time", "latitude": "lat", "longitude": "lon"})
data = data[["pm2p5", "aod550", "u10", "v10", "t2m", "d2m", "sp"]]
data = data.sortby("lat")

# 3. Daily averages (8 values per day -> 1 value per day).
data = data.resample(time="1D").mean()
print("Number of days:", data.sizes["time"])

# 4. Units: PM2.5 kg/m3 -> µg/m3, temperature K -> °C, pressure Pa -> hPa.
data["pm2p5"] = data["pm2p5"] * 1e9
data["t2m"] = data["t2m"] - 273.15
data["d2m"] = data["d2m"] - 273.15
data["sp"] = data["sp"] / 100

# 5. Add which region each grid cell belongs to (made earlier from state borders).
regions = xr.open_dataset("data/india_regions.nc")
data["region"] = regions["region"]
data["in_india"] = regions["in_india"]

data.to_netcdf("data/pm25_daily.nc")

# 6. Make a table and keep only grid cells inside India.
table = data.to_dataframe().reset_index()
table = table[table["in_india"]].drop(columns="in_india")
table = table.rename(columns={"time": "date", "pm2p5": "pm25", "aod550": "aod"})

# Season of each day (India Meteorological Department seasons).
seasons = {1: "winter", 2: "winter",
           3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
           6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
           10: "post_monsoon", 11: "post_monsoon", 12: "post_monsoon"}
table["month"] = table["date"].dt.month
table["season"] = table["month"].map(seasons)

# 7. Quick checks.
print("Rows:", len(table))
print("Grid cells in India:", table.groupby(["lat", "lon"]).ngroups)
print("Missing values:", table.isna().sum().sum())
print(table.describe().round(2))

table.to_csv("data/pm25_daily.csv", index=False)
print("Saved data/pm25_daily.nc and data/pm25_daily.csv")
