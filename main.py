import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





df = pd.read_csv("kanto_earthquakes.csv")



df["time"] = pd.to_datetime(df["time"])

df["month"] = df["time"].dt.month


monthly_counts = df.groupby("month").size()

print("\n=== Earthquakes by Month ===\n")
print(monthly_counts)


top10 = df.nlargest(10, "mag")
print("\n=== Top10 earthquakes by year ===\n")
print(top10[["time", "place", "mag", "depth"]])


df["year"] = df["time"].dt.year

yearly_counts = df.groupby("year").size()
print("\n=== counts by year ===\n")
print(yearly_counts)


mag = df["mag"]
depth = df["depth"]


print("\n=== Earthquake Statistics ===\n")

print(f"Number of earthquakes : {len(df)}")

print(f"Average magnitude : {np.mean(mag):.2f}")
print(f"Median magnitude : {np.median(mag):.2f}")
print(f"Standard deviation : {np.std(mag):.2f}")

print(f"Maximum magnitude : {np.max(mag):.2f}")
print(f"Minimum magnitude : {np.min(mag):.2f}")

print(f"Average depth : {np.mean(depth):.2f} km")

largest = df.loc[df["mag"].idxmax()]

print("\n=== Largest Earthquake ===\n")

print(largest[["time", "place", "mag", "depth"]])


correlation = df["depth"].corr(df["mag"])

print("\n=== Correlation ===")
print(f"Depth vs Magnitude : {correlation:.3f}")



large_eq = df[df["mag"] >= 6.0]

print(large_eq[["time","place","mag"]])


yearly_mag = df.groupby("year")["mag"].mean()

print(yearly_mag)




df["energy"] = 10 ** (1.5 * df["mag"] + 4.8)

yearly_energy = df.groupby("year")["energy"].sum()






plt.hist(mag, bins=30)

plt.title("Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Number of Earthquakes")

plt.grid(True)

plt.savefig("magnitude_distribution.png")
plt.show()

plt.hist(depth, bins=40)

plt.title("Depth Distribution")
plt.xlabel("Depth (km)")
plt.ylabel("Number of Earthquakes")

plt.savefig("depth_distribution.png")
plt.show()





yearly_counts.plot(kind="bar")

plt.title("Earthquakes by Year")
plt.xlabel("Year")
plt.ylabel("Count")

plt.savefig("earthquakes_by_year.png")
plt.show()   

plt.figure(figsize=(8,5))

plt.scatter(depth, mag)

plt.title("Depth vs Magnitude")
plt.xlabel("Depth (km)")
plt.ylabel("Magnitude")

plt.grid(True)

plt.savefig("Depth_vs_Magnitude.png")
plt.show()




plt.figure(figsize=(10,8))

plt.scatter(
    df["longitude"],
    df["latitude"],
    alpha=0.5,
    s=5
)

plt.title("Earthquake Locations")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.grid(True)

plt.savefig("earthquake_map.png")

plt.show()


yearly_mag.plot()

plt.title("Average Magnitude by Year")
plt.xlabel("Year")
plt.ylabel("Average Magnitude")

plt.grid(True)

plt.savefig("Average_Magnitude_by_Year.png")
plt.show()


yearly_energy.plot()

plt.title("Total Earthquake Energy by Year")
plt.xlabel("Year")
plt.ylabel("Energy")

plt.savefig("Total_Earthquake_Energy_by_Year.png")
plt.show()  