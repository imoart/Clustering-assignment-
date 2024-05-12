import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
from scipy.optimize import curve_fit
import scipy.optimize as opt

# Read the data into a pandas DataFrame
data = pd.read_csv("climatedatacombinednew.csv")

# Filter the DataFrame for each indicator separately
population_total_df = data[data["Indicator Name"] == "Population, total"]
population_growth_df = data[data["Indicator Name"] == "Population growth (annual %)"]
mortality_rate_df = data[data["Indicator Name"] == "Mortality rate, under-5 (per 1,000 live births)"]
co2_emissions_df = data[data["Indicator Name"] == "CO2 emissions (kt)"]
electric_power_consumption_df = data[data["Indicator Name"] == "Electric power consumption (kWh per capita)"]
forest_area_df = data[data["Indicator Name"] == "Forest area (sq. km)"]
agricultural_land_df = data[data["Indicator Name"] == "Agricultural land (sq. km)"]

'''
# Save each DataFrame to a separate CSV file
population_total_df.to_csv("population_total.csv", index=False)
population_growth_df.to_csv("population_growth.csv", index=False)
mortality_rate_df.to_csv("mortality_rate.csv", index=False)
co2_emissions_df.to_csv("co2_emissions.csv", index=False)
electric_power_consumption_df.to_csv("electric_power_consumption.csv", index=False)
forest_area_df.to_csv("forest_area.csv", index=False)
agricultural_land_df.to_csv("agricultural_land.csv", index=False)'''

population_total_df = pd.read_csv('population_total.csv')
population_growth_df = pd.read_csv("population_growth.csv")
mortality_rate_df = pd.read_csv("mortality_rate.csv")
co2_emissions_df = pd.read_csv("co2_emissions.csv")
electric_power_consumption_df = pd.read_csv("electric_power_consumption.csv")
forest_area_df = pd.read_csv("forest_area.csv")
agricultural_land_df = pd.read_csv("agricultural_land.csv")

# List of all DataFrames
dfs = [population_total_df, population_growth_df, mortality_rate_df, co2_emissions_df, 
       electric_power_consumption_df, forest_area_df, agricultural_land_df]

# Iterate through each DataFrame
for i, df in enumerate(dfs):
    # Extracting columns containing data (excluding non-numeric columns)
    data_columns = df.columns[2:]
    
    # Calculating the mean of each row and storing it in a new column
    df['mean_value'] = df[data_columns].mean(axis=1)

    # Save the modified DataFrame back to the same variable
    dfs[i] = df
''''
population_growth_df = population_growth_df.drop(population_growth_df.columns[1], axis=1)
'''
# List of all DataFrames
dfs = [population_total_df, population_growth_df, mortality_rate_df, co2_emissions_df, 
       electric_power_consumption_df, forest_area_df, agricultural_land_df]

# Drop the column at index 1 for each DataFrame
for df in dfs:
    df.drop(df.columns[1], axis=1, inplace=True)

# Example: Printing the first DataFrame after dropping the column
print(population_total_df)

co2_emissions_df['Co2 per head'] = co2_emissions_df['mean_value']/ population_total_df['mean_value']

# Transpose each DataFrame and set the index to 'Country Name'
population_total_df_t = population_total_df.set_index('Country Name').T
population_growth_df_t = population_growth_df.set_index('Country Name').T
mortality_rate_df_t = mortality_rate_df.set_index('Country Name').T
co2_emissions_df_t = co2_emissions_df.set_index('Country Name').T
electric_power_consumption_df_t = electric_power_consumption_df.set_index('Country Name').T
forest_area_df_t = forest_area_df.set_index('Country Name').T
agricultural_land_df_t = agricultural_land_df.set_index('Country Name').T

df_combo = {}

# Add columns from each DataFrame to df_combo
df_combo['poptotal'] = population_total_df['mean_value']
df_combo['popgrowth'] = population_growth_df['mean_value']
df_combo['co2emissions'] = co2_emissions_df['mean_value']
df_combo['agriculture'] = agricultural_land_df['mean_value']
df_combo['mortality'] = mortality_rate_df['mean_value']
df_combo['electric'] = electric_power_consumption_df['mean_value']
df_combo['forest'] = forest_area_df['mean_value']

# Convert df_combo to a DataFrame
df_combo = pd.DataFrame(df_combo)

print(df_combo)

corr = df_combo.corr(numeric_only=True)
print(corr.round(4))
plt.figure()
plt.imshow(corr)
plt.colorbar()
# Set tick positions and labels
tick_positions = list(range(len(df_combo.columns)))
tick_labels = df_combo.columns
plt.xticks(tick_positions, tick_labels, rotation=30)
plt.yticks(tick_positions, tick_labels, rotation=0)
plt.show()


#pd.plotting.scatter_matrix(df_combo, figsize=(10, 10), s=10)
#plt.show()


'''
# For the first 15 years (1990-2004)
poptfirst15 = population_total_df.loc[:, '1990':'2004']
popgfirst15 = population_growth_df.loc[:, '1990':'2004']
mortfirst15 = mortality_rate_df.loc[:, '1990':'2004']
co2first15 = co2_emissions_df.loc[:, '1990':'2004']
electricfirst15 = electric_power_consumption_df.loc[:, '1990':'2004']
forestfirst15 = forest_area_df.loc[:, '1990':'2004']
agrifirst15 = agricultural_land_df.loc[:, '1990':'2004']
'''

# For the middle 15 years (2005-2019)
poptmid15 = population_total_df.loc[:, '1990':'2004']
popgmid15 = population_growth_df.loc[:, '1990':'2004']
mortmid15 = mortality_rate_df.loc[:, '1990':'2004']
co2mid15 = co2_emissions_df.loc[:, '1990':'2004']
electricmid15 = electric_power_consumption_df.loc[:, '1990':'2004']
forestmid15 = forest_area_df.loc[:, '1990':'2004']
agrimid15 = agricultural_land_df.loc[:, '1990':'2004']

poptrec15 = population_total_df.loc[:, '2005':'2020']
popgrec15 = population_growth_df.loc[:, '2005':'2020']
mortrec15 = mortality_rate_df.loc[:, '2005':'2020']
co2rec15 = co2_emissions_df.loc[:, '2005':'2020']
electricrec15 = electric_power_consumption_df.loc[:, '2005':'2020']
forestrec15 = forest_area_df.loc[:, '2005':'2020']
agrirec15 = agricultural_land_df.loc[:, '2005':'2020']

#%%
#print(poptfirst15)
#print(poptmid15)
#print(poptrec15)

'''
# For the first 15 years (1990-2004)
poptfirst15['mean_value'] = poptfirst15.mean(axis=1)
popgfirst15['mean_value'] = popgfirst15.mean(axis=1)
mortfirst15['mean_value'] = mortfirst15.mean(axis=1)
co2first15['mean_value'] = co2first15.mean(axis=1)
electricfirst15['mean_value'] = electricfirst15.mean(axis=1)
forestfirst15['mean_value'] = forestfirst15.mean(axis=1)
agrifirst15['mean_value'] = agrifirst15.mean(axis=1)
'''

# For the middle 15 years (2005-2019)
poptmid15['mean_value'] = poptmid15.mean(axis=1)
popgmid15['mean_value'] = popgmid15.mean(axis=1)
mortmid15['mean_value'] = mortmid15.mean(axis=1)
co2mid15['mean_value'] = co2mid15.mean(axis=1)
electricmid15['mean_value'] = electricmid15.mean(axis=1)
forestmid15['mean_value'] = forestmid15.mean(axis=1)
agrimid15['mean_value'] = agrimid15.mean(axis=1)

# For the recent 15 years (2008-2022)
poptrec15['mean_value'] = poptrec15.mean(axis=1)
popgrec15['mean_value'] = popgrec15.mean(axis=1)
mortrec15['mean_value'] = mortrec15.mean(axis=1)
co2rec15['mean_value'] = co2rec15.mean(axis=1)
electricrec15['mean_value'] = electricrec15.mean(axis=1)
forestrec15['mean_value'] = forestrec15.mean(axis=1)
agrirec15['mean_value'] = agrirec15.mean(axis=1)

df_combomid15 = {}
df_comborec15 = {}

# Add columns from each DataFrame to df_combo
df_combomid15['co2mid15_mean'] = co2mid15['mean_value']
df_combomid15['agrimid15_mean'] = agrimid15['mean_value']
df_combomid15['electricmid15_mean'] = electricmid15['mean_value']
df_combomid15['forestmid15_mean'] = forestmid15['mean_value']

df_combomid15 = pd.DataFrame(df_combomid15)


# Create DataFrame for recent 15 years (2008-2022)
df_comborec15 = pd.DataFrame()

# Add mean_value columns from the respective DataFrames
df_comborec15['co2rec15_mean'] = co2rec15['mean_value']
df_comborec15['agrirec15_mean'] = agrirec15['mean_value']
df_comborec15['electricrec15_mean'] = electricrec15['mean_value']
df_comborec15['forestrec15_mean'] = forestrec15['mean_value']

# Print or use df_comborec15 as needed
print(df_comborec15)

'''
df_co2first15 = co2first15['mean_value']
df_electricfirst15 = electricfirst15['mean_value']
df_forestfirst15 = forestfirst15['mean_value'] 
df_agrifirst15 = agrifirst15['mean_value']
'''

df_co2mid15 = co2mid15['mean_value']
df_electricmid15 = electricmid15['mean_value']
df_forestmid15 = forestmid15['mean_value'] 
df_agrimid15 = agrimid15['mean_value']

df_co2rec15 = co2rec15['mean_value']
df_electricrec15 = electricrec15['mean_value']
df_forestrec15 = forestrec15['mean_value'] 
df_agrirec15 = agrirec15['mean_value']

print(df_co2mid15)
print(df_co2rec15)

plt.figure(figsize=(10,6))
plt.scatter(df_co2mid15, df_forestmid15)
plt.scatter(df_co2rec15, df_forestrec15)
plt.xlabel('Mean')
plt.ylabel('Range')
plt.grid(True)
plt.show()


# Setup scaler objects
scaler1 = pp.RobustScaler()
scaler2 = pp.RobustScaler()
scaler3 = pp.RobustScaler()

# Extract columns
df_clust1 = df_combo[["co2emissions", "forest"]]
df_clust2 = df_combo[["co2emissions", "agriculture"]]
df_clust3 = df_combo[["co2emissions", "electric"]]

# Fit and transform each DataFrame separately
df_norm1 = scaler1.fit_transform(df_clust1)
df_norm2 = scaler2.fit_transform(df_clust2)
df_norm3 = scaler3.fit_transform(df_clust3)

# Print the results
print(df_norm1)
print(df_norm2)
print(df_norm3)
'''
plt.figure(figsize=(8, 8))
plt.scatter(df_norm1[:,0], df_norm1[:, 1], 10, marker="o")
plt.xlabel("CO2 emissions")
plt.ylabel("Forest Area")
plt.show()
'''
'''
# Create scatter plot for CO2 emissions and agriculture
plt.figure(figsize=(8, 6))
plt.scatter(df_norm2[:, 0], df_norm2[:, 1], 10, marker="o")
plt.xlabel("CO2 emissions")
plt.ylabel("Agriculture")
plt.title("Scatter Plot: CO2 emissions vs Agriculture")
plt.grid?(True)
plt.show()


# Create scatter plot for CO2 emissions and electric
plt.figure(figsize=(8, 6))
plt.scatter(df_norm3[:, 0], df_norm3[:, 1], 10, marker="o", color='green')
plt.xlabel("CO2 emissions")
plt.ylabel("Electric")
plt.title("Scatter Plot: CO2 emissions vs Electric")
plt.grid(True)
plt.show()
'''

def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy) # fit done on x,y pairs
    labels = kmeans.labels_
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))
    return score

# Extract columns from all DataFrames
df_clust4 = df_combomid15[["co2mid15_mean", "forestmid15_mean"]]
df_clust5 = df_combomid15[["co2mid15_mean", "agrimid15_mean"]]
df_clust6 = df_combomid15[["co2mid15_mean", "electricmid15_mean"]]

# Set up a scaler for each DataFrame
scaler4 = pp.RobustScaler()
scaler5 = pp.RobustScaler()
scaler6 = pp.RobustScaler()

# Fit and transform each DataFrame separately
df_norm4 = scaler4.fit_transform(df_clust4)
df_norm5 = scaler5.fit_transform(df_clust5)
df_norm6 = scaler6.fit_transform(df_clust6) 

# Print the results
print(df_norm4)
print(df_norm5)
print(df_norm6)


# Extract columns from all DataFrames
df_clust7 = df_comborec15[["co2rec15_mean", "forestrec15_mean"]]
df_clust8 = df_comborec15[["co2rec15_mean", "agrirec15_mean"]]
df_clust9 = df_comborec15[["co2rec15_mean", "electricrec15_mean"]]

# Set up a scaler for each DataFrame
scaler7 = pp.RobustScaler()
scaler8 = pp.RobustScaler()
scaler9 = pp.RobustScaler()

# Fit and transform each DataFrame separately
df_norm7 = scaler7.fit_transform(df_clust7)
df_norm8 = scaler8.fit_transform(df_clust8)
df_norm9 = scaler9.fit_transform(df_clust9) 

# Print the results
print(df_norm7)
print(df_norm8)
print(df_norm9)

#calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(df_norm1, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}") 

for ic in range(2, 11):
    score = one_silhoutte(df_norm2, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
    
for ic in range(2, 11):
    score = one_silhoutte(df_norm3, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

for ic in range(2, 11):
    score = one_silhoutte(df_norm4, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

for ic in range(2, 11):
    score = one_silhoutte(df_norm5, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

for ic in range(2, 11):
    score = one_silhoutte(df_norm6, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

for ic in range(2, 11):
    score = one_silhoutte(df_norm7, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

for ic in range(2, 11):
    score = one_silhoutte(df_norm8, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

for ic in range(2, 11):
    score = one_silhoutte(df_norm9, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

# set up the clusterer with the number of expected clusters
kmeans1 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans2 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans3 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans4 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans5 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans6 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans7 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans8 = cluster.KMeans(n_clusters=2, n_init=20)
kmeans9 = cluster.KMeans(n_clusters=2, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans1.fit(df_norm1) # fit done on x,y pairs
kmeans2.fit(df_norm2)
kmeans3.fit(df_norm3)
kmeans4.fit(df_norm4)
kmeans5.fit(df_norm5)
kmeans6.fit(df_norm6)
kmeans7.fit(df_norm7)
kmeans8.fit(df_norm8)
kmeans9.fit(df_norm9)
# extract cluster labels
labels1 = kmeans1.labels_
labels2 = kmeans2.labels_
labels3 = kmeans3.labels_
labels4 = kmeans4.labels_
labels5 = kmeans5.labels_
labels6 = kmeans6.labels_
labels7 = kmeans7.labels_
labels8 = kmeans8.labels_
labels9 = kmeans9.labels_
# extract the estimated cluster centres and convert to original scales
cen1 = kmeans1.cluster_centers_
cen2 = kmeans2.cluster_centers_
cen3 = kmeans3.cluster_centers_
cen4 = kmeans4.cluster_centers_
cen5 = kmeans5.cluster_centers_
cen6 = kmeans6.cluster_centers_
cen7 = kmeans7.cluster_centers_
cen8 = kmeans8.cluster_centers_
cen9 = kmeans9.cluster_centers_

cen1 = scaler1.inverse_transform(cen1)
cen2 = scaler2.inverse_transform(cen2)
cen3 = scaler3.inverse_transform(cen3)
cen4 = scaler4.inverse_transform(cen4)
cen5 = scaler5.inverse_transform(cen5)
cen6 = scaler6.inverse_transform(cen6)
cen7 = scaler7.inverse_transform(cen7)
cen8 = scaler8.inverse_transform(cen8)
cen9 = scaler9.inverse_transform(cen9)

xkmeans1 = cen1[:, 0]
ykmeans1 = cen1[:, 1]
xkmeans2 = cen2[:, 0]
ykmeans2 = cen2[:, 1]
xkmeans3 = cen3[:, 0]
ykmeans3 = cen3[:, 1]
xkmeans4 = cen4[:, 0]
ykmeans4 = cen4[:, 1]
xkmeans5 = cen5[:, 0]
ykmeans5 = cen5[:, 1]
xkmeans6 = cen6[:, 0]
ykmeans6 = cen6[:, 1]
xkmeans7 = cen7[:, 0]
ykmeans7 = cen7[:, 1]
xkmeans8 = cen8[:, 0]
ykmeans8 = cen8[:, 1]
xkmeans9 = cen9[:, 0]
ykmeans9 = cen9[:, 1]

# extract x and y values of data points
x1 = df_clust1["co2emissions"]
y1 = df_clust1["forest"]
x2 = df_clust2["co2emissions"]
y2 = df_clust2["agriculture"]
x3 = df_clust3["co2emissions"]
y3 = df_clust3["electric"]
x4 = df_clust4["co2mid15_mean"]
y4 = df_clust4["forestmid15_mean"]
x5 = df_clust5["co2mid15_mean"]
y5 = df_clust5["agrimid15_mean"]
x6 = df_clust6["co2mid15_mean"]
y6 = df_clust6["electricmid15_mean"]
x7 = df_clust7["co2rec15_mean"]
y7 = df_clust7["forestrec15_mean"]
x8 = df_clust8["co2rec15_mean"]
y8 = df_clust8["agrirec15_mean"]
x9 = df_clust9["co2rec15_mean"]
y9 = df_clust9["electricrec15_mean"]

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x1, y1, 10, labels1, marker="o")
# show cluster centres
plt.scatter(xkmeans1, ykmeans1, 45, "k", marker="d")
plt.xlabel("Co2emissions")
plt.ylabel("Forest Area")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x2, y2, 10, labels2, marker="o", cmap='winter')
# show cluster centres
plt.scatter(xkmeans2, ykmeans2, 45, "k", marker="d")
plt.xlabel("Co2emissions")
plt.ylabel("Agriculture")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x3, y3, 10, labels3, marker="o", cmap='viridis')
# show cluster centres
plt.scatter(xkmeans3, ykmeans3, 45, "k", marker="d")
plt.xlabel("Co2emissions")
plt.ylabel("Electricity")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x4, y4, 10, labels4, marker="o", cmap='plasma')
# show cluster centres
plt.scatter(xkmeans5, ykmeans5, 45, "k", marker="d")
plt.xlabel("Co2emissions First 15 Years")
plt.ylabel("Forest Area First 15 Years")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x5, y5, 10, labels5, marker="o", cmap='inferno')
# show cluster centres
plt.scatter(xkmeans5, ykmeans5, 45, "k", marker="d")
plt.xlabel("Co2emissions First 15 Years")
plt.ylabel("Agriculture First 15 Years")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x6, y6, 10, labels6, marker="o", cmap='magma')
# show cluster centres
plt.scatter(xkmeans6, ykmeans6, 45, "k", marker="d")
plt.xlabel("Co2emissions First 15 Years")
plt.ylabel("Electricity First 15 Years")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x7, y7, 10, labels7, marker="o", cmap='cool')
# show cluster centres
plt.scatter(xkmeans7, ykmeans7, 45, "k", marker="d")
plt.xlabel("Co2emissions Last 15 Years")
plt.ylabel("Forest Area Last 15 Years")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x8, y8, 10, labels8, marker="o", cmap='hot')
# show cluster centres
plt.scatter(xkmeans8, ykmeans8, 45, "k", marker="d")
plt.xlabel("Co2emissions Last 15 Years")
plt.ylabel("Agriculture Last 15 Years")
plt.grid(True)
plt.show() 

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x9, y9, 10, labels9, marker="o", cmap='spring')
# show cluster centres
plt.scatter(xkmeans9, ykmeans9, 45, "k", marker="d")
plt.xlabel("Co2emissions Last 15 Years")
plt.ylabel("Electricity Last 15 Years")
plt.grid(True)
plt.show() 

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f

bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
param, covar = opt.curve_fit(logistic, co2mid15["mean_value"], forestmid15["mean_value"], bounds=bounds)

print(param)
print(covar)

forestmid15["trial"] = logistic(forestmid15["Year"], 3e12, 0.10, 1990)
forestmid15.plot("mean_value", "trial")
plt.show()

