import pygris
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# List of state FIPS codes
state_fips_codes = [
    '01', '02', '04', '05', '06', '08', '09', '10', '11', '12', '13', '15', '16', '17', '18', '19', '20', '21', '22', '23', 
    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', 
    '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56'
]

# Initialize an empty GeoDataFrame
all_census_tracts = gpd.GeoDataFrame()

# Loop through each state and pull the census tract data
for fips in state_fips_codes:
    state_tracts = pygris.tracts(state=fips, year=2020)
    all_census_tracts = pd.concat([all_census_tracts, state_tracts], ignore_index=True)

# Checking the first few rows of the combined data
print(all_census_tracts.head())

# Plotting the combined census tracts
all_census_tracts.plot()
plt.title('Census Tracts in the US')

# Save the plot as a PNG file
plt.savefig('/Users/mitchellhamilton/m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/outputs/census_tracts_us.png', dpi=300)

plt.show()
