layout: post
title: "Mapping the MLB with Python"
date: 2024-06-30
categories: projects

## Introduction

Welcome to Explorations in Geospatial Analysis (aka Mitchy Maps). I've been exploring the world of GIS and geospatial analytics in my work recently. It's been slow going so far, but every time I make progress I'm excited and more energized to learn about this space. I created this blog as a way to learn and share my progress.

The world of data science is generally opaque for people who aren't trained data scientists. Geospatial data science is even more opaque. This will broadly be an attempt to help data-curious people like me understand how to use common tools, techniques, and libraries to perform geospatial analysis for business & pleasure. Thanks for following along!

## Project Overview

People who know me well know that I love baseball (go Braves!). As a sort of ceremonial first pitch for this blog, I decided to map MLB stadiums and explore the areas around them. I wanted to see at a basic level whether population density and other demographic factors might be related to team attendance. Short answer: they're not really related at all. We'll get deeper into the answer throughout the post.

I used python exclusively for this analysis. There are many great geospatial and geospatial-adjacent libraries available for python, including geopandas, geopy, shapely, folium, pygris, cenpy, and many more. I'll go into more detail about the libraries I used in the detailed summary of my analysis below. 

I got inspiration from https://walker-data.com/posts for some of this analysis.

## Analysis
Distance and proximity analyses are common in spatial data science. It's often important to understand how something is to something else. Proximity analysis can give a business clues about where to place a new location or help a government understand a population's access to critical resources like hospitals. For this project, I started by exploring how close/far people are from MLB stadiums.

### Mapping MLB stadiums
First, I copied MLB stadium names & addresses from [`MLB`](#https://www.mlb.com/team) and stadium capacity from [`Wikipedia`](#https://en.wikipedia.org/wiki/List_of_current_Major_League_Baseball_stadiums) into a csv file. I thought about scraping this data, but it was easier/quicker to collect the data manually.

Then, I geocoded the MLB ballparks using the Photon geocoder within geopy. Geocoding is the process of converting addresses, place names, or other location-based data into geographic coordinates (latitude and longitude). Geocoding is an important part of geospatial analytics for many reasons but particularly for mapping and integrating with other geographic data.

```python
import pandas as pd
from geopy.geocoders import Photon
import time

# Load the CSV file with pandas
csv_file_path = '~/mlb_stadiums.csv'
df = pd.read_csv(csv_file_path)

# Initialize the Photon geocoder
geolocator = Photon(user_agent="geoapiExercises") # User agent name is important to avoid issues

# Function to geocode an address
def geocode_address(address, retries=3):
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        if retries > 0:
            time.sleep(1)
            return geocode_address(address, retries - 1)
        else:
            print(f"Error geocoding {address}: {e}")
            return None, None

# Geocode the addresses
latitudes = []
longitudes = []

for index, row in df.iterrows():
    full_address = f"{row['Address']}, {row['City']}, {row['State']} {row['ZIP']}"
    lat, lon = geocode_address(full_address)
    latitudes.append(lat)
    longitudes.append(lon)
    time.sleep(1)  # To prevent hitting the API rate limit

# Add the latitude and longitude to the DataFrame
df['Latitude'] = latitudes
df['Longitude'] = longitudes

# Save the updated DataFrame to a new CSV file
output_csv_file_path = '~/mlb_stadiums_geocoded.csv'
df.to_csv(output_csv_file_path, index=False)

print(f"Geocoded addresses saved to {output_csv_file_path}")
```

Usually, I use Nominatim for quick geocoding in python but I kept getting a 403 error code regardless of what I used for my parameters, so I switched to Photon for this analysis. There are many ways to geocode addresses, most of them free in small batches. My favorites are Nominatim and the [`Census Geocoder`](#https://www.census.gov/programs-surveys/geography/technical-documentation/complete-technical-documentation/census-geocoder.html) batch processing tool. The Google Maps and Mapbox APIs are quite robust but also can be expensive. 

It's also important to note that the input data quality affects geocoding success and accuracy quite dramatically. The Google Maps API can handle geocoding from a place name input like 'Truist Park', but many geocoders require an address. You can achieve accurate results quite often with quality address inputs with Street, City, State, and ZIP included.

Once the stadium addresses are geocoded, we can add move on to mapping. This involves importing a few libraries, including folium for the map and os & base64 to be able to use SVG images of team logos as the markers on the map.

```python
import folium
import base64
import os

# Initialize the Folium map
folium_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)  # Centered on the US

# Function to add custom markers with logos
def add_custom_marker(row):
    logo_path = row['Logo']
    if not os.path.isfile(logo_path):
        print(f"File not found: {logo_path}")
        return

    try:
        with open(logo_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()
        html = f"""
        <div style="background-color: transparent; width: 20px; height: 20px;">
            <img src="data:image/svg+xml;base64,{encoded}" style="width: 20px; height: 20px;"/>
        </div>
        """
        icon = folium.DivIcon(html=html)
        popup_html = f"""
        <div style="white-space: nowrap;">
            <strong>{row['Team']}</strong><br>
            {row['Stadium']}<br>
            Capacity: {row['Capacity']}
        </div>
        """
        popup = folium.Popup(popup_html, max_width=300)
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup,
            icon=icon
        ).add_to(folium_map)
    except Exception as e:
        print(f"Error loading {logo_path}: {e}")

# Add markers to the map
df.apply(add_custom_marker, axis=1)

# Show the map
folium_map

```

We ended up with a simple map of MLB stadiums (the overlap of some teams is not ideal, but it’s cooler to have the team logos than basic dots in my opinion!).

<iframe src="/projects/mlb-analysis/outputs/mlb_ballparks_map.html" width="100%" height="600px"></iframe>

### Proximity analysis
MLB stadiums are located in diverse areas of their cities, so I wanted to understand the areas that are near MLB stadiums. To do this, we had to have some locations to compare against the MLB stadiums. I used Census tracts as my geographic areas because they’re small and have a lot of available Census data attached to them. Census tracts are small geographic areas with 1,200-8,000 people. They're great for geospatial & demographic analysis because they're much more comparable to each other in terms of population size than other geographic areas like ZIP codes and counties.

First, we pulled the census tracts in via pygris and used geopandas to load the MLB stadium data we gathered earlier.

```python
import geopandas as gpd
import pandas as pd
from pygris import tracts
from shapely.geometry import Point

# List of state FIPS codes excluding Alaska (02) and Hawaii (15)
state_fips_codes = [
    '01', '04', '05', '06', '08', '09', '10', '11', '12', '13', '16', '17', '18', '19', '20', '21', '22', '23', 
    '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '44', 
    '45', '46', '47', '48', '49', '50', '51', '53', '54', '55', '56'
]

# Initialize an empty GeoDataFrame
all_census_tracts = gpd.GeoDataFrame()

# Loop through each state and pull the census tract data
for fips in state_fips_codes:
    state_tracts = tracts(state=fips, year=2020)
    all_census_tracts = pd.concat([all_census_tracts, state_tracts], ignore_index=True)

# Set the CRS to NAD83 (EPSG:4269), a common CRS for national datasets in the US
all_census_tracts.set_crs(epsg=4269, inplace=True)

# Load the baseball stadiums data
stadium_csv_path = '~/mlb_stadiums_geocoded.csv'
stadiums = pd.read_csv(stadium_csv_path)

# Convert stadium DataFrame to a GeoDataFrame
stadiums['geometry'] = stadiums.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
stadiums_gdf = gpd.GeoDataFrame(stadiums, geometry='geometry')

# Set the CRS for the stadiums to match the census tracts CRS
stadiums_gdf.set_crs(epsg=4269, inplace=True)

print("Census tracts and stadiums data loaded with CRS set.")
```

Then, we wanted to find out which MLB stadiums are _near_ each census tract. We don't want to search by state here because some Census tracts, and therefore people, are closest to an MLB stadium outside their state. For example, some Maryland residents are closer to Nationals Park in DC than they are to Camden Yards in Baltimore. To do this, we identified all MLB stadiums within 100 km of the census tracts. We use a sequence of common GIS operations with geopandas to accomplish this. Our steps include:

- Combining the Census tracts into a single shape with the .dissolve() method;
- Drawing a new shape that extends to 100km beyond the border of all tracts with the .buffer() method;
- Using an inner spatial join to retain only the stadiums that fall within the 100km buffer shape (all of them).

```python
# Create a buffer around the dissolved census tracts
buffer_distance = 100000  # Distance in meters

# Dissolve all tracts into one geometry and create a buffer
national_buffer = gpd.GeoDataFrame(geometry=all_census_tracts.dissolve().buffer(buffer_distance))

# Spatial join between the stadiums and the buffer to find stadiums within the buffer
stadiums_within_buffer = gpd.sjoin(stadiums_gdf, national_buffer, how="inner", op="within")

print("Buffer created and spatial join completed.")
```

After running this operation, we can draw a quick plot to show the relationships between Census tracts and MLB stadiums.

```python
import matplotlib.pyplot as plt

# Plot the results
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all census tracts in grey
all_census_tracts.plot(ax=ax, color="grey", linewidth=0.1)

# Plot the stadiums within the buffer in red
stadiums_within_buffer.plot(ax=ax, color="red", markersize=5)

ax.set_title('MLB Stadiums within Buffer of Dissolved US Census Tracts')
ax.set_axis_off()
plt.show()
```
The output looks like this, confirming that we have captured all MLB stadiums for future analysis.

![Map of the Census tracts with MLB stadiums](projects/mlb-analysis/outputs/mlb_stadiums_within_buffer.png)

Now, we have to caclulate the distance from each Census tract to the nearest MLB stadium to get a proximity distribution. Census tracts are geographic areas (polygons), not points, so we have to first identify the "centroid" of each Census tract. We commonly use centroids to calculate distances and understand relationships between complex shapes and points in geospatial analysis. By using centroids, you can easily measure the distance between the center of a polygon and other points (e.g., comparing the distance between the center of a city boundary and specific locations like stores or schools). 

We had to re-project the coordinates to a different Coordinate Reference System (CRS) to enable easy comparison.

```python
# Reproject to a projected CRS for accurate distance calculations
projected_crs = 2163  # US National Atlas Equal Area
all_census_tracts = all_census_tracts.to_crs(epsg=projected_crs)
stadiums_gdf = stadiums_gdf.to_crs(epsg=projected_crs)
```
Then, we calculated the centroid of each Census tract, calculated the distance from each centroid to the stadiums to identify the nearest one, added the distance back to the census gdf, and, finally, mapped it!

```python
# Calculate centroids of census tracts
tract_centroids = all_census_tracts.centroid

# Calculate distances from each tract centroid to the nearest stadium
distances = tract_centroids.apply(lambda g: stadiums_gdf.distance(g).min())

# Add the distance information back to the original census tracts GeoDataFrame
all_census_tracts['distance_to_stadium_miles'] = distances * 0.000621371  # Convert from meters to miles

# Plot the results
fig, ax = plt.subplots(figsize=(10, 8))

# Plot all census tracts color-coded by distance to the nearest stadium
all_census_tracts.plot(column='distance_to_stadium_miles', cmap='magma', legend=True,
                       legend_kwds={'label': "Distance to nearest stadium (miles)",
                                    'orientation': "horizontal"}, ax=ax)

ax.set_title('Travel Distance (miles) to Nearest MLB Stadium')
ax.set_axis_off()
plt.show()
```
![Map of travel distance to MLB stadiums](projects/mlb-analysis/outputs/travel_distance_to_stadiums.png)

This is super interesting, but not unexpected. There are many areas in the US where people are quite far from an MLB stadium, especially across the great plains and parts of the western US (though this will change once Oakland moves to Sacramento and then Las Vegas). The average distance from a Census tract to an MLB stadium is 128 miles, but a simple histogram shows that most Census tracts are within 100 miles of a stadium.

![Histogram of travel time distribution](projects/mlb-analysis/outputs/travel_time_distribution.png)

We know that in rural areas, however, straight-line distances can be misleading. Given the geography of highway networks, accessibility to a trauma center is mediated through accessibility to that road network." Let's look at drive time in addition to distance to get a better sense of which Census tracts are "close" to MLB stadiums.

### Drive time analysis: Truist Park
We can connect to Mapbox’s navigation services with the routingpy package, an interface to several hosted navigation APIs. This allows us to calculate drive time for our dataset. The Mapbox API has a meaningful free tier, but there are thousands of Census tracts and 30 MLB stadiums, so we would have spent $1,000+ on this analysis had we calculated drive time for every stadium. Therefore, I focused on Truist Park in Atlanta for the remaining analysis.

We start with importing the necessary libraries and connecting to the Mapbox API via routingpy.

```python
import geopandas as gpd
import pandas as pd
from pygris import tracts
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from routingpy.routers import MapboxOSRM

# Set your Mapbox access token
mapbox_key = 'YOUR_KEY_HERE'
mb = MapboxOSRM(api_key=mapbox_key)

# API usage limits
max_requests_per_minute = 60  # For driving, walking, and cycling profiles
max_requests_per_day = 100000  # Standard daily limit for free tier
```
Then, we followed the same steps as above to gather Census tracts, dissolve them into a single shape, and merge them with the stadium data. This time, however, we only collected Census data for GA (FIPS code '13'). After we have the data loaded and prepared, we can generate a list of coordinate pairs to feed to Mapbox. We ultimately calculated 2,300 'elements', the output of the [Mapbox Matrix API](#https://docs.mapbox.com/api/navigation/matrix/). This was almost 25% of our monthly free limit, so it's a good thing we focused on 1 state instead of all 50.

```python
# Function to convert points to coordinates
def points_to_coords(input_gdf):
    geom = input_gdf.to_crs(epsg=4326).geometry
    return [[g.x, g.y] for g in geom]

# Generate list of coordinates
tract_coords = points_to_coords(gpd.GeoDataFrame(geometry=tract_centroids))
stadium_coords = points_to_coords(stadiums_within_buffer)

# Split the tract coordinates into chunks to meet the API limits
split_size = 10  # Ensure we stay within the 25 coordinate limit per request
chunks = [tract_coords[x:x + split_size] for x in range(0, len(tract_coords), split_size)]

times_list = []
total_requests = 0

for chunk in chunks:
    all_coords = chunk + stadium_coords

    # Find the indices of origin and destination
    origin_ix = list(range(len(chunk)))
    stadium_ix = list(range(len(chunk), len(all_coords)))

    # Check API limits
    if total_requests >= max_requests_per_day:
        print("Reached the daily API limit. Stopping requests.")
        break

    # Implement rate limiting
    if total_requests % max_requests_per_minute == 0 and total_requests != 0:
        print("Reached the per-minute API limit. Pausing for 60 seconds.")
        time.sleep(60)

    # Run the travel-time matrix
    matrix_result = mb.matrix(
        locations=all_coords,
        profile='driving',
        sources=origin_ix,
        destinations=stadium_ix
    )

    # Increment request count
    total_requests += 1

    # Extract durations from the result
    times = matrix_result.durations

    # Convert the result to a DataFrame
    times_df = pd.DataFrame(times)

    times_list.append(times_df)

all_times = pd.concat(times_list, ignore_index=True)
```
Now we have the data we need on drive times to do some meaningful analysis and visualization. We start by calculating the minimum travel time in minutes to the nearest stadium (which will be Truist Park in all cases). Then, we plot the results! 

```python
# Calculate minimum travel time (in minutes) to the nearest stadium
min_times = all_times.min(axis="columns") / 60

# Add the minimum travel time to the census tracts GeoDataFrame
ga_tracts_within_buffer['time'] = min_times

# Plot the results
fig, ax = plt.subplots(figsize=(8, 10))

ga_tracts_within_buffer.plot(column="time", legend=True, cmap="magma", 
                             legend_kwds={"location": "top", "shrink": 0.5}, ax=ax, aspect=1)

# Adjust the title with padding
plt.title("Travel time (minutes) to nearest MLB Stadium", pad=75, fontsize=14)

# Remove axis
ax.set_axis_off()

# Annotate the plot with more padding and better positioning
ax.annotate('Census tracts in Georgia and surrounding areas\nData sources: US Census Bureau, MLB, Mapbox', 
            xy=(0.25, 0.05), xycoords='figure fraction',
            fontsize=10, ha='center', va='center')

# Adjust the aspect ratio to ensure the plot is not skewed
ax.set_aspect('equal')

# Set axis limits to ensure the state fits well within the plot area
minx, miny, maxx, maxy = ga_tracts_within_buffer.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

plt.show()
```
The map is not surprising at all. Areas far away from Truist Park are far in terms of distance and drive time. Truist Park is located in Metro Atlanta, so the Atlanta area has the easiest access to the stadium. There are no stadiums in north Florida or in TN/SC, so people in N/S Georgia do not have access to another nearby stadium.

![Map of GA Census tract travel times to Truist Park](projects/mlb-analysis/outputs/travel_time_to_truist.png)

### Travel time isochrones
Isochrones are a really interesting way to show layers of proximity and travel time from a given point. An isochrone is a line or area on a map that represents all points that can be reached within a certain time or distance from a given location. Isochrones are commonly used in transportation planning, logistics, and urban planning to visualize accessibility and travel times.

We can use isochrones to explore the immediate areas around Truist Park in more detail. We'll use the Mapbox API via routingpy again to generate an isochrone. We'll reload the stadiums data into a new dataframe, then call the Mapbox API to generate the isochrones, and map the results.

```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from routingpy.routers import MapboxOSRM
import numpy as np

# Load the baseball stadiums data
stadium_csv_path = '/~/mlb_stadiums_geocoded.csv'
stadiums = pd.read_csv(stadium_csv_path)

# Filter for Truist Park
truist_park = stadiums[stadiums['Stadium'] == 'Truist Park']

# Convert to GeoDataFrame
truist_park = truist_park.copy()  # Make a copy to avoid SettingWithCopyWarning
truist_park['geometry'] = truist_park.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
truist_park_gdf = gpd.GeoDataFrame(truist_park, geometry='geometry', crs="EPSG:4326")

def mb_isochrone(gdf, time=[5, 10, 15], profile="driving"):
    # Grab X and Y values in 4326
    gdf['LON_VALUE'] = gdf.to_crs(4326).geometry.x
    gdf['LAT_VALUE'] = gdf.to_crs(4326).geometry.y

    coordinates = gdf[['LON_VALUE', 'LAT_VALUE']].values.tolist()

    # Build a list of shapes
    isochrone_shapes = []

    if type(time) is not list:
        time = [time]

    # Use minutes as input, but the API requires seconds
    time_seconds = [60 * x for x in time]

    # Given the way that routingpy works, we need to iterate through the list of 
    # coordinate pairs, then iterate through the object returned and extract the 
    # isochrone geometries.  
    for c in coordinates:
        iso_request = mb.isochrones(locations=c, profile=profile,
                                    intervals=time_seconds, polygons="true")

        for i in iso_request:
            iso_geom = Polygon(i.geometry[0])
            isochrone_shapes.append(iso_geom)

    # Here, we re-build the dataset but with isochrone geometries
    df_values = gdf.drop(columns=['geometry', 'LON_VALUE', 'LAT_VALUE'])

    time_col = time * len(df_values)

    # We'll need to repeat the dataframe to account for multiple time intervals
    df_values_rep = pd.DataFrame(np.repeat(df_values.values, len(time_seconds), axis=0))
    df_values_rep.columns = df_values.columns

    isochrone_gdf = gpd.GeoDataFrame(
        data=df_values_rep,
        geometry=isochrone_shapes,
        crs=4326
    )

    isochrone_gdf['time'] = time_col

    # We are sorting the dataframe in descending order of time to improve visualization
    # (the smallest isochrones should go on top, which means they are plotted last)
    isochrone_gdf = isochrone_gdf.sort_values('time', ascending=False)

    return isochrone_gdf
```

We'll generate an isochrone with layers for drive time of 15, 30, 45, and 60 minutes and produce the map with folium.

```python
# Generate isochrones for Truist Park
truist_isos = mb_isochrone(truist_park_gdf, time=[15, 30, 45, 60], profile="driving-traffic")

# Visualize the isochrones
truist_isos.explore(column="time")

import folium
from folium import features

# Initialize the Folium map
m = folium.Map(location=[truist_park_gdf.geometry.y.mean(), truist_park_gdf.geometry.x.mean()], zoom_start=12)

# Function to add isochrones to Folium map
def add_isochrones_to_map(isochrones, folium_map):
    color_map = {15: 'green', 30: 'yellow', 45: 'orange', 60: 'red'}

    for idx, row in isochrones.iterrows():
        color = color_map[row['time']]
        folium.GeoJson(row['geometry'].__geo_interface__, style_function=lambda x, color=color: {'color': color}).add_to(folium_map)

# Add isochrones to the map
add_isochrones_to_map(truist_isos, m)

# Add a custom marker for Truist Park
braves_logo = '~/Braves.png'  
for idx, row in truist_park_gdf.iterrows():
    icon = folium.CustomIcon(braves_logo, icon_size=(20, 20))  # Adjust the icon size as needed
    folium.Marker([row['Latitude'], row['Longitude']], popup=row['Stadium'], icon=icon).add_to(m)

# Display the map
m
```
The results are fascinating! Once we stop using straight-line distance calculations, we can start to see how road networks and other factors affect drive time to the stadiums. The isochrones are complex polygons and demonstrate that a relatively small portion of GA's population is within 60 minutes driving from Truist Park.

<iframe src="projects/mlb-analysis/outputs/truist_park_isochrones.html" width="100%" height="600px"></iframe>

### Adding demographic data
We can use pygris get_census to pull Census demographic data into this analysis of the areas around Truist Park. 

```python
from pygris.data import get_census
import pandas as pd

# Define the variables to fetch
variables = [
    "B01003_001E",  # Total Population
    "B19013_001E",  # Median Household Income
    "B25077_001E",  # Median Home Value
    "B02001_002E",  # White Population
    "B02001_003E",  # Black Population
    "B02001_004E",  # American Indian and Alaska Native Population
    "B02001_005E",  # Asian Population
    "B02001_006E",  # Native Hawaiian and Other Pacific Islander Population
    "B02001_007E",  # Some Other Race Population
    "B02001_008E"   # Two or More Races Population
]

# Fetch census data for the required variables
census_data = get_census(
    dataset='acs/acs5',
    variables=variables,
    year=2021,
    params={
        "for": "tract:*",
        "in": f"state:{state_fips}"
    },
    guess_dtypes=True,
    return_geoid=True
)

# Convert to DataFrame
census_df = pd.DataFrame(census_data)
```
This will allow us to see how proximity and population metrics like density, income, and race are related (or not). We'll analyze these demographic dimensions within the 60-minute isochrone around Truist Park for easy comparison. If you know Atlanta, the following maps will not surprise you.

![Population within 60 mins of Truist Park](projects/mlb-analysis/outputs/population_density-3.png)
![Median household income within 60 mins of Truist Park](projects/mlb-analysis/outputs/median_household_income-3.png)
![Black population within 60 mins of Truist Park](projects/mlb-analysis/outputs/black_population-3.png)
![White population within 60 mins of Truist Park](projects/mlb-analysis/outputs/white_population-3.png)

Here's how I created these maps using matplotlib.pyplot. 

```python
# Plot Population Density
fig, ax = plt.subplots(figsize=(10, 6))
tracts_with_census_data.plot(column='B01003_001E', cmap='viridis', legend=True, ax=ax)
ax.plot(truist_park_location.x, truist_park_location.y, marker='o', color='red', markersize=10, label='Truist Park')
plt.title('Population Density within 60-min Isochrone of Truist Park')
plt.legend(loc='upper right')
plt.show()
```
### Demographics and attendance
Unfortunately, I forgot to create a population _density_ variable before making those maps of Truist Park... so we're left with pure population numbers. So I figured why not analyze population density around each MLB ballpark?

I used similar code to the above to pull in the Census tracts, MLB stadiums, and Census demographic data within 20 miles of each MLB stadium (for consistency). I mapped each area using a logarithmic scale to account for the dramatic differences in population within the tracts (e.g., Kansas City vs. New York City).



### Correlation between demographics and attendance

## Final results

```python
def my_function():
    print("Hello, World!")
```

Here is some inline code: `print("Hello, World!")`.
