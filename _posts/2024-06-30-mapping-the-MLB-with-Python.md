---
layout: post
title: "Mapping Major League Baseball with Python"
date: 2024-06-30
categories: projects
---

# Mapping the MLB with Python

## Introduction

Welcome to Explorations in Geospatial Analysis (aka Mitchy Maps). I've been exploring the world of GIS and geospatial analytics in my work recently. It's been slow going so far, but every time I make progress I'm excited and more energized to learn about this space. I created this blog as a way to learn and share my progress.

The world of data science is generally opaque for people who aren't trained data scientists. Geospatial data science is even more opaque. This will broadly be an attempt to help data-curious people like me understand how to use common tools, techniques, and libraries to perform geospatial analysis for business & pleasure. Thanks for following along!

## Project Overview

People who know me well know that I love baseball (go Braves!). As a sort of ceremonial first pitch for this blog, I decided to map MLB stadiums and explore the areas around them. I wanted to see at a basic level whether population density and other demographic factors might be related to team attendance. Short answer: they're not really related at all. We'll get deeper into the answer throughout the post.

I used python exclusively for this analysis. There are many great geospatial and geospatial-adjacent libraries available for python, including geopandas, geopy, shapely, folium, pygris, cenpy, and many more. I'll go into more detail about the libraries I used in the detailed summary of my analysis below. 

I got inspiration from https://walker-data.com/posts for some of this analysis.

## Analysis
Distance and proximity analyses are common in spatial data science. It's often important to understand how close something is to something else. Proximity analysis can give a business clues about where to place a new location or help a government understand a population's access to critical resources like hospitals. For this project, I started by exploring how close/far people are from MLB stadiums.

### Mapping MLB stadiums
First, I copied MLB stadium names & addresses from [`MLB`](#https://www.mlb.com/team) and stadium capacity from [`Wikipedia`](#https://en.wikipedia.org/wiki/List_of_current_Major_League_Baseball_stadiums) into a csv file. I thought about scraping this data, but it was easier/quicker to collect the data manually.

Then, I geocoded the MLB ballparks using the Photon geocoder within geopy. Geocoding is the process of converting addresses, place names, or other location-based data into geographic coordinates (latitude and longitude). Geocoding is an important part of geospatial analytics for many reasons but particularly for mapping and integrating with other geographic data.

<details>
<summary>Click to expand code</summary>

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
</details>

Usually, I use Nominatim for geocoding addresses in Python but I kept getting a 403 error code regardless of what I used for my parameters, so I switched to Photon for this analysis. There are many ways to geocode addresses, most of them free in small batches. My favorites are Nominatim through geopy and the [`Census Geocoder`](#https://www.census.gov/programs-surveys/geography/technical-documentation/complete-technical-documentation/census-geocoder.html) batch processing tool. The Google Maps and Mapbox APIs are quite robust but also can be expensive. 

It's also important to note that the input data quality affects geocoding success and accuracy quite dramatically. The Google Maps API can handle geocoding from a place name input like 'Truist Park', but many geocoders require an address. You can achieve accurate results quite often with quality address inputs including Street, City, State, and ZIP.

Once the stadium addresses are geocoded, we can add move on to mapping. This involves importing a few libraries, including folium for the map and os & base64 to be able to use SVG images of team logos as the markers on the map.

<details>
    <summary>Click to expand code</summary>

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
</details>
```

We ended up with a simple map of MLB stadiums (the overlap of some teams is not ideal, but it’s cooler to have the team logos than basic dots in my opinion!).

<iframe src="https://raw.githubusercontent.com/m-r-ham/mitchymaps.github.io/main/projects/mlb-analysis/outputs/mlb_ballparks_map.html" width="100%" height="600px"></iframe>

This is nice, but it doesn't tell us anything. Proximity analysis can help us understand the population's access to MLB stadiums.

### Proximity analysis
MLB stadiums are located in diverse areas of their cities, so I wanted to understand the areas that are near MLB stadiums. To do this, we had to have some locations to compare against the MLB stadiums. I used Census tracts as my geographic areas because they’re small and have a lot of available Census data attached to them. Census tracts are small geographic areas with 1,200-8,000 people. They're great for geospatial & demographic analysis because they're much more comparable to each other in terms of population size than other geographic areas like ZIP codes and counties.

First, we pulled the census tracts in via pygris and used geopandas to load the MLB stadium data we gathered earlier.

<details>
    <summary>Click to expand code</summary>

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
</details>
```

Then, we wanted to find out which MLB stadiums are _near_ each census tract. We don't want to search by state here because some Census tracts, and therefore people, are closest to an MLB stadium outside their state. For example, some Maryland residents are closer to Nationals Park in DC than they are to Camden Yards in Baltimore. To do this, we identified all MLB stadiums within 100 km of the census tracts. We use a sequence of common GIS operations with geopandas to accomplish this. Our steps include:

- Combining the Census tracts into a single shape with the .dissolve() method;
- Drawing a new shape that extends to 100km beyond the border of all tracts with the .buffer() method;
- Using an inner spatial join to retain only the stadiums that fall within the 100km buffer shape (all of them).

<details>
    <summary>Click to expand code</summary>

```python
# Create a buffer around the dissolved census tracts
buffer_distance = 100000  # Distance in meters

# Dissolve all tracts into one geometry and create a buffer
national_buffer = gpd.GeoDataFrame(geometry=all_census_tracts.dissolve().buffer(buffer_distance))

# Spatial join between the stadiums and the buffer to find stadiums within the buffer
stadiums_within_buffer = gpd.sjoin(stadiums_gdf, national_buffer, how="inner", op="within")

print("Buffer created and spatial join completed.")
</details>
```

After running this operation, we can draw a quick plot to show the relationships between Census tracts and MLB stadiums.

<details>
    <summary>Click to expand code</summary>

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
</details>
```

The output looks like this, confirming that we have captured all MLB stadiums for future analysis.

![Map of the Census tracts with MLB stadiums](https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/mlb_stadiums_within_buffer.png)

Now, we have to caclulate the distance from each Census tract to the nearest MLB stadium to get a proximity distribution. Census tracts are geographic areas (polygons), not points, so we have to first identify the "centroid" of each Census tract. We commonly use centroids to calculate distances and understand relationships between complex shapes and points in geospatial analysis. By using centroids, you can easily measure the distance between the center of a polygon and other points (e.g., comparing the distance between the center of a city boundary and specific locations like stores or schools). 

We had to re-project the coordinates to a different Coordinate Reference System (CRS) to enable easy comparison.

<details>
    <summary>Click to expand code</summary>
    
```python
# Reproject to a projected CRS for accurate distance calculations
projected_crs = 2163  # US National Atlas Equal Area
all_census_tracts = all_census_tracts.to_crs(epsg=projected_crs)
stadiums_gdf = stadiums_gdf.to_crs(epsg=projected_crs)
</details>
```

Then, we calculated the centroid of each Census tract, calculated the distance from each centroid to the stadiums to identify the nearest one, added the distance back to the census gdf, and, finally, mapped it!

<details>
    <summary>Click to expand code</summary>
    
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
</details>
```

![Map of travel distance to MLB stadiums](https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/travel_distance_to_stadiums.png)

This is super interesting, but not unexpected. There are many areas in the US where people are quite far from an MLB stadium, especially across the great plains and parts of the western US (though this will change once Oakland moves to Sacramento and then Las Vegas). The average distance from a Census tract to an MLB stadium is 128 miles, but a simple histogram shows that most Census tracts are within 100 miles of a stadium.

![Histogram of travel time distribution](https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/travel_time_distribution.png)

We know that in many areas, however, straight-line distances can be misleading. Let's look at drive time in addition to distance to get a better sense of which Census tracts are "close" to MLB stadiums.

### Drive time analysis: Truist Park
We can connect to Mapbox’s navigation services with the routingpy package, an interface to several hosted navigation APIs. This allows us to calculate drive time for our dataset. The Mapbox API has a meaningful free tier, but there are thousands of Census tracts and 30 MLB stadiums, so we would have spent $1,000+ on this analysis had we calculated drive time for every stadium. Therefore, I focused on Truist Park in Atlanta for the remaining analysis.

We start with importing the necessary libraries and connecting to the Mapbox API via routingpy.

<details>
    <summary>Click to expand code</summary>

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
</details>
```

Then, we followed the same steps as above to gather Census tracts, dissolve them into a single shape, and merge them with the stadium data. This time, however, we only collected Census data for GA (FIPS code '13'). After we have the data loaded and prepared, we can generate a list of coordinate pairs to feed to Mapbox. We ultimately calculated 2,300 'elements', the output of the [Mapbox Matrix API](#https://docs.mapbox.com/api/navigation/matrix/). This was almost 25% of our monthly free limit, so it's a good thing we focused on 1 state instead of all 50.

<details>
    <summary>Click to expand code</summary>

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
</details>
```

Now we have the data we need on drive times to do some meaningful analysis and visualization. We start by calculating the minimum travel time in minutes to the nearest stadium (which will be Truist Park in all cases). Then, we plot the results! 

<details>
    <summary>Click to expand code</summary>

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
</details>
```

The map is not surprising at all. Areas far away from Truist Park are far in terms of distance and drive time. Truist Park is located in Metro Atlanta, so the Atlanta area has the easiest access to the stadium. There are no stadiums in north Florida or in TN/SC, so people in N/S Georgia do not have access to another nearby stadium.

![Map of GA Census tract travel times to Truist Park](https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/travel_time_to_truist.png)

### Travel time isochrones
Isochrones are a really interesting way to show layers of proximity and travel time from a given point. An isochrone is a line or area on a map that represents all points that can be reached within a certain time or distance from a given location. Isochrones are commonly used in transportation planning, logistics, and urban planning to visualize accessibility and travel times.

We can use isochrones to explore the immediate areas around Truist Park in more detail. We'll use the Mapbox API via routingpy again to generate an isochrone. We'll reload the stadiums data into a new dataframe, then call the Mapbox API to generate the isochrones, and map the results.

<details>
    <summary>Click to expand code</summary>

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
</details>
```

We'll generate an isochrone with layers for drive time of 15, 30, 45, and 60 minutes and produce the map with folium.

<details>
    <summary>Click to expand code</summary>

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
</details>
```

The results are fascinating! Once we stop using straight-line distance calculations, we can start to see how road networks and other factors affect drive time to the stadiums. The isochrones are complex polygons and demonstrate that a relatively small portion of GA's population is within 60 minutes driving from Truist Park.

<iframe src="[https://m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/outputs/truist_park_isochrones.html](https://github.com/m-r-ham/mitchymaps.github.io/blob/1fba5e31935db54a9c079aa46dfbd8ab44719eb2/projects/mlb-analysis/outputs/truist_park_isochrones.html)" width="100%" height="600px"></iframe>

### Adding demographic data
We can use pygris get_census to pull Census demographic data into this analysis of the areas around Truist Park. 

<details>
    <summary>Click to expand code</summary>

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
</details>
```

This will allow us to see how proximity and population metrics like density, income, and race are related (or not). We'll analyze these demographic dimensions within the 60-minute isochrone around Truist Park for easy comparison. If you know Atlanta, the following maps will not surprise you.

<div style="display: flex; flex-wrap: wrap; gap: 10px;">

  <div style="flex: 1 1 calc(50% - 10px);">
    <img src="https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/population_density-3.png" alt="Population within 60 mins of Truist Park" style="width: 100%; height: auto;">
    <p>Population within 60 mins of Truist Park</p>
  </div>

  <div style="flex: 1 1 calc(50% - 10px);">
    <img src="https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/median_household_income-3.png" alt="Median household income within 60 mins of Truist Park" style="width: 100%; height: auto;">
    <p>Median household income within 60 mins of Truist Park</p>
  </div>

  <div style="flex: 1 1 calc(50% - 10px);">
    <img src="https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/black_population-3.png" alt="Black population within 60 mins of Truist Park" style="width: 100%; height: auto;">
    <p>Black population within 60 mins of Truist Park</p>
  </div>

  <div style="flex: 1 1 calc(50% - 10px);">
    <img src="https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/white_population-3.png" alt="White population within 60 mins of Truist Park" style="width: 100%; height: auto;">
    <p>White population within 60 mins of Truist Park</p>
  </div>

</div>

Here's how I created these maps using matplotlib.pyplot. 

<details>
    <summary>Click to expand code</summary>

```python
# Plot Population Density
fig, ax = plt.subplots(figsize=(10, 6))
tracts_with_census_data.plot(column='B01003_001E', cmap='viridis', legend=True, ax=ax)
ax.plot(truist_park_location.x, truist_park_location.y, marker='o', color='red', markersize=10, label='Truist Park')
plt.title('Population Density within 60-min Isochrone of Truist Park')
plt.legend(loc='upper right')
plt.show()
</details>
```

### Demographics and attendance
Unfortunately, I forgot to create a population _density_ variable before making those maps of Truist Park... so we're left with pure population numbers in the "density" visualization above. So I figured why not analyze population density around each MLB ballpark?

I used similar code to the above to pull in the Census tracts, MLB stadiums, and Census demographic data within 20 miles of each MLB stadium (for consistency). I mapped each area using a logarithmic scale to account for the dramatic differences in population within the tracts (e.g., Kansas City vs. New York City).

<div style="display: flex; flex-wrap: wrap; gap: 10px;">

  <div style="flex: 1;">
    <img src="https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/logarithmic_population_density_New_York_Yankees.png" alt="Population density around Yankee Stadium" style="width: 100%; height: auto;">
    <p style="text-align: center;">Population density around Yankee Stadium</p>
  </div>

  <div style="flex: 1;">
    <img src="https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/logarithmic_population_density_Kansas_City_Royals.png" alt="Population density around Kansas City Royals" style="width: 100%; height: auto;">
    <p style="text-align: center;">Population density around Kauffman Stadium</p>
  </div>

</div>

Yankee Stadium is squarely within NYC in a densely populated area of the Bronx. Kauffman Stadium is a good bit outside of Kansas City, which already has a much lower population than NYC. The population density near the stadiums therefore differs dramaticaly.

### Correlation between demographics and attendance
My final question for this analysis was, "Is attendance correlated with population dynamics like density and household income?" To analyze this, I created a simple regression model.

I started with team attendance. The top 5 teams last year by average attendance per game were: 

1. Los Angeles Dodgers: 47,371 
2. San Diego Padres: 40,390
3. New York Yankees: 40,358
4. St. Louis Cardinals: 40,013
5. Atlanta Braves: 39,401

Interestingly, the area around Busch Stadium in St. Louis has the 3rd lowest population density and 3rd lowest median income out of all 30 MLB teams. That didn't stop them from making it into the top 5 in attendance per game last year (even though they also sucked). There are obviously other factors at play, such as strength of the fandom and baseball culture in the city, ease of transportation and parking, cost, capacity, schedule, and many others.

Even with this data in hand, I still wanted to see the results of the regression.

<details>
    <summary>Click to expand code</summary>

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Ensure all data is numeric
numeric_cols = ['population_density', 'avg_median_income', 'AttendancePerGame_2023']
stadium_data_numeric = stadium_data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
stadium_data_numeric = stadium_data_numeric.dropna()

# Calculate correlations
correlations = stadium_data_numeric.corr()
print("Correlations with Attendance:")
print(correlations['AttendancePerGame_2023'])

# Single Variable Linear Regression
X_density = stadium_data_numeric[['population_density']]
y = stadium_data_numeric['AttendancePerGame_2023']
X_income = stadium_data_numeric[['avg_median_income']]

model_density = sm.OLS(y, sm.add_constant(X_density)).fit()
model_income = sm.OLS(y, sm.add_constant(X_income)).fit()

print("\nLinear Regression Model: Population Density")
print(model_density.summary())
print("\nLinear Regression Model: Average Median Income")
print(model_income.summary())

# Multiple Regression Analysis
X = stadium_data_numeric[['population_density', 'avg_median_income']]
model = sm.OLS(y, sm.add_constant(X)).fit()

print("\nMultiple Regression Model")
print(model.summary())

# Visualize relationships
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.scatterplot(data=stadium_data_numeric, x='population_density', y='AttendancePerGame_2023', ax=axes[0])
axes[0].set_xlabel('Population Density')
axes[0].set_ylabel('Attendance Per Game (2023)')
axes[0].set_title('Population Density vs Attendance')
sns.regplot(data=stadium_data_numeric, x='population_density', y='AttendancePerGame_2023', ax=axes[0], scatter=False, color='red')

sns.scatterplot(data=stadium_data_numeric, x='avg_median_income', y='AttendancePerGame_2023', ax=axes[1])
axes[1].set_xlabel('Average Median Income')
axes[1].set_ylabel('Attendance Per Game (2023)')
axes[1].set_title('Income vs Attendance')
sns.regplot(data=stadium_data_numeric, x='avg_median_income', y='AttendancePerGame_2023', ax=axes[1], scatter=False, color='red')

plt.tight_layout()
plt.savefig('mlb_attendance_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print(stadium_data_numeric.describe())
</details>
```

The results? With this simple analysis, there was almost 0 correlation between household income and population density around stadiums with attendance. 

![Plot of correlation between demographic factors and MLB attendance](https://github.com/m-r-ham/mitchymaps.github.io/blob/657ace14fa61eb238d1a5bd3a9c59915aac5b7f8/projects/mlb-analysis/outputs/mlb_attendance_correlations copy.png)

The R^2 value for both regressions was under 0.1, indicating very low correlation. As I mentioned, there are clearly many other factors influencing attendance. Maybe I'll create a more robust model another time!

Thanks for following along with my first public analysis/post! I'll be exploring other interesting topics in the weeks and months to come. 
