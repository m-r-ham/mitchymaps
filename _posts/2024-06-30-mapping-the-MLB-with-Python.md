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

### Mapping MLB stadiums
I started by gathering the data on ballparks & their capacities from https://www.mlb.com/team & https://en.wikipedia.org/wiki/List_of_current_Major_League_Baseball_stadiums. I thought about scraping this data, but it was easier/quicker to collect the data manually.

Then, I geocoded the MLB ballparks using geopy and Photon. Geocoding is the process of… Geocoding is an important part of mapping and geospatial analytics because it enables… 

```python
def my_function():
    print("Hello, World!")
```

Usually, I use Nominatim for quick geocoding in python but I kept getting a 403 error code regardless of what I used for my parameters, so I switched to Photon for this analysis. There are many ways to geocode addresses, most of them free in small batches. My favorites are Nominatim and the Census Geocoder batch processing tool. The Google Maps and Mapbox APIs are quite robust but also can be expensive.

After geocoding, I mapped the MLB stadiums.

```python
def my_function():
    print("Hello, World!")
```

We ended up with a simple map of MLB stadiums (the overlap of some teams is not ideal, but it’s cooler to have the team logos than basic dots in my opinion!).

![Map of the US with MLB stadiums indicated by team logos](/path/to/your/image.png)

### Proximity
MLB stadiums are located in diverse areas of their cities, so I wanted to understand the areas that are near MLB stadiums. I used Census tracts as my geographic areas because they’re small and have a lot of available Census data attached to them. Census tracts are small geographic areas with 1,200-8,000 people. They're great for geospatial + demographic analysis because they're much more easily comparable than larger geographic areas like ZIP codes and counties, which are generally larger and more diverse in both area and population.

First, we pulled the census tracts in via pygris and used geopandas to load the MLB stadium data we gathered earlier.

```python
def my_function():
    print("Hello, World!")
```

Then, we wanted to find out which MLB stadiums are _near_ each census tract. We don't want to search by state here because some Census tracts, and therefore people, are closest to an MLB stadium outside their state. For example, some Maryland residents are closer to Nationals Park in DC than they are to Camden Yards in Baltimore. To do this, we identified all MLB stadiums within 100 km of the census tracts. We use a sequence of common GIS operations with geopandas to accomplish this. Our steps include:

- Combining the Census tracts into a single shape with the .dissolve() method;
- Drawing a new shape that extends to 100km beyond the border of the US with the .buffer() method;
- Using an inner spatial join to retain only those trauma centers that fall within the 100km buffer shape.

```python
def my_function():
    print("Hello, World!")
```

After running this operation, we can draw a quick plot to show the relationships between Census tracts and MLB stadiums.

![Map of the Census tracts with MLB stadiums](/path/to/your/image.png)

Then, we calculated the distance from the centroid of each census tract to the nearest MLB stadium. The average distance is 128 miles. The number of tracts within 10 miles of a stadium is 8,851 (only 12% of all tracts). 

REWRITE ---> "The simplest way to calculate proximity is with straight-line distances. In a projected coordinate system, this amounts to little more than Euclidean geometry, and such distance calculations are readily available to us using the .distance() method in geopandas. Conceptually, we’ll want to think through how to represent polygon-to-point distances. The most accurate approach would likely be to find the population-weighted centroid of each Census tract using some underlying dataset like Census blocks. In the example here, I’m taking a more simplistic approach by finding the geographic centroid of each tract. The centroids are found in the centroid attribute of any polygon GeoDataFrame.

Once the centroids are identified, we can iterate over them with apply and build a dataset that represents a distance matrix between Census tracts and trauma hospitals. Distances are calculated in meters, the base unit of our coordinate reference system (State Plane South Dakota North)."

```python
def my_function():
    print("Hello, World!")
```

Now that we've calculated the distance between Census tracts (their centroids, at least), we can plot a histogram to see the distribution of tract distance to the nearest MLB stadium.

![Histogram of distances from Census tract centroids to the closest MLB stadium](/path/to/your/image.png)

REWRITE ---> "Over 60 Census tracts are within 10km of a trauma center, reflecting tracts located within the population centers of Rapid City and Sioux Falls. However, many tracts in the state are beyond 100km from the nearest trauma center, with 26 200km or more away.

We know that in rural areas, however, straight-line distances can be misleading. Given the geography of highway networks, accessibility to a trauma center is mediated through accessibility to that road network." Let's look at drive time in addition to distance to get a better sense of which Census tracts are "close" to MLB stadiums.

### Drive time
REWRITE --> "Analysts who need to calculate proximity along a road network in Python have multiple options available to them. In Python, we can connect to Mapbox’s navigation services with the routingpy package, an interface to several hosted navigation APIs."


Isochrones are a really interesting way... 

### Demographics

### Correlation

## Results

```python
def my_function():
    print("Hello, World!")
```

Here is some inline code: `print("Hello, World!")`.
