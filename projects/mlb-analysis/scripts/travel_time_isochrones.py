import geopandas as gp
import pandas as pd
from shapely.geometry import Polygon
from routingpy.routers import MapboxOSRM
import numpy as np

mb = MapboxOSRM(api_key = "pk.eyJ1IjoibWl0Y2h5bWFwcyIsImEiOiJjbHh2YTQ0c3gwa2l4MnFvb3F2aXQzeHNyIn0.qtVm5jR-fYTJaWZB3Su29w")

mlb_stadiums = gp.read_file('/Users/mitchellhamilton/m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/data/mlb_stadiums_geocoded.csv')

mlb_stadiums.explore()