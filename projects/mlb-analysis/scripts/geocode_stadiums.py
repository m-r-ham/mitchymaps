import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Load the CSV file
csv_file_path = '/Users/mitchellhamilton/m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/data/mlb_stadiums.csv'  # Update with your actual file path
df = pd.read_csv(csv_file_path)

# Initialize the geocoder
geolocator = Nominatim(user_agent="geoapiExercises")

# Function to geocode an address
def geocode_address(address, retries=3):
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderTimedOut:
        if retries > 0:
            time.sleep(1)
            return geocode_address(address, retries - 1)
        else:
            return None, None
    except Exception as e:
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
output_csv_file_path = '/Users/mitchellhamilton/m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/data/mlb_stadiums_geocoded.csv'  # Update with your desired output file path
df.to_csv(output_csv_file_path, index=False)

print(f"Geocoded addresses saved to {output_csv_file_path}")