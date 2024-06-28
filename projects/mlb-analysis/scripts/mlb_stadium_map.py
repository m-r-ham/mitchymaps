import folium
import pandas as pd
import base64
import os

# Load the CSV file with geocoded data
csv_file_path = '/Users/mitchellhamilton/m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/data/mlb_stadiums_geocoded_logos.csv'
df = pd.read_csv(csv_file_path)

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

# Save the map to an HTML file
output_file_path = '/Users/mitchellhamilton/m-r-ham.github.io/mitchymaps.github.io/projects/mlb-analysis/outputs/mlb_ballparks_map.html'
folium_map.save(output_file_path)

print(f"Map saved to {output_file_path}")
