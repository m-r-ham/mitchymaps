import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def get_mlb_teams():
    url = "https://www.mlb.com/team"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    teams_data = []
    
    team_elements = soup.find_all('div', class_='p-featured-content__body')
    
    for team in team_elements:
        name_element = team.find('span', class_='u-text-h4')
        if name_element:
            team_name = name_element.text.strip()
            
            # Extract full address
            address_element = team.find('div', class_='p-featured-content__address')
            if address_element:
                address_lines = address_element.find_all('div')
                full_address = ' '.join([line.text.strip() for line in address_lines])
                
                # Extract stadium name
                stadium_name = address_lines[0].text.strip() if address_lines else "N/A"
            else:
                full_address = "N/A"
                stadium_name = "N/A"
            
            teams_data.append({
                'Team': team_name,
                'Stadium': stadium_name,
                'Address': full_address
            })
    
    return pd.DataFrame(teams_data)

if __name__ == "__main__":
    mlb_teams = get_mlb_teams()
    print(mlb_teams)
    mlb_teams.to_csv('data/mlb_stadiums.csv', index=False)
    print(f"\nTotal number of teams: {len(mlb_teams)}")
    print("Data saved to 'data/mlb_stadiums.csv'")
