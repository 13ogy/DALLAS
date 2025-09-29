# This script performs a multi-step web scraping and data visualization process.
# 1. It downloads the main Wikipedia page for the Olympic Games.
# 2. It scrapes the page to get a list of host cities and their individual URLs.
# 3. It visits each city's page to scrape its geographic coordinates (latitude and longitude).
# 4. It stores the data in a pandas DataFrame.
# 5. It uses the data to create an interactive map with markers for each city.

import bs4
import pandas as pd
import requests
import re
import os
import folium
import geopandas as gpd
from pathlib import Path
import time

# A User-Agent header to mimic a browser, avoiding a 403 Forbidden error.
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

def dms2dd(dms_string):
    """
    Converts a Degree, Minute, Second (DMS) string to decimal degrees.
    This function is useful for parsing coordinates from alternative sources.
    It's not used in this script, as the scraped data is already in decimal format.
    """
    try:
        parts = re.split(r'[^\d\w]+', dms_string)
        if len(parts) < 4:
            return None
        
        degrees = float(parts[0])
        minutes = float(parts[1]) if parts[1] else 0.0
        seconds = float(parts[2]) if parts[2] else 0.0
        direction = parts[3].upper()

        dd = degrees + minutes / 60 + seconds / 3600
        
        if direction in ('S', 'O', 'W'):
            dd *= -1
        return dd
    except (ValueError, IndexError):
        return None

def scrape_wikipedia_table(html_content):
    """
    Parses the main Olympic Games table from the HTML content and returns
    a unique list of city dictionaries, each with a name and a URL.
    """
    page = bs4.BeautifulSoup(html_content, "lxml")
    table = page.find('table', {'class': 'wikitable center'})

    if not table:
        print("Error: Could not find the main table on the page.")
        return []

    rows = table.find('tbody').find_all('tr')
    unique_cities = {}

    for row in rows:
        # Find all <td> and <th> elements in the row
        cells = row.find_all(['td', 'th'])
        for cell in cells:
            # Look for a link within each cell
            city_link = cell.find('a')
            if city_link:
                # The text of the link contains the city name and a count like "(1)"
                city_name = city_link.text.strip().split('(')[0].strip()
                city_url = f"https://fr.wikipedia.org{city_link.get('href')}"
                
                # We only want actual cities, not links to pages like "Jeux olympiques de 1900"
                if '/wiki/' in city_link.get('href') and city_name not in unique_cities and not city_name.isdigit():
                    unique_cities[city_name] = city_url
            
    # Convert the dictionary back to a list of dictionaries
    cities_to_scrape = [{'name': name, 'url': url} for name, url in unique_cities.items()]
    return cities_to_scrape

def scrape_city_coordinates(city_url):
    """
    Visits a city's Wikipedia page and scrapes its decimal coordinates.
    """
    try:
        response = requests.get(city_url, headers=HEADERS)
        response.raise_for_status()
        city_page = bs4.BeautifulSoup(response.text, "lxml")
        
        # Look for coordinates in the infobox or on the page
        coords_element = city_page.find('span', {'class': 'geo'})
        if coords_element:
            coords_str = coords_element.text.strip().replace('\u2009', '')
            coords_list = coords_str.split(';')
            if len(coords_list) == 2:
                lat = float(coords_list[0].strip())
                lon = float(coords_list[1].strip())
                return lat, lon

        # Fallback to the mw-kartographer-maplink attributes
        map_link = city_page.find('a', {'class': 'mw-kartographer-maplink'})
        if map_link:
            data_lat = map_link.get('data-lat')
            data_lon = map_link.get('data-lon')
            if data_lat and data_lon:
                return float(data_lat), float(data_lon)

    except Exception as e:
        print(f"    Error scraping coordinates from {city_url}: {e}")
    
    return None, None

def create_olympic_map(data):
    """
    Creates an interactive Folium map with markers for each Olympic city.
    """
    # Create a GeoDataFrame from the scraped data
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))
    
    # Calculate the center and bounds of the map
    center = gdf[['latitude', 'longitude']].mean().values.tolist()
    sw = gdf[['latitude', 'longitude']].min().values.tolist()
    ne = gdf[['latitude', 'longitude']].max().values.tolist()
    
    # Create the map
    m = folium.Map(location=center, tiles='openstreetmap')
    
    # Add a marker for each city
    for i, row in gdf.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=row['name']
        ).add_to(m)
        
    m.fit_bounds([sw, ne])
    
    # Save the map to an HTML file
    map_filepath = "olympic_cities_map.html"
    m.save(map_filepath)
    print(f"\nMap saved to {map_filepath}. Open the file in a browser to view it.")
    

if __name__ == "__main__":
    main_url = "https://fr.wikipedia.org/wiki/Jeux_olympiques"
    print(f"Attempting to download HTML from: {main_url}")
    
    try:
        response = requests.get(main_url, headers=HEADERS)
        response.raise_for_status()
        main_page_html = response.text
        
        print("Scraping main page for city information...")
        cities = scrape_wikipedia_table(main_page_html)

        if not cities:
            print("No cities found on the main page. Exiting.")
        else:
            print(f"Found {len(cities)} unique host cities. Now scraping each city page for coordinates...")
            
            city_data = []
            
            for city_info in cities:
                lat, lon = scrape_city_coordinates(city_info['url'])
                if lat is not None and lon is not None:
                    city_data.append({
                        'name': city_info['name'],
                        'url': city_info['url'],
                        'latitude': lat,
                        'longitude': lon
                    })
                    print(f"  -> Found coordinates for {city_info['name']}: {lat}, {lon}")
                else:
                    print(f"  -> No coordinates found for {city_info['name']}")
                
                # Add a delay to avoid overloading the server
                time.sleep(1)

            df = pd.DataFrame(city_data)
            
            if not df.empty:
                print("\nFinal DataFrame of scraped cities and their coordinates:")
                print(df.head())
                create_olympic_map(df)
            else:
                print("\nNo city data with coordinates was successfully scraped.")
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the URL: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
