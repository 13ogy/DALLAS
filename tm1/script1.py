# This script is a complete, self-contained solution to scrape the main Olympic Games table
# from a Wikipedia page. It downloads the HTML, saves it, and then parses the data.

import bs4
import pandas as pd
import requests
import re
import os

def scrape_olympics_table(html_content):
    """
    Parses the HTML content to find the main table of Summer Olympic Games and
    extracts its data into a pandas DataFrame.
    """
    page = bs4.BeautifulSoup(html_content, "lxml")
    
    # Find the main table for "Jeux olympiques d'été"
    # This table has the class 'wikitable center'
    table = page.find('table', {'class': 'wikitable center'})

    if not table:
        print("Error: Could not find the main table on the page.")
        return None

    # Find the table body and extract all rows
    tbody = table.find('tbody')
    if not tbody:
        print("Error: Table body (<tbody>) not found.")
        return None
    
    # The last row is for notes, so we slice it off with [:-1]
    rows = tbody.find_all('tr')[:-1]

    data = []
    # Keep track of the year for multi-row entries (e.g., 1940)
    current_year = None
    current_edition = None

    for row in rows:
        cells = row.find_all(['td', 'th'])
        row_data = [cell.text.strip() for cell in cells]

        # Handle rows with fewer columns by using the data from the previous row
        if len(row_data) < 5:
            # Check if there is a valid previous year
            if current_year is not None:
                row_data.insert(0, current_year)
                row_data.insert(1, current_edition)
            
        else:
            # This is a new main row, so we capture the year and edition
            current_year = row_data[0]
            current_edition = row_data[1]

        data.append(row_data)

    # Define headers based on the table's structure
    headers = ['Année', 'Édition', 'Ville', 'Pays', 'Continent']
    
    # Create the DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    return df

if __name__ == "__main__":
    url = "https://fr.wikipedia.org/wiki/Jeux_olympiques"
    file_name = "file.html"

    print(f"Attempting to download HTML from: {url}")
    try:
        # User-Agent header to mimic a browser and avoid a 403 Forbidden error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        # Step 1: Download the HTML content
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Step 2: Save the HTML to a local file
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"Successfully saved HTML content to '{file_name}'.")

        # Step 3: Read the local HTML file and begin scraping
        with open(file_name, "r", encoding="utf-8") as file:
            request_text = file.read()
        
        # We can see the raw HTML content was successfully fetched, as in your output.
        # Now, we proceed to scrape the data from the content.
        df = scrape_olympics_table(request_text)

        if df is not None:
            print("\nSuccessfully scraped the table and created a DataFrame:")
            print(df)
            print(f"\nDataFrame shape: {df.shape}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the URL: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")