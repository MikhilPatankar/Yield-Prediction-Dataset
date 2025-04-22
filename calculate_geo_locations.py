import ee
import pandas as pd
import geopandas as gpd
import json
import zipfile
import os
import time
import warnings
import csv # Use the csv module for efficient appending
from datetime import datetime

# Ignore warnings
warnings.filterwarnings('ignore', module='ee')
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 0. Configuration ---
INPUT_CSV_PATH = 'crop_production.csv' # Path to your input crop production data
GEOJSON_ZIP_PATH = 'gadm41_IND_2.json.zip' # Path to your GeoJSON zip file
# Output file for *only* the GEE results
GEE_RESULTS_CSV_PATH = 'gee_seasonal_results.csv'
SAVE_EVERY_N_COMBINATIONS = 50 # Save results to CSV after every N combinations

print(f"--- Configuration ---")
print(f"Input Crop CSV: {INPUT_CSV_PATH}")
print(f"Input GeoJSON Zip: {GEOJSON_ZIP_PATH}")
print(f"Output GEE Results CSV: {GEE_RESULTS_CSV_PATH}")
print(f"Saving progress every: {SAVE_EVERY_N_COMBINATIONS} combinations")
print(f"---")

# --- 1. Authentication and Initialization ---
try:
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    print("Google Earth Engine initialized successfully (High Volume Endpoint).")
except Exception as e:
    print(f"Standard Earth Engine initialization failed: {e}. Attempting authentication...")
    try:
        ee.Authenticate()
        ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
        print("Google Earth Engine authenticated and initialized successfully (High Volume Endpoint).")
    except Exception as auth_e:
        print(f"Authentication and Initialization failed: {auth_e}")
        print("Please ensure you have authenticated Google Earth Engine access.")
        exit()

# --- 2. Load and Prepare GeoJSON Data ---
print(f"\nLoading GeoJSON from {GEOJSON_ZIP_PATH}...")
# (Same GeoJSON loading and preparation logic as before)
geojson_filename = None
try:
    with zipfile.ZipFile(GEOJSON_ZIP_PATH, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith('.json'):
                geojson_filename = name
                extract_path = os.path.splitext(GEOJSON_ZIP_PATH)[0] + "_extracted"
                os.makedirs(extract_path, exist_ok=True)
                z.extract(geojson_filename, path=extract_path)
                geojson_filepath = os.path.join(extract_path, geojson_filename)
                print(f"Extracted {geojson_filename}")
                break
        if not geojson_filename:
             raise FileNotFoundError("No .json file found inside the zip archive.")

    gdf = gpd.read_file(geojson_filepath)
    print(f"Loaded {len(gdf)} features from GeoJSON.")

    def clean_name(name):
        if not isinstance(name, str): return ""
        return ''.join(filter(str.isalnum, name)).lower()

    gdf['clean_NAME_1'] = gdf['NAME_1'].apply(clean_name)
    gdf['clean_NAME_2'] = gdf['NAME_2'].apply(clean_name)

    geometry_lookup = {
        (row['clean_NAME_1'], row['clean_NAME_2']): row.geometry # Store shapely geometry
        for index, row in gdf.iterrows()
    }
    print(f"Created geometry lookup for {len(geometry_lookup)} unique state/district combinations.")

except Exception as e:
    print(f"Error loading or processing GeoJSON: {e}")
    exit()


# --- 3. Load Input CSV and Get Unique Combinations ---
print(f"\nLoading Input CSV to find unique combinations: {INPUT_CSV_PATH}...")
try:
    df_input = pd.read_csv(INPUT_CSV_PATH)
    # Drop rows with missing key values if any - important for unique combinations
    df_input.dropna(subset=['State_Name', 'District_Name', 'Crop_Year', 'Season'], inplace=True)

    # Clean names needed for matching with GeoJSON keys
    df_input['clean_State_Name'] = df_input['State_Name'].apply(clean_name)
    df_input['clean_District_Name'] = df_input['District_Name'].apply(clean_name)
    df_input['Crop_Year'] = df_input['Crop_Year'].astype(int)
    # Standardize Season names (strip whitespace, maybe convert case)
    df_input['Season'] = df_input['Season'].str.strip()

    unique_combinations = df_input[['clean_State_Name', 'clean_District_Name', 'Crop_Year', 'Season']].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_combinations)} unique State/District/Year/Season combinations to process.")

except Exception as e:
    print(f"Error loading or processing Input CSV: {e}")
    exit()

# --- 4. Define Season Date Function ---
def get_season_dates(year, season_name):
    """Maps year and season name to start and end dates."""
    season_name_lower = season_name.lower()

    if 'kharif' in season_name_lower: # June to October
        start_date = f"{year}-06-01"
        end_date = f"{year}-10-31"
    elif 'rabi' in season_name_lower: # November to March (next year)
        start_date = f"{year}-11-01"
        end_date = f"{year + 1}-03-31" # Crosses year boundary
    elif 'summer' in season_name_lower: # March to May
        start_date = f"{year}-03-01"
        end_date = f"{year}-05-31"
    elif 'whole year' in season_name_lower: # Jan to Dec
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
    elif 'autumn' in season_name_lower: # Roughly Sept-Nov, often overlaps Kharif end/Rabi start
        start_date = f"{year}-09-01"
        end_date = f"{year}-11-30"
    elif 'winter' in season_name_lower: # Roughly Dec-Feb, often overlaps Rabi
        start_date = f"{year}-12-01"
        end_date = f"{year + 1}-02-28" # Handle leap year if needed, but GEE filterDate is inclusive
        # Basic leap year check for end date Feb 29th
        try:
            if (year + 1) % 4 == 0 and ((year + 1) % 100 != 0 or (year + 1) % 400 == 0):
                 end_date = f"{year + 1}-02-29"
        except: pass # Ignore potential date issues for now
    else:
        # Default or warning for unknown seasons
        # print(f"Warning: Unknown season '{season_name}'. Using whole year.")
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    return start_date, end_date

# --- 5. Define GEE Calculation Functions (Modified for start/end dates) ---
# (Includes elevation cache and get_value_safely as before)
elevation_cache = {}
# --- get_value_safely function (same as before) ---
def get_value_safely(data_dict, key, variable_name):
    # ... (identical to previous version) ...
    if data_dict is None: return None
    try:
        value = data_dict.get(key)
        if value is None: return None
        # Convert numpy types if necessary (though csv writer might handle them)
        if hasattr(value, 'item'): value = value.item() # General way to handle numpy types
        return value
    except Exception as e:
        print(f"--> Error processing {variable_name} result: {e}")
        return None


# --- GEE Functions modified for start_date, end_date ---
def calculate_mean_lst_season(aoi_ee, start_date, end_date):
    # ... (LST calculation logic using start_date, end_date in filterDate) ...
    try:
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterBounds(aoi_ee) \
            .filterDate(start_date, end_date) \
            .select(['LST_Day_1km', 'QC_Day'])

        def scale_lst(image):
            qa = image.select('QC_Day')
            cloud_mask = qa.bitwiseAnd(1).eq(0).And(qa.bitwiseAnd(2).eq(0))
            lst_celsius = image.select('LST_Day_1km').multiply(0.02).subtract(273.15)
            return lst_celsius.updateMask(cloud_mask).rename('LST_Celsius').copyProperties(image, ["system:time_start"])

        lst_celsius_collection = lst_collection.map(scale_lst).select('LST_Celsius')
        mean_lst_image = lst_celsius_collection.mean()
        if not mean_lst_image.bandNames().size().getInfo(): return None

        mean_lst_region = mean_lst_image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi_ee, scale=1000, maxPixels=5e8 ).getInfo()
        return get_value_safely(mean_lst_region, 'LST_Celsius', f'Mean LST {start_date}-{end_date}')
    except Exception as e: # Catch GEE and other errors
        # print(f"--> Error LST {start_date}-{end_date}: {e}") # Reduce verbosity
        return None


def calculate_mean_precipitation_season(aoi_ee, start_date, end_date):
    # ... (Precipitation calculation logic using start_date, end_date) ...
    try:
        precip_collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterBounds(aoi_ee).filterDate(start_date, end_date).select('precipitation')
        if precip_collection.size().getInfo() == 0: return None

        mean_precip_image = precip_collection.mean()
        mean_precip_region = mean_precip_image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi_ee, scale=5000, maxPixels=5e8 ).getInfo()
        return get_value_safely(mean_precip_region, 'precipitation', f'Mean Precip {start_date}-{end_date}')
    except Exception as e:
        # print(f"--> Error Precip {start_date}-{end_date}: {e}")
        return None


def calculate_mean_solar_radiation_season(aoi_ee, start_date, end_date):
    # ... (Solar radiation calculation logic using start_date, end_date) ...
     try:
        solar_collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
            .filterBounds(aoi_ee).filterDate(start_date, end_date).select('surface_solar_radiation_downwards_hourly')
        if solar_collection.size().getInfo() == 0: return None

        def calculate_hourly_watts(image):
            watts_m2 = image.select('surface_solar_radiation_downwards_hourly').divide(3600)
            return watts_m2.rename('ssrd_W_m2').copyProperties(image, ["system:time_start"])

        solar_watts_collection = solar_collection.map(calculate_hourly_watts)
        mean_solar_image = solar_watts_collection.mean()
        mean_solar_region = mean_solar_image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi_ee, scale=10000, maxPixels=5e8 ).getInfo()
        return get_value_safely(mean_solar_region, 'ssrd_W_m2', f'Mean Solar {start_date}-{end_date}')
     except Exception as e:
        # print(f"--> Error Solar {start_date}-{end_date}: {e}")
        return None


# --- Elevation function (remains the same, uses cache) ---
def calculate_mean_elevation(aoi_ee, district_key):
    # ... (identical to previous version, uses cache) ...
    if district_key in elevation_cache:
        return elevation_cache[district_key]
    try:
        elevation_image = ee.Image('USGS/SRTMGL1_003').select('elevation')
        mean_elevation_region = elevation_image.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi_ee, scale=90, maxPixels=5e8 ).getInfo()
        result = get_value_safely(mean_elevation_region, 'elevation', 'Mean Elevation')
        elevation_cache[district_key] = result
        return result
    except Exception as e:
        # print(f"--> Error Elevation {district_key}: {e}")
        elevation_cache[district_key] = None
        return None

# --- 6. Initialize Output CSV and Process Combinations ---
print(f"\nInitializing GEE Results CSV: {GEE_RESULTS_CSV_PATH}")

# Define the header for the results CSV
# Using original names + calculated fields for easier merging later
header = [
    'State_Name', 'District_Name', 'Crop_Year', 'Season', # Original keys
    'Mean_LST_C_Season', 'Mean_Elevation_m', 'Mean_Precip_mm_day_Season',
    'Mean_Solar_Rad_W_m2_Season', 'geometry_found'
]

# Check if file exists, if not write header
file_exists = os.path.isfile(GEE_RESULTS_CSV_PATH)
# OVERWRITE MODE: Always start fresh by uncommenting the next line
# file_exists = False # Uncomment this to always overwrite the results file

try:
    # Open in 'w' (write) mode only if file doesn't exist or we want to overwrite
    if not file_exists:
        with open(GEE_RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
        print("Created new results CSV with header.")
    else:
        print("Results CSV already exists. Will append results. (Delete file to start fresh)")
except Exception as e:
    print(f"Error initializing results CSV: {e}")
    exit()


print("\nStarting GEE calculations and appending results...")
processed_count = 0
combinations_since_last_save = 0
start_time = time.time()
batch_results = [] # Accumulate results for batch writing

# --- Iterate through UNIQUE combinations ---
for index, row in unique_combinations.iterrows():
    # Get keys using the cleaned names for lookup
    state_clean = row['clean_State_Name']
    district_clean = row['clean_District_Name']
    year = row['Crop_Year']
    season = row['Season'] # Use original season name from unique combo
    district_key = (state_clean, district_clean)

    processed_count += 1

    # --- Get Original Names for Output CSV ---
    # Find one matching row in the original df to get original names
    # This assumes clean names + year + season are unique enough for this purpose
    original_row = df_input[
        (df_input['clean_State_Name'] == state_clean) &
        (df_input['clean_District_Name'] == district_clean) &
        (df_input['Crop_Year'] == year) &
        (df_input['Season'] == season)
    ].iloc[0] # Get the first match
    original_state_name = original_row['State_Name']
    original_district_name = original_row['District_Name']


    if processed_count % 20 == 0 or processed_count == len(unique_combinations): # Print progress
       elapsed_time = time.time() - start_time
       print(f"Processing combination {processed_count}/{len(unique_combinations)}: {original_state_name}, {original_district_name}, {year}, {season} (Time: {elapsed_time:.1f}s)")

    # --- Find Geometry ---
    shapely_geometry = geometry_lookup.get(district_key)
    current_geom_found = False
    mean_lst_s, mean_elev, mean_precip_s, mean_solar_s = None, None, None, None

    if shapely_geometry:
        try:
            # Convert shapely geometry to GeoJSON dict, then to ee.Geometry
            geo_json_geometry = mapping(shapely_geometry)
            aoi_ee = ee.Geometry(geo_json_geometry)
            current_geom_found = True

            # --- Get Season Dates ---
            start_date, end_date = get_season_dates(year, season)

            # --- Perform GEE Calculations ---
            try:
                mean_lst_s = calculate_mean_lst_season(aoi_ee, start_date, end_date)
                mean_elev = calculate_mean_elevation(aoi_ee, district_key) # Static, uses cache
                mean_precip_s = calculate_mean_precipitation_season(aoi_ee, start_date, end_date)
                mean_solar_s = calculate_mean_solar_radiation_season(aoi_ee, start_date, end_date)
                print(f"LST: {mean_lst_s} | ELEV: {mean_elev} | PRECIPITATE: {mean_precip_s} | SOLAR: {mean_solar_s}")
            except Exception as gee_calc_e:
                 print(f"--> Unexpected Error during GEE calcs for {district_key} ({year}/{season}): {gee_calc_e}")
                 mean_elev = elevation_cache.get(district_key, None) # Attempt to get cached elevation

        except Exception as geom_e:
            print(f"--> Error processing geometry or creating ee.Geometry for {district_key}: {geom_e}")
            current_geom_found = False

    # --- Store Result for Batch ---
    result_dict = {
        'State_Name': original_state_name, # Use original names as keys for merging
        'District_Name': original_district_name,
        'Crop_Year': year,
        'Season': season,
        'Mean_LST_C_Season': mean_lst_s,
        'Mean_Elevation_m': mean_elev,
        'Mean_Precip_mm_day_Season': mean_precip_s,
        'Mean_Solar_Rad_W_m2_Season': mean_solar_s,
        'geometry_found': current_geom_found
    }
    batch_results.append(result_dict)
    combinations_since_last_save += 1

    # --- Append Batch to CSV Periodically ---
    if combinations_since_last_save >= SAVE_EVERY_N_COMBINATIONS or processed_count == len(unique_combinations):
        if batch_results: # Only write if there's something in the batch
            print(f"--- Appending {len(batch_results)} results to {GEE_RESULTS_CSV_PATH} ({processed_count}/{len(unique_combinations)} processed) ---")
            try:
                # Open in append mode ('a')
                with open(GEE_RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                    # Use DictWriter to handle mapping dict keys to columns easily
                    writer = csv.DictWriter(csvfile, fieldnames=header)
                    # If the file didn't exist initially, we wrote the header.
                    # If it did exist, we assume header is already there.
                    # Safety: Check file size - if empty after opening 'a', write header? More complex.
                    # For simplicity, we assume header exists if file exists.
                    writer.writerows(batch_results)

                batch_results = [] # Clear the batch
                combinations_since_last_save = 0 # Reset counter
            except Exception as e:
                print(f"--- ERROR appending batch to {GEE_RESULTS_CSV_PATH}: {e} ---")
                # Decide if you want to stop or continue if saving fails
        else:
             print(f"--- Batch empty, skipping save ({processed_count}/{len(unique_combinations)} processed) ---")


# --- 7. Final Summary ---
total_time = time.time() - start_time
print(f"\nFinished GEE calculations for {len(unique_combinations)} unique combinations in {total_time:.2f} seconds.")
print(f"GEE results saved incrementally to {GEE_RESULTS_CSV_PATH}")

# Clean up extracted GeoJSON directory (optional)
# ... (cleanup code from previous script) ...

print("\nStage 1 Script finished.")
