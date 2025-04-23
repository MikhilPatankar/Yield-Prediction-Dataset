# calculate_ndvi_seasonal.py

import ee
import pandas as pd
import geopandas as gpd
import json
# import zipfile # No longer needed
import os
import time
import warnings
import csv
from datetime import datetime
from shapely.geometry import mapping # To convert shapely geometry

# Ignore warnings
warnings.filterwarnings('ignore', module='ee')
warnings.filterwarnings('ignore', category=FutureWarning) # Pandas future warnings
pd.options.mode.chained_assignment = None # Suppress SettingWithCopyWarning (use with caution)


# --- 0. Configuration ---
INPUT_CSV_PATH = 'crop_production.csv' # Path to your input crop production data
# --- Path to the actual GeoJSON file ---
GEOJSON_FILE_PATH = 'gadm41_IND_2.json'
# --- Output file for *only* the NDVI results ---
NDVI_RESULTS_CSV_PATH = 'gee_ndvi_seasonal_results.csv'
SAVE_EVERY_N_COMBINATIONS = 50 # Save results to CSV after every N combinations

# --- Date range for Landsat availability ---
# Using Landsat Collection 2, Tier 1, Surface Reflectance
LANDSAT_START_YEAR = 1984 # Landsat 5 start
LANDSAT_END_YEAR = datetime.now().year # Up to current year

print(f"--- Configuration ---")
print(f"Input Crop CSV: {INPUT_CSV_PATH}")
print(f"Input GeoJSON File: {GEOJSON_FILE_PATH}") # Updated message
print(f"Output NDVI Results CSV: {NDVI_RESULTS_CSV_PATH}")
print(f"Saving progress every: {SAVE_EVERY_N_COMBINATIONS} combinations")
print(f"Processing years from {LANDSAT_START_YEAR} up to {LANDSAT_END_YEAR}")
print(f"Note: Client-side threading is NOT used due to GEE request limits and .getInfo() blocking nature.")
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
print(f"\nLoading GeoJSON from {GEOJSON_FILE_PATH}...")
try:
    # --- Directly load the GeoJSON file using geopandas ---
    gdf = gpd.read_file(GEOJSON_FILE_PATH)
    print(f"Loaded {len(gdf)} features from GeoJSON.")

    # --- Clean names and create lookup ---
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

# Handle file not found specifically
except FileNotFoundError:
    print(f"ERROR: GeoJSON file not found at '{GEOJSON_FILE_PATH}'. Please ensure the file exists.")
    exit()
# Handle other potential errors during file reading or processing
except Exception as e:
    print(f"Error loading or processing GeoJSON file '{GEOJSON_FILE_PATH}': {e}")
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

    # Filter out years outside Landsat availability if necessary
    df_input_filtered_years = df_input[df_input['Crop_Year'] >= LANDSAT_START_YEAR].copy() # Use copy to avoid SettingWithCopyWarning later if needed

    unique_combinations = df_input_filtered_years[['clean_State_Name', 'clean_District_Name', 'Crop_Year', 'Season']].drop_duplicates().reset_index(drop=True)
    print(f"Found {len(unique_combinations)} unique combinations within Landsat range ({LANDSAT_START_YEAR}-present) to process.")

    # Keep df_input (unfiltered by year initially) for looking up original names
    # Ensure clean names exist on df_input as well
    if 'clean_State_Name' not in df_input.columns:
        df_input['clean_State_Name'] = df_input['State_Name'].apply(clean_name)
    if 'clean_District_Name' not in df_input.columns:
        df_input['clean_District_Name'] = df_input['District_Name'].apply(clean_name)
    if df_input['Season'].dtype == 'object':
         df_input['Season'] = df_input['Season'].str.strip()


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
    elif 'autumn' in season_name_lower: # Roughly Sept-Nov
        start_date = f"{year}-09-01"
        end_date = f"{year}-11-30"
    elif 'winter' in season_name_lower: # Roughly Dec-Feb
        start_date = f"{year}-12-01"
        end_y = year + 1
        end_date = f"{end_y}-02-28" # Handle leap year if needed
        # Basic leap year check for end date Feb 29th
        try: # Use try-except for safety if year is weird
            if end_y % 4 == 0 and (end_y % 100 != 0 or end_y % 400 == 0):
                 end_date = f"{end_y}-02-29"
        except: pass
    else:
        # Default or warning for unknown seasons
        # print(f"Warning: Unknown season '{season_name}'. Using whole year.")
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

    return start_date, end_date


# --- 5. Define Landsat Cloud Masking Function ---
# Cloud masking function for Landsat Collection 2 SR data
def mask_landsat_sr(image):
    try:
        # Bits 3 (Cloud Shadow), 4 (Snow), and 5 (Cloud) are the primary flags.
        qa_pixel = image.select('QA_PIXEL')
        cloud_shadow_bit = 1 << 3
        snow_bit = 1 << 4
        cloud_bit = 1 << 5
        # Pixels that are clear have these bits set to 0.
        mask = qa_pixel.bitwiseAnd(cloud_shadow_bit).eq(0) \
                    .And(qa_pixel.bitwiseAnd(snow_bit).eq(0)) \
                    .And(qa_pixel.bitwiseAnd(cloud_bit).eq(0))

        # Mask saturated pixels using QA_RADSAT if available
        if 'QA_RADSAT' in image.bandNames().getInfo():
             qa_radsat = image.select('QA_RADSAT')
             # Mask if any saturation bit is set (conservative)
             sat_mask = qa_radsat.eq(0)
             final_mask = mask.And(sat_mask)
        else:
             final_mask = mask # Older collections might not have QA_RADSAT

        # Return the image with the updated mask applied.
        return image.updateMask(final_mask)
    except Exception as e:
        # print(f"Error during cloud masking: {e}") # Optional debug print
        # Return original image if masking fails catastrophically for some reason
        return image


# --- 6. Define GEE NDVI Calculation Function ---
# --- get_value_safely function ---
def get_value_safely(data_dict, key, variable_name):
    if data_dict is None: return None
    try:
        value = data_dict.get(key)
        if value is None: return None
        # Convert numpy types to standard python types for easier saving/compatibility
        if hasattr(value, 'item'): value = value.item()
        return value
    except Exception as e:
        print(f"--> Error processing {variable_name} result: {e}")
        return None


def calculate_mean_ndvi_season(aoi_ee, start_date, end_date, year):
    """Calculates mean NDVI for a given AOI, date range, and year, selecting appropriate Landsat."""
    try:
        # Select Landsat Collection based on year
        collection_id = None
        if year < 1999: # Use Landsat 5
            collection_id = 'LANDSAT/LT05/C02/T1_L2'
            nir_band = 'SR_B4'; red_band = 'SR_B3'; scale = 30
        elif 1999 <= year < 2013: # Use Landsat 7
            collection_id = 'LANDSAT/LE07/C02/T1_L2'
            nir_band = 'SR_B4'; red_band = 'SR_B3'; scale = 30
        elif 2013 <= year < 2022: # Use Landsat 8
             collection_id = 'LANDSAT/LC08/C02/T1_L2'
             nir_band = 'SR_B5'; red_band = 'SR_B4'; scale = 30
        elif year >= 2022: # Use Landsat 9
            collection_id = 'LANDSAT/LC09/C02/T1_L2'
            nir_band = 'SR_B5'; red_band = 'SR_B4'; scale = 30

        if collection_id is None:
             print(f"Warning: Year {year} outside expected Landsat 5/7/8/9 range. Skipping NDVI.")
             return None

        # Load, filter, and mask collection
        landsat_col = ee.ImageCollection(collection_id) \
            .filterBounds(aoi_ee) \
            .filterDate(start_date, end_date) \
            .map(mask_landsat_sr) # Apply cloud mask

        # Function to calculate NDVI for each image AFTER applying scale/offset
        def add_ndvi(image):
            # Apply scale factor for SR bands before NDVI - VERY IMPORTANT
            # Optical bands (SR_B*) scale factor is 0.0000275, offset is -0.2
            nir = image.select(nir_band).multiply(0.0000275).add(-0.2)
            red = image.select(red_band).multiply(0.0000275).add(-0.2)
            # Calculate NDVI from scaled reflectance
            ndvi_scaled = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
            # Return only NDVI band, copy time properties for potential median calc later if needed
            return ndvi_scaled.copyProperties(image, ["system:time_start"])

        # Calculate NDVI for the collection
        ndvi_col = landsat_col.map(add_ndvi)

        # Calculate mean NDVI image for the season
        # Using median() might be more robust to outliers than mean() for NDVI
        # median_ndvi_image = ndvi_col.median() # Option 1: Median
        mean_ndvi_image = ndvi_col.mean()      # Option 2: Mean (as requested)

        # Check if the result image is valid (has bands) before reducing
        band_names = mean_ndvi_image.bandNames().getInfo()
        if not band_names:
            # print(f"--> No valid NDVI pixels found for {start_date} to {end_date}.")
            return None # No valid pixels after masking/calculation

        # Reduce to get mean NDVI value for the AOI
        mean_ndvi_region = mean_ndvi_image.reduceRegion(
            reducer=ee.Reducer.mean(), # Or ee.Reducer.median()
            geometry=aoi_ee,
            scale=scale,  # Native Landsat resolution
            maxPixels=5e8, # Adjust as needed, GEE might optimize this
            bestEffort=True # Allow GEE to potentially use a coarser scale if needed
        ).getInfo() # Use getInfo() to fetch the result

        # Safely extract the NDVI value
        return get_value_safely(mean_ndvi_region, 'NDVI', f'Mean NDVI {start_date}-{end_date}')

    except ee.EEException as e:
        # print(f"--> GEE Error calculating NDVI for {year}/{start_date}: {e}") # Reduce verbosity
        return None
    except Exception as e:
        # Print non-GEE errors which might indicate logic issues
        print(f"--> Non-GEE Error calculating NDVI for {year}/{start_date}: {e}")
        return None


# --- 7. Initialize Output CSV and Process Combinations ---
print(f"\nInitializing NDVI Results CSV: {NDVI_RESULTS_CSV_PATH}")
header = [
    'State_Name', 'District_Name', 'Crop_Year', 'Season', # Original keys
    'Mean_NDVI_Season', 'geometry_found' # NDVI specific result
]
file_exists = os.path.isfile(NDVI_RESULTS_CSV_PATH)
# OVERWRITE MODE: Uncomment to always start fresh
# file_exists = False

try:
    # Open in 'w' (write) mode only if file doesn't exist or we intend to overwrite
    if not file_exists:
        with open(NDVI_RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
        print("Created new NDVI results CSV with header.")
    else:
        print("NDVI results CSV already exists. Will append results. (Delete file to start fresh)")
except Exception as e:
    print(f"Error initializing NDVI results CSV: {e}")
    exit()


print("\nStarting GEE NDVI calculations and appending results...")
processed_count = 0
combinations_since_last_save = 0
start_time = time.time()
batch_results = [] # Accumulate results for batch writing

# --- Iterate through UNIQUE combinations ---
for index, row in unique_combinations.iterrows():
    # Keys for lookup
    state_clean = row['clean_State_Name']
    district_clean = row['clean_District_Name']
    year = row['Crop_Year']
    season = row['Season'] # Use original season name from unique combo
    district_key = (state_clean, district_clean)

    processed_count += 1

    # --- Get Original Names for Output CSV (Corrected Block) ---
    try:
        # Find one matching row in the original df_input to get original names
        original_row = df_input[
            (df_input['clean_State_Name'] == state_clean) &
            (df_input['clean_District_Name'] == district_clean) &
            (df_input['Crop_Year'] == year) &
            (df_input['Season'] == season)
        ].iloc[0] # Get the first match
        original_state_name = original_row['State_Name']
        original_district_name = original_row['District_Name']
    except IndexError:
        # Handle case where the unique combo might somehow not be in df_input
        # This might happen if df_input had NAs dropped but unique_combinations didn't perfectly align
        print(f"Warning: Could not find original names for keys: {state_clean}, {district_clean}, {year}, {season}. Using cleaned names.")
        original_state_name = state_clean # Assign fallback value
        original_district_name = district_clean # Assign fallback value
    # --- End of Corrected Block ---

    if processed_count % 20 == 0 or processed_count == len(unique_combinations): # Print progress
       elapsed_time = time.time() - start_time
       print(f"Processing combination {processed_count}/{len(unique_combinations)}: {original_state_name}, {original_district_name}, {year}, {season} (Time: {elapsed_time:.1f}s)")

    # --- Find Geometry ---
    shapely_geometry = geometry_lookup.get(district_key)
    current_geom_found = False
    mean_ndvi_s = None # Initialize result for this iteration

    if shapely_geometry:
        try:
            # Convert shapely geometry to GeoJSON dict, then to ee.Geometry
            geo_json_geometry = mapping(shapely_geometry)
            aoi_ee = ee.Geometry(geo_json_geometry)
            current_geom_found = True

            # --- Get Season Dates ---
            start_date, end_date = get_season_dates(year, season)

            # --- Perform GEE NDVI Calculation ---
            # Wrap the specific GEE call in its own try-except
            try:
                mean_ndvi_s = calculate_mean_ndvi_season(aoi_ee, start_date, end_date, year)
                print(f"NDVI: {mean_ndvi_s}")
            except Exception as gee_calc_e:
                 # Catch unexpected errors during the calculation function itself
                 print(f"--> Unexpected Error during NDVI calc function for {district_key} ({year}/{season}): {gee_calc_e}")
                 mean_ndvi_s = None # Ensure it's None on error

        except Exception as geom_e:
            # Catch errors during geometry conversion or date calculation
            print(f"--> Error processing geometry or getting dates for {district_key}: {geom_e}")
            current_geom_found = False # Mark geometry as problematic

    # --- Store Result for Batch ---
    result_dict = {
        'State_Name': original_state_name,
        'District_Name': original_district_name,
        'Crop_Year': year,
        'Season': season,
        'Mean_NDVI_Season': f"{mean_ndvi_s:.4f}" if mean_ndvi_s is not None else None, # Format NDVI to 4 decimals
        'geometry_found': current_geom_found
    }
    batch_results.append(result_dict)
    combinations_since_last_save += 1

    # --- Append Batch to CSV Periodically ---
    if combinations_since_last_save >= SAVE_EVERY_N_COMBINATIONS or processed_count == len(unique_combinations):
        if batch_results: # Only write if there's something in the batch
            print(f"--- Appending {len(batch_results)} NDVI results to {NDVI_RESULTS_CSV_PATH} ({processed_count}/{len(unique_combinations)} processed) ---")
            try:
                # Open in append mode ('a')
                with open(NDVI_RESULTS_CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
                    # Use DictWriter to handle mapping dict keys to columns easily
                    writer = csv.DictWriter(csvfile, fieldnames=header)
                    # If the file didn't exist initially (file_exists=False), we wrote the header.
                    # If it did exist, we assume header is already there when appending.
                    writer.writerows(batch_results)

                batch_results = [] # Clear the batch
                combinations_since_last_save = 0 # Reset counter
            except Exception as e:
                print(f"--- ERROR appending NDVI batch to {NDVI_RESULTS_CSV_PATH}: {e} ---")
                # Consider maybe pausing or stopping if saving fails repeatedly
        else:
             # This case should ideally not happen if the outer condition is met, but safe to have
             print(f"--- NDVI Batch empty, skipping save ({processed_count}/{len(unique_combinations)} processed) ---")


# --- 8. Final Summary ---
total_time = time.time() - start_time
print(f"\nFinished GEE NDVI calculations for {len(unique_combinations)} unique combinations in {total_time:.2f} seconds.")
print(f"NDVI results saved incrementally to {NDVI_RESULTS_CSV_PATH}")

# Clean up extracted GeoJSON directory (optional - remove if not needed)
try:
    import shutil
    if 'extract_path' in locals() and os.path.exists(extract_path): # Check if path was defined
        shutil.rmtree(extract_path)
        # print(f"Cleaned up temporary directory: {extract_path}")
except Exception as e:
    # print(f"Could not clean up temporary directory {extract_path}: {e}")
    pass


print("\nNDVI Calculation Script finished.")
