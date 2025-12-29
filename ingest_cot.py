import os
import requests
import pandas as pd
import zipfile
import io
from datetime import datetime
import logging
import config

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("COT_Ingest")

# Constants
COT_BASE_URL = "https://www.cftc.gov/files/dea/history/fut_fin_txt_{year}.zip"
START_YEAR = 2015
CURRENT_YEAR = datetime.now().year

# Mapping: Friendly Name -> CFTC Contract Name (Partial Match)
CONTRACT_MAP = {
    'cot_es': 'E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE',
    'cot_zn': '10-YEAR U.S. TREASURY NOTES - CHICAGO BOARD OF TRADE',
    'cot_6e': 'EURO FX - CHICAGO MERCANTILE EXCHANGE',
    'cot_btc': 'BITCOIN - CHICAGO MERCANTILE EXCHANGE'
}

# CFTC Columns (Based on "Code-098385-Format-Input.txt" standard)
# We only care about a few:
# Market_and_Exchange_Names (Field 1)
# Report_Date_as_MM_DD_YYYY (Field 3)
# NonComm_Positions_Long_All (Field 8)
# NonComm_Positions_Short_All (Field 9)
# Change_in_NonComm_Long_All (Field 10)
# Change_in_NonComm_Short_All (Field 11)
# Pct_of_OI_NonComm_Long_All (Field 19)
# Pct_of_OI_NonComm_Short_All (Field 20)
# Traders_NonComm_Long_All (Field 23)
# Traders_NonComm_Short_All (Field 24)

COL_NAMES = [
    "Market_and_Exchange_Names", "As_of_Date_In_Form_YYMMDD", "Report_Date_as_MM_DD_YYYY", 
    "CFTC_Contract_Market_Code", "CFTC_Market_Code", "CFTC_Region_Code", "CFTC_Commodity_Code", 
    "Open_Interest_All", 
    "NonComm_Positions_Long_All", "NonComm_Positions_Short_All", "NonComm_Positions_Spreading_All",
    "Comm_Positions_Long_All", "Comm_Positions_Short_All",
    "Tot_Rept_Positions_Long_All", "Tot_Rept_Positions_Short_All",
    "NonRept_Positions_Long_All", "NonRept_Positions_Short_All"
]

def fetch_year(year):
    url = COT_BASE_URL.format(year=year)
    logger.info(f"Downloading {url}...")
    try:
        response = requests.get(url, verify=False) # verify=False because CFTC certs sometimes have issues
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # There should be one .txt file inside
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                # CFTC files are comma delimited, but often have issues. 
                # We'll try standard read_csv
                df = pd.read_csv(f, usecols=range(len(COL_NAMES)), names=COL_NAMES, header=0, low_memory=False)
                return df
                
    except Exception as e:
        logger.error(f"Failed to fetch {year}: {e}")
        return None

def process_cot_data():
    # Suppress SSL warnings for CFTC (Government certs are often flaky)
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    all_data = []
    
    for year in range(START_YEAR, CURRENT_YEAR + 1):
        df = fetch_year(year)
        if df is not None:
            all_data.append(df)
            
    if not all_data:
        logger.warning("No COT data fetched.")
        return

    logger.info("Merging years...")
    full_df = pd.concat(all_data, ignore_index=True)
    
    # Clean Date
    full_df['date'] = pd.to_datetime(full_df['Report_Date_as_MM_DD_YYYY'])
    full_df = full_df.sort_values('date')
    
    # Process each tracked contract
    final_dfs = []
    
    for friendly_name, contract_name in CONTRACT_MAP.items():
        logger.info(f"Processing {friendly_name} ({contract_name})...")
        
        # Exact match is risky due to whitespace, so we strip
        mask = full_df['Market_and_Exchange_Names'].str.strip() == contract_name
        contract_df = full_df[mask].copy()
        
        if contract_df.empty:
            logger.warning(f"No data found for {contract_name}")
            continue
            
        # Calculate Net Positioning
        # Net Non-Commercial = Long - Short
        contract_df['long'] = pd.to_numeric(contract_df['NonComm_Positions_Long_All'], errors='coerce').fillna(0)
        contract_df['short'] = pd.to_numeric(contract_df['NonComm_Positions_Short_All'], errors='coerce').fillna(0)
        contract_df['oi'] = pd.to_numeric(contract_df['Open_Interest_All'], errors='coerce').fillna(0)
        
        contract_df['net_pos'] = contract_df['long'] - contract_df['short']
        
        # Calculate Sentiment
        # Sentiment = (Long - Short) / (Long + Short)
        contract_df['sentiment'] = contract_df['net_pos'] / (contract_df['long'] + contract_df['short']).replace(0, 1)
        
        # Rename and Select
        out_df = contract_df[['date', 'net_pos', 'sentiment', 'oi']].rename(columns={
            'net_pos': f'{friendly_name}_net',
            'sentiment': f'{friendly_name}_sentiment',
            'oi': f'{friendly_name}_oi'
        })
        
        final_dfs.append(out_df.set_index('date'))
        
    if not final_dfs:
        return
        
    # Merge all contracts
    logger.info("Merging contract data...")
    merged_df = pd.concat(final_dfs, axis=1).sort_index()
    
    # Reset index for saving
    merged_df = merged_df.reset_index()
    
    # Save
    output_path = os.path.join(config.DIRS['DATA_DIR'], "cot_weekly.parquet")
    merged_df.to_parquet(output_path, index=False)
    logger.info(f"Saved COT data to {output_path} ({len(merged_df)} rows)")

if __name__ == "__main__":
    # Suppress SSL warnings for CFTC
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    process_cot_data()
