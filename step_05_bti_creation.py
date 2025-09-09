from datetime import datetime
import json
import os
from google.cloud import bigquery
import pandas as pd
import numpy as np
from typing import Dict

def log_message(script_name: str, message: str) -> None:
    """Log message to file and console with timestamp."""
    log_entry = f"{datetime.now()} - {message}\n"
    log_file = f"{os.path.splitext(script_name)[0]}_logs.txt"
    with open(log_file, 'a') as f:
        f.write(log_entry)
    print(message)

def clean_logs(script_name: str) -> None:
    """Clean log file at the start of orchestration."""
    log_file = f"{os.path.splitext(script_name)[0]}_logs.txt"
    if os.path.exists(log_file):
        os.remove(log_file)
        
def load_and_join_data(client: bigquery.Client, dataset: str) -> pd.DataFrame:
    """Load and join benchmark and guidelines data."""
    log_message(__file__, "Loading benchmark data")
    benchmark_query = f"""
    SELECT * FROM `{dataset}.t_main_data_source_benchmark`
    """
    benchmark_df = client.query(benchmark_query).to_dataframe()
    
    # Add channel_segment to benchmark data
    log_message(__file__, "Adding channel segment mapping to benchmark data")
    benchmark_df['channel_segment'] = benchmark_df['channel_code'].apply(map_channel_segment)
    
    log_message(__file__, "Loading guidelines data")
    guidelines_query = f"""
    SELECT * FROM `{dataset}.t_guidelines`
    """
    guidelines_df = client.query(guidelines_query).to_dataframe()
    
    # Ensure year is int in benchmark_df data
    benchmark_df['year'] = benchmark_df['year'].astype(int)

    log_message(__file__, "Inner joining benchmark and guidelines data")
    joined_df = benchmark_df.merge(
        guidelines_df[['scenario', 'year', 'product_code', 'channel_segment', 'Min_price', 'Max_price', 'Floor_price']],
        on=['scenario', 'year', 'product_code', 'channel_segment'],
        how='inner'
    )
    
    return joined_df

def load_risk_data(client: bigquery.Client, dataset: str) -> pd.DataFrame:
    """Load and join risk exposure data."""
    log_message(__file__, "Loading risk exposure data")
    risk_query = f"SELECT * FROM `{dataset}.t_main_risk_exposure`"
    risk_df = client.query(risk_query).to_dataframe()
    
    # Add channel_segment to risk data
    log_message(__file__, "Adding channel segment mapping to risk data")
    risk_df['channel_segment'] = risk_df['channel_code'].apply(map_channel_segment)
    
    # Ensure year is int in risk_df data
    risk_df['year'] = risk_df['year'].astype(int)

    return risk_df

def map_channel_segment(channel_code: str) -> str:
    """Map channel_code to channel_segment."""
    return 'Pure Player' if channel_code == 'CH011' else 'O+O'

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns according to BTI specifications."""
    rename_mapping = {
        'year': 'Year',
        'period': 'Period',
        'scenario': 'Scenario',
        'multidivision_cluster': 'Geography_Hub',
        'customer': 'Cust',
        'customer_segment': 'Cust_Channel_Reclassification',
        'product': 'Product_BI_Central',
        'product_code': 'Compass_code',
        'axis': 'Categ_Axis',
        'sub_axis': 'Categ_New_Category',
        'metier': 'Categ_Metier',
        'brand': 'Categ_Brand',
        'signature': 'Categ_Signature',
        'channel_segment': 'Channel_Segment',  
        'R00100AA_Invoiced_units_Impact': 'Invoiced_units',
        '5N_SALES_Impact': '5N_sales',
        '5N_Price_Impact': 'Conso_Net_Price_ECRV'
    }
    
    return df.rename(columns=rename_mapping)

def calculate_bti_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate BTI metrics based on price thresholds."""
    log_message(__file__, "Calculating BTI metrics")
    
    # BTI Floor Price
    df['BTI_FP'] = df.apply(
        lambda x: x['Invoiced_units'] * x['Conso_Net_Price_ECRV']
        if x['Conso_Net_Price_ECRV'] < x['Floor_price']
        else None,
        axis=1
    )
    
    # BTI Corridor
    df['BTI_Corridor'] = df.apply(
        lambda x: x['Invoiced_units'] * x['Conso_Net_Price_ECRV']
        if x['Conso_Net_Price_ECRV'] < x['Min_price']
        else None,
        axis=1
    )
    
    # Business Above Corridor Max
    df['Business_Above_Corridor_Max'] = df.apply(
        lambda x: x['Invoiced_units'] * x['Conso_Net_Price_ECRV']
        if x['Conso_Net_Price_ECRV'] > x['Max_price']
        else None,
        axis=1
    )
    
    # Additional Valorization Floor Price
    df['Add_valorization_FP'] = df.apply(
        lambda x: (x['Floor_price'] - x['Conso_Net_Price_ECRV']) * x['Invoiced_units']
        if x['Conso_Net_Price_ECRV'] < x['Floor_price']
        else None,
        axis=1
    )
    
    # Additional Valorization Corridor
    df['Add_valorization_Corridor'] = df.apply(
        lambda x: (x['Min_price'] - x['Conso_Net_Price_ECRV']) * x['Invoiced_units']
        if x['Conso_Net_Price_ECRV'] < x['Min_price']
        else None,
        axis=1
    )
    
    # Value Generated Corridor Max
    df['Value_generated_Corridor_Max'] = df.apply(
        lambda x: (x['Conso_Net_Price_ECRV'] - x['Max_price']) * x['Invoiced_units']
        if x['Conso_Net_Price_ECRV'] > x['Max_price']
        else None,
        axis=1
    )
    
    return df

def calculate_risk_exposure(df: pd.DataFrame, min_invoiced_units: int = 50, price_threshold: float = 0.0) -> pd.DataFrame:
    """Calculate risk exposure metrics by scenario."""
    log_message(__file__, "Starting risk exposure calculations")
    
    # Load external data
    purchasing_groups_df = pd.read_csv('src/data/purchasing_groups_crosscount_np_alignment.csv', sep=';')
    df['retailer_scope'] = df['purchasing_group_custom'].isin(purchasing_groups_df['purchasing_group_custom']).astype(int)
    
    excluded_customers_df = pd.read_csv('src/data/risk_exposure_customers_exclusion.csv', sep=';')
    excluded_customers = set(excluded_customers_df['customer_code'])
    
    # Process each scenario separately
    results = []
    for scenario in df['Scenario'].unique():
        scenario_df = df[df['Scenario'] == scenario].copy()
        
        # Calculate average prices for this scenario
        avg_mask = (
            (scenario_df['Invoiced_units'] > 0) & 
            (scenario_df['5N_sales'] > 0) & 
            (scenario_df['SK_product_scope'] == 1)
        )
        avg_prices = scenario_df[avg_mask].groupby('Compass_code').agg({
            '5N_sales': 'sum',
            'Invoiced_units': 'sum'
        })
        avg_prices['Avg_5N'] = avg_prices['5N_sales'] / avg_prices['Invoiced_units']

        # Create valid rows cache for this scenario
        valid_base = scenario_df.loc[
            (scenario_df['retailer_scope'] == 1) & 
            (scenario_df['SK_product_scope'] == 1) & 
            (scenario_df['Invoiced_units'] > min_invoiced_units),
            ['Compass_code', 'Year', 'Geography_Hub', 'Conso_Net_Price_ECRV']
        ]

        # Pre-compute minimum prices cache for this scenario
        min_prices_cache = {}
        for code in scenario_df['Compass_code'].unique():
            for year in scenario_df[scenario_df['Compass_code'] == code]['Year'].unique():
                for hub in scenario_df[scenario_df['Compass_code'] == code]['Geography_Hub'].unique():
                    rows = valid_base[
                        (valid_base['Compass_code'] == code) & 
                        (valid_base['Year'] == year) & 
                        (valid_base['Geography_Hub'] != hub)
                    ]
                    if len(rows) > 0:
                        min_prices_cache[(code, year, hub)] = rows['Conso_Net_Price_ECRV'].min()
        
        def get_min_price(row):
            try:
                try:
                    if pd.notna(row['Avg_5N']) and row['Conso_Net_Price_ECRV'] < (price_threshold * row['Avg_5N']):
                        return row['Conso_Net_Price_ECRV']
                except:
                    return row['Conso_Net_Price_ECRV']
                
                if row['customer_code'] in excluded_customers:
                    return row['Conso_Net_Price_ECRV']
                
                cache_key = (row['Compass_code'], row['Year'], row['Geography_Hub'])
                if cache_key in min_prices_cache:
                    return min_prices_cache[cache_key]
                    
                return row['Conso_Net_Price_ECRV']
                
            except Exception:
                return row['Conso_Net_Price_ECRV']
        
        scenario_df = scenario_df.merge(
            avg_prices[['Avg_5N']], 
            left_on='Compass_code',
            right_index=True,
            how='left'
        )
        
        scenario_df['MinCust_Average_5N_Price'] = scenario_df.apply(get_min_price, axis=1)
        
        scenario_df['Effective_Min_5N_Price'] = scenario_df.apply(
            lambda x: x['Conso_Net_Price_ECRV'] if (
                x['MinCust_Average_5N_Price'] > x['Conso_Net_Price_ECRV'] or 
                x['MinCust_Average_5N_Price'] == 0
            ) else x['MinCust_Average_5N_Price'],
            axis=1
        )
        scenario_df['Effective_Min_5N_Sales'] = scenario_df['Effective_Min_5N_Price'] * scenario_df['Invoiced_units']
        scenario_df['Business_Sales_at_Risk'] = scenario_df['5N_sales'] - scenario_df['Effective_Min_5N_Sales']

        def excel_format_key(x, code):
            formatted = f"{x:.13f}".replace(".", ",")
            if "," in formatted and len(formatted.split(",")[1]) == 12:  # Check if has decimal and 12 digits
                formatted += "0"
            return formatted + f"_{code}"

        scenario_df['KEY_WORST_MIN'] = np.where(
            scenario_df['Business_Sales_at_Risk'].isna(),
            np.nan,
            scenario_df.apply(lambda x: excel_format_key(x['Effective_Min_5N_Price'], x['Compass_code']), axis=1)
        )

        scenario_df['KEY_WORST_CURRENT_5N'] = np.where(
            scenario_df['Business_Sales_at_Risk'].isna(),
            np.nan,
            scenario_df.apply(lambda x: excel_format_key(x['Conso_Net_Price_ECRV'], x['Compass_code']), axis=1)
        )

        # Check problematic cases that would affect our mapping
        relevant_min_keys = set(scenario_df['KEY_WORST_MIN'].dropna())
        duplicates = scenario_df.groupby('KEY_WORST_CURRENT_5N')['Geography_Hub'].nunique()
        issues = duplicates[duplicates > 1]

        problematic = [
            f"Scenario: {scenario_df.loc[scenario_df['KEY_WORST_CURRENT_5N'] == key, 'Scenario'].iloc[0]} | "
            f"{key}: {set(scenario_df[scenario_df['KEY_WORST_CURRENT_5N'] == key]['Geography_Hub'])}"
            for key in issues.index
            if key in relevant_min_keys and not str(key).startswith("0,0000000000000_")
        ]

        if problematic:
            log_message(__file__, f"Problematic keys with multiple Geography_Hubs: {problematic}")
        
        # Create mapping excluding keys starting with "0,0000000000000_"
        geography_map = (scenario_df[scenario_df['KEY_WORST_CURRENT_5N'].notna()]
                .drop_duplicates('KEY_WORST_CURRENT_5N')
                .set_index('KEY_WORST_CURRENT_5N')['Geography_Hub']
                .to_dict())

        geography_map = {k:v for k,v in geography_map.items() if not k.startswith("0,0000000000000")}

        scenario_df['Geography_Hub_MinPrice'] = scenario_df['KEY_WORST_MIN'].map(geography_map)

        # Filter by SK_product_scope at end of each scenario
        scenario_df = scenario_df[scenario_df['SK_product_scope'] == 1]

        # Remove rows with negative or small Business_Sales_at_Risk
        scenario_df = scenario_df[
                    (scenario_df['Business_Sales_at_Risk'] >= 0) & 
                    (abs(scenario_df['Business_Sales_at_Risk']) >= 1e-3)
                    ]

        results.append(scenario_df)
 
    return pd.concat(results, ignore_index=True)

def write_to_bigquery(client: bigquery.Client, df: pd.DataFrame, table_id: str, is_cumulative: bool = False) -> None:
    """Write DataFrame to BigQuery table.
    
    Args:
        client: BigQuery client
        df: DataFrame to write
        table_id: Target table ID
        is_cumulative: If True, append data to existing table instead of replacing it
    """
    log_message(__file__, f"Writing results to {table_id}")
    
    schema = [
        bigquery.SchemaField(col, "STRING" if df[col].dtype == 'object' else "FLOAT64")
        for col in df.columns
    ]
    
    # Check if table exists
    try:
        client.get_table(table_id)
        table_exists = True
        log_message(__file__, f"Table {table_id} exists")
    except Exception:
        table_exists = False
        log_message(__file__, f"Table {table_id} does not exist")
    
    if not table_exists:
        # Create new table if it doesn't exist
        log_message(__file__, f"Creating new table {table_id}")
        table = bigquery.Table(table_id, schema=schema)
        client.create_table(table)
    elif not is_cumulative:
        # Delete and recreate table if it exists and we're not in cumulative mode
        log_message(__file__, f"Deleting table {table_id}")
        try:
            client.delete_table(table_id)
            log_message(__file__, "Table deleted successfully")
            
            # Create new table
            log_message(__file__, f"Creating new table {table_id}")
            table = bigquery.Table(table_id, schema=schema)
            client.create_table(table)
        except Exception as e:
            log_message(__file__, f"Error while recreating table: {str(e)}")
            raise
    
    # Configure job to append or replace based on is_cumulative
    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_APPEND if is_cumulative and table_exists
            else bigquery.WriteDisposition.WRITE_TRUNCATE
        )
    )
    
    log_message(__file__, f"Loading data with {'APPEND' if is_cumulative and table_exists else 'TRUNCATE'} disposition")
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    
    log_message(__file__, f"Successfully loaded {len(df)} rows into {table_id}")

def orchestrate_bti_creation(dataset: str) -> None:
    """Main orchestration function for BTI table creation."""
    clean_logs(__file__)
    try:
        client = bigquery.Client()
        
        log_message(__file__, "\nStarting BTI and risk exposure table creation")
        
        # BTI path
        # Load and join data
        joined_df = load_and_join_data(client, dataset)
        bti_df = rename_columns(joined_df)
        bti_df = calculate_bti_metrics(bti_df)

        # Risk exposure path
        risk_df = load_risk_data(client, dataset)
        risk_df = rename_columns(risk_df)
        risk_df = calculate_risk_exposure(risk_df)

        # Write results - Both tables are cumulative as they are used in the tool
        write_to_bigquery(client, bti_df, f"{dataset}.BTI_assessment_table", is_cumulative=True)
        write_to_bigquery(client, risk_df, f"{dataset}.Risk_exposure_table", is_cumulative=True)
        
        log_message(__file__, "BTI and risk exposure table creation completed successfully")
        
    except Exception as e:
        log_message(__file__, f"Error in BTI and risk exposure table creation: {str(e)}")
        raise
