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
        
def write_to_bigquery(client: bigquery.Client, df: pd.DataFrame, table_id: str, is_cumulative: bool = True) -> None:
    """Write DataFrame to BigQuery table.
    
    Args:
        client: BigQuery client
        df: DataFrame to write
        table_id: Target table ID
        is_cumulative: If True, append data to existing table instead of replacing it
    """
    try:
        log_message(__file__, f"Writing results to {table_id} (Cumulative: {is_cumulative})")
        
        # Create schema with proper handling of integer types
        schema = []
        for col in df.columns:
            if col == 'Year' or df[col].dtype == 'int64':
                field_type = "INT64"
            elif df[col].dtype == 'object':
                field_type = "STRING"
            else:
                field_type = "FLOAT64"
            schema.append(bigquery.SchemaField(col, field_type))
        
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
        
        # Configure job to append or replace based on is_cumulative
        job_config = bigquery.LoadJobConfig(
            write_disposition=(
                bigquery.WriteDisposition.WRITE_APPEND if is_cumulative and table_exists
                else bigquery.WriteDisposition.WRITE_TRUNCATE
            ),
            schema=schema,
        )
        
        write_mode = 'APPEND' if is_cumulative and table_exists else 'TRUNCATE'
        log_message(__file__, f"Attempting to write {len(df)} rows with {write_mode} disposition")
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()
        
        log_message(__file__, f"Successfully loaded {len(df)} rows into {table_id}")
        
    except Exception as e:
        log_message(__file__, f"Error writing to BigQuery: {str(e)}")
        raise

def create_cockpit_country(df: pd.DataFrame) -> pd.DataFrame:
    """Create cockpit country visualization by duplicating EU5 and MSC rows."""
    log_message(__file__, "Creating cockpit country visualization")
    
    # Create a copy of original data
    original_df = df.copy()
    
    # Create copies for EU5 and MSC rows
    eu5_df = df[df['region_classification'] == 'EU5'].copy()
    eu5_df['Geography_Hub'] = 'EU5'
    
    msc_df = df[df['region_classification'] == 'MSC'].copy()
    msc_df['Geography_Hub'] = 'MSC'
    
    # Concatenate original and aggregated rows
    cockpit_df = pd.concat([original_df, eu5_df, msc_df], ignore_index=True)
    
    log_message(__file__, f"Cockpit country: Original rows: {len(df)}, EU5 rows: {len(eu5_df)}, MSC rows: {len(msc_df)}, Final rows: {len(cockpit_df)}")
    
    return cockpit_df

def create_cockpit_category(df: pd.DataFrame) -> pd.DataFrame:
    """Create cockpit category visualization by adding additional rows with 'All' in Categ_New_Category."""
    log_message(__file__, "Creating cockpit category visualization")
    
    # Create a copy of original data
    original_df = df.copy()
    
    # Create a copy for the "All" rows
    all_df = df.copy()
    all_df['Categ_New_Category'] = 'All'
    
    # Concatenate original and "All" rows
    cockpit_df = pd.concat([original_df, all_df], ignore_index=True)
    
    log_message(__file__, f"Cockpit category: Original rows: {len(df)}, Final rows: {len(cockpit_df)}")
    
    return cockpit_df

def create_cockpit_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Create cockpit brand visualization by adding additional rows with 'All' in category columns."""
    log_message(__file__, "Creating cockpit brand visualization")
    
    # Create a copy of original data
    original_df = df.copy()
    
    # Create a copy for the "All" rows
    all_df = df.copy()
    replace_columns = ['Categ_Signature', 'Categ_Brand', 'Categ_New_Category']
    for col in replace_columns:
        all_df[col] = 'All'
    
    # Concatenate original and "All" rows
    cockpit_df = pd.concat([original_df, all_df], ignore_index=True)
    
    log_message(__file__, f"Cockpit brand: Original rows: {len(df)}, Final rows: {len(cockpit_df)}")
    
    return cockpit_df

def create_profit_pool(client: bigquery.Client, dataset: str, bti_df: pd.DataFrame, vat_path: str) -> pd.DataFrame:
    """
    Create profit pool table combining:
    - BTI assessment data
    - Sell-out data with VAT adjustments for actual prices
    - Theoretical prices using list prices and coefficients
    
    Args:
        client: BigQuery client
        dataset: Dataset name
        bti_df: BTI assessment DataFrame
        vat_path: Path to VAT countries CSV
        
    Returns:
        DataFrame with profit pool calculations including both actual and theoretical prices
    """
    log_message(__file__, "Creating profit pool table")
    
    # Load signature coefficients for theoretical prices
    log_message(__file__, "Loading signature coefficients")
    try:
        coeff_path = 'src/data/signature_coefficients.csv'
        coeff_df = pd.read_csv(coeff_path, sep=';')
        required_cols = ['signature_code', 'multidivision_cluster', 'Coefficient']
        missing_cols = [col for col in required_cols if col not in coeff_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in coefficients file: {missing_cols}")
            
        # Rename column to match BTI data
        coeff_df = coeff_df.rename(columns={'multidivision_cluster': 'Geography_Hub'})
        
    except Exception as e:
        log_message(__file__, f"Error loading coefficients: {str(e)}")
        raise
   
    # Load VAT data for actual prices
    log_message(__file__, "Loading VAT data")
    vat_df = pd.read_csv(vat_path, sep=';')

    # Load retailer mapping
    log_message(__file__, "Loading retailer mapping")
    try:
        retailer_map_df = pd.read_csv('src/data/sell_out_retailer_mapping.csv', sep=';')
        if 'Retailer' not in retailer_map_df.columns or 'Cust_Channel_Reclassification' not in retailer_map_df.columns:
            raise ValueError("Retailer mapping file must contain 'Retailer' and 'Cust_Channel_Reclassification' columns")
    except Exception as e:
        log_message(__file__, f"Error loading retailer mapping: {str(e)}")
        raise
   
    # Load sell-out data with corrected column mapping
    sellout_query = f"""
    SELECT
        Year,
        Geography_Hub,
        Axis as Categ_Axis,
        Signature as Categ_Signature,
        Brand as Categ_Brand,
        sub_axis as Categ_New_Category,
        Retailer,
        SUM(value_sales_euro) as total_value_sales,
        SUM(volume_units_sales) as total_unit_sales
    FROM `{dataset}.t_sellout`
    GROUP BY 
        Year,
        Geography_Hub,
        Axis,
        Signature,
        Brand,
        sub_axis,
        Retailer
    """
   
    log_message(__file__, "Loading sell-out data")
    sellout_df = client.query(sellout_query).to_dataframe()

    # Map retailers to customer codes
    log_message(__file__, "Mapping retailers to customers from sell_in")
    sellout_df = sellout_df.merge(
        retailer_map_df,
        on='Retailer',
        how='left'
    )
   
    # Join VAT data
    log_message(__file__, "Joining VAT data")
    sellout_df = sellout_df.merge(
        vat_df,
        on=['Year', 'Geography_Hub'],
        how='left'
    )
   
    # Calculate average unit price at different aggregation levels
    log_message(__file__, "Calculating VAT-adjusted average unit prices at different levels")
   
    # List of dimension combinations to calculate average prices
    dimension_groups = [
        ['Year', 'Geography_Hub', 'Categ_New_Category'],
        ['Year', 'Geography_Hub', 'Categ_Signature'],
        ['Year', 'Geography_Hub', 'Cust_Channel_Reclassification']
    ]
   
    # Calculate average prices for each dimension group
    price_dfs = []
    for dims in dimension_groups:
        agg_df = sellout_df.groupby(dims).agg({
            'total_value_sales': 'sum',
            'total_unit_sales': 'sum',
            'VAT_rate': 'first'  # Take first VAT value for the group
        }).reset_index()
       
       # Generate the column name for avg_unit_price based on the dimension group
        avg_price_column = f'avg_unit_price_{"_".join(dims[2:]).lower()}'

        # Calculate VAT-adjusted average unit price
        agg_df[f'avg_unit_price_{"_".join(dims[2:]).lower()}'] = (
            (agg_df['total_value_sales'] / (1 + agg_df['VAT_rate']/100)) / 
            agg_df['total_unit_sales'].replace(0, np.nan)
        ).fillna(0)

        # Rename the column for Cust_Channel_Reclassification only
        if 'avg_unit_price_cust_channel_reclassification' in agg_df.columns:
            agg_df.rename(columns={'avg_unit_price_cust_channel_reclassification': 'avg_unit_price_customer_code'}, inplace=True)

        # Keep only dimensions and the calculated price
        price_cols = dims + [avg_price_column if 'Cust_Channel_Reclassification' not in dims else 'avg_unit_price_customer_code']
        price_dfs.append(agg_df[price_cols])
   
    # Start with BTI data and add coefficients
    log_message(__file__, "Adding coefficients to BTI data")
    result_df = bti_df.merge(
        coeff_df,
        on=['signature_code', 'Geography_Hub'],
        how='left'
    )
    
    # Join VAT data with BTI data
    log_message(__file__, "Joining VAT data")
    result_df = result_df.merge(
        vat_df,
        on=['Year', 'Geography_Hub'],
        how='left'
    )

    # Calculate theoretical price
    log_message(__file__, "Calculating theoretical prices")
    result_df['avg_unit_price_theoretical'] = (
        result_df['R00100BA_Unit_list_price'] * result_df['Coefficient'] / 
        (1 + result_df['VAT_rate'] / 100)
    ).fillna(0)

    
    # Remove coefficient column
    result_df = result_df.drop('Coefficient', axis=1)
   
    # Add actual price calculations
    log_message(__file__, "Merging actual price calculations")
    for price_df in price_dfs:
        merge_cols = [col for col in price_df.columns if col.startswith('avg_unit_price')]
        result_df = result_df.merge(
            price_df,
            on=[col for col in price_df.columns if not col.startswith('avg_unit_price')],
            how='left'
        )
       
    log_message(__file__, f"Profit pool table created with {len(result_df)} rows")
    return result_df

def orchestrate_visuals_creation(dataset: str) -> None:
    """Main orchestration function for creating visualization tables."""
    clean_logs(__file__)
    try:
        client = bigquery.Client()
        
        log_message(__file__, "\nStarting visuals creation")
        
        # Load BTI assessment data
        bti_query = f"""
        SELECT * FROM `{dataset}.BTI_assessment_table`
        """
        bti_df = client.query(bti_query).to_dataframe()
        log_message(__file__, f"Loaded {len(bti_df)} rows from BTI assessment table")
        
        vat_path = 'src/data/vat_countries.csv'

        # Create cockpit country visualization (EU5 and MSC)
        cockpit_country_df = create_cockpit_country(bti_df)
        country_table = f"{dataset}.cockpit_country"
        write_to_bigquery(client, cockpit_country_df, country_table, is_cumulative=True)
        
        # Create cockpit category visualization
        cockpit_category_df = create_cockpit_category(bti_df)
        category_table = f"{dataset}.cockpit_category"
        write_to_bigquery(client, cockpit_category_df, category_table, is_cumulative=True)
        
        # Create cockpit brand visualization
        cockpit_brand_df = create_cockpit_brand(bti_df)
        brand_table = f"{dataset}.cockpit_brand"
        write_to_bigquery(client, cockpit_brand_df, brand_table, is_cumulative=True)
        
        # Create profit pool with both actual and theoretical prices
        log_message(__file__, "Creating profit pool visualizations")
        profit_pool_df = create_profit_pool(client, dataset, bti_df, vat_path)
        profit_pool_table = f"{dataset}.t_profit_pool"
        write_to_bigquery(client, profit_pool_df, profit_pool_table, is_cumulative=True)
        
        log_message(__file__, "Visuals creation completed successfully")
        
    except Exception as e:
        log_message(__file__, f"Error in visuals creation: {str(e)}")
        raise