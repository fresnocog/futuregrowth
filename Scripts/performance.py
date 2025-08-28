# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 13:17:42 2025

@author: joshi
"""

from datetime import datetime
import pandas as pd
import os
import sys
import logging
import openmatrix as omx
import numpy.ma as ma
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s', # format='%(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), 'Setup/Logs/performance.log'), mode='w')  # Overwrites log file each run
    ]
)
logger = logging.getLogger(__name__)

class PerformanceIndicatorCalculator:
    VISION_YEAR = 2035
    NO_DEV = 9999
    
    def __init__(self, param_file: str, target_year: int=None):
        """Initialize with parameters file and optional target year."""
        self.parameters = pd.read_csv(param_file)
        self.parameters.columns = ['Key', 'Value', 'Notes']
        self.working_dir = self._get_param('WORKING_DIR').strip()
        self.data_dir = os.path.join(self.working_dir, 'Data')
        self.abm_dir = os.path.join(self.working_dir, 'Data', 'ABM')
        self.output_dir = os.path.join(self.working_dir, 'Setup', 'Data')
        self.base_year = int(self._get_param('baseYear').strip())
        self.target_year = target_year if target_year else int(self._get_param('targetYear').strip())
        self.cube_p = float(self._get_param('Cube_P').strip())

    def _get_param(self, key: str) -> str:
        """Helper method to get parameter value."""
        return self.parameters[self.parameters.Key == key]['Value'].item()
    
    def load_data(self) -> tuple:
        """Load base MAZ, new MAZ, and development table data."""
        logger.info("Loading input data")
        try:
            base_maz = pd.read_csv(os.path.join(self.data_dir, "Base_MAZ_2023.csv"))
            maz_new = pd.read_csv(os.path.join(self.output_dir, "maz_new.csv"))
            dev_table = pd.read_csv(os.path.join(self.output_dir, "devtable.csv"))
            logger.debug(f"Loaded base_maz: {base_maz.shape}, maz_new: {maz_new.shape}, dev_table: {dev_table.shape}")
            return base_maz, maz_new, dev_table
        except Exception as e:
            logger.error(f"Error loading input data: {str(e)}")
            raise
    
    def calculate_performance_measures(self, base_maz: pd.DataFrame, maz_new: pd.DataFrame, dev_table: pd.DataFrame) -> dict:
        """Calculate performance indicators for land use allocation."""
        logger.info("Calculating performance measures")
        
        # Merge and calculate net growth
        maz_new = maz_new.merge(base_maz, how='left', on='MAZ')
        maz_new['HU_NET'] = maz_new['HU'] - maz_new['Base_HU']
        maz_new['EMP_NET'] = maz_new['EMP'] - maz_new['Base_EMP']
        hu_tot = maz_new['HU_NET'].sum()
        emp_tot = maz_new['EMP_NET'].sum()
        
        # Geographic area breakdown
        pm_geo_area = maz_new.groupby('GEO_AREA').agg({'HU_NET': 'sum', 'EMP_NET': 'sum'}).reset_index()
        
        # Filter development table for allocated parcels and merge with parcels data
        dev_table = dev_table.filter(items=['parcelid', 'DEV'])
        dev_table = dev_table[dev_table['DEV'] <= self.target_year]
        dev_table = dev_table.merge(pd.read_csv(os.path.join(self.output_dir, "parcels.csv")), how='left', on='parcelid')
        hu_tot_dev = (dev_table['HU_NET'] + dev_table['HU']).sum()
        emp_tot_dev = (dev_table['EMP_NET'] + dev_table['EMP']).sum()
        
        # Total acres developed
        acres_tot = (dev_table['Vacant'] * dev_table['ACRES'] * dev_table['PLANNED']).sum()
        acres_res = dev_table[dev_table['HU_NET'] > 0]['ACRES'].sum()
        
        # TOD, DT, and MF percentages
        pm_tod_hu = (dev_table['HU_NET'] * dev_table['TOD']).sum() / hu_tot if hu_tot > 0 else 0
        pm_dt_hu = (dev_table['HU_NET'] * dev_table['DT']).sum() / hu_tot if hu_tot > 0 else 0
        pm_mf = (dev_table['HU_NET'] * dev_table['HU_MF_P']).sum() / hu_tot if hu_tot > 0 else 0
        pm_tod_emp = (dev_table['EMP_NET'] * dev_table['TOD']).sum() / emp_tot if emp_tot > 0 else 0
        pm_dt_emp = (dev_table['EMP_NET'] * dev_table['DT']).sum() / emp_tot if emp_tot > 0 else 0
        
        # MU and redevelopment percentages
        total_growth = hu_tot + emp_tot
        pm_mu = (dev_table['MU'] * (dev_table['HU_NET'] + dev_table['EMP_NET'])).sum() / total_growth if total_growth > 0 else 0
        pm_redev = (dev_table['Developed'] * (dev_table['HU_NET'] + dev_table['EMP_NET'])).sum() / total_growth if total_growth > 0 else 0
        pm_geo_area['PERC'] = 100 * (pm_geo_area['HU_NET'] + pm_geo_area['EMP_NET']) / total_growth if total_growth > 0 else 0
        
        # Residential density
        pm_res_den = hu_tot_dev / acres_res if acres_res > 0 else 0
        
        # Farmland impact
        pm_impfarm = (dev_table['ACRES'] * dev_table['Vacant'] * (1 - dev_table['Incorporated']) * 
                      (dev_table['FMMP_P'] + dev_table['FMMP_S'] + dev_table['FMMP_U'])).sum()
        pm_farm = (dev_table['ACRES'] * dev_table['Vacant'] * 
                   (dev_table['FMMP_P'] + dev_table['FMMP_S'] + dev_table['FMMP_U'])).sum()
        
        # Fresno infill percentage
        dev_table_fresno = dev_table[dev_table['SOI'] == 'Fresno']
        dev_table_notfresno = dev_table[(dev_table['SOI'] != 'Fresno') & (dev_table['SOI'] != "Unincorporate")]
        hu_tot_fresno = dev_table_fresno['HU_NET'].sum()
        hu_tot_notfresno = dev_table_notfresno['HU_NET'].sum()
        pm_infill = dev_table_fresno[dev_table_fresno['Infill'] == 0]['HU_NET'].sum() / hu_tot_fresno if hu_tot_fresno > 0 else 0
        pm_infill_notfresno = dev_table_notfresno[dev_table_notfresno['Infill'] == 0]['HU_NET'].sum() / hu_tot_notfresno if hu_tot_notfresno > 0 else 0
        
        # Compile performance measures
        measures = {
            'HU_Tot': hu_tot,
            'EMP_Tot': emp_tot,
            'ACRES_Tot': acres_tot,
            'TOD_HU': pm_tod_hu,
            'TOD_EMP': pm_tod_emp,
            'DT_HU': pm_dt_hu,
            'DT_EMP': pm_dt_emp,
            'MU': pm_mu,
            'RES_DEN': pm_res_den,
            'MF': pm_mf,
            'FARM_AC': pm_farm,
            'FARM_IMP': pm_impfarm,
            'Redev': pm_redev,
            'Infill (Fresno)': pm_infill,
            'Infill (non-Fresno)': pm_infill_notfresno,
            'Geo_Area': pm_geo_area
        }
        return measures
    
    def calculate_new_indicators(self, base_maz: pd.DataFrame, maz_new: pd.DataFrame, dev_table: pd.DataFrame):

        # Employment Density

        dev_table = dev_table.filter(items=['parcelid', 'DEV'])
        dev_table = dev_table[dev_table['DEV'] <= self.target_year]
        dev_table = dev_table.merge(pd.read_csv(os.path.join(self.output_dir, "parcels.csv")), how='left', on='parcelid')

        emp_tot_dev = dev_table['EMP_NET'].sum()
        acres_emp= dev_table[dev_table['EMP_NET'] > 0]['ACRES'].sum()

        pm_emp_den = emp_tot_dev / acres_emp if acres_emp > 0 else 0

        # Job-housing Balance

        taz_skims = pd.DataFrame({'TAZ': range(1, 3001)}).merge(
            maz_new.groupby('TAZ').agg({'SOI': 'first', 'HU': 'sum', 'EMP': 'sum'}),
            how='left',
            on='TAZ'
        ).fillna({'SOI': '', 'HU': 0, 'EMP': 0})

        hu_array_taz = taz_skims['HU'].to_numpy()
        emp_array_taz = taz_skims['EMP'].to_numpy()

        # Use accessiblity
        taz_skims[['job_housing_gravity', 'job_housing_nongravity', 'job_housing_ratio']] = 0.0
        try:
            skim_file = os.path.join(self.abm_dir, f"FC{str(self.base_year)[-2:]}_BASE_SKM_PM_D1.omx")
            with omx.open_file(skim_file) as sov_skim_omx:
                        sov_skims = sov_skim_omx['TIME_1Veh']
                        
                        for i in taz_skims.index:
                            travel_times = sov_skims[i]

                            valid_mask = ma.masked_where(travel_times > 0, travel_times).mask
                            # gravity based
                            gravity_val = (valid_mask * emp_array_taz * hu_array_taz * np.exp(-0.049601 * travel_times)).sum()
                            # non- gravity based
                            nongravity_val = (valid_mask * emp_array_taz * hu_array_taz).sum()
                            # gravity-based ratio
                            gravity_ratio_val = (valid_mask * emp_array_taz * np.exp(-0.049601 * travel_times)).sum() / (valid_mask * hu_array_taz * np.exp(-0.049601 * travel_times)).sum()
                            # non-gravoty based ratio
                            nongravity_ratio_val = (valid_mask * emp_array_taz).sum() / (valid_mask * hu_array_taz).sum()

                            taz_skims.at[i, 'job_housing_gravity'] = gravity_val
                            taz_skims.at[i, 'job_housing_nongravity'] = nongravity_val
                            taz_skims.at[i, 'job_housing_ratio_gravity'] = gravity_ratio_val
                            taz_skims.at[i, 'job_housing_ratio_nongravity'] = nongravity_ratio_val
                        
                        job_housing_gravity= taz_skims['job_housing_gravity'].mean()
                        job_housing_nongravity= taz_skims['job_housing_nongravity'].mean()
                        job_housing_ratio_gravity= taz_skims['job_housing_ratio_gravity'].mean()
                        job_housing_ratio_nongravity= taz_skims['job_housing_ratio_nongravity'].mean()


        except Exception as e:
            logger.error(f"Error processing job-housing balance: {str(e)}")
            raise

        # Compile new performance measures
        measures_new = {
            'EMP_Den': pm_emp_den,
            'job_housing_gravity': job_housing_gravity,
            'job_housing_nongravity': job_housing_nongravity,
            'job_housing_ratio_gravity': job_housing_ratio_gravity,
            'job_housing_ratio_nongravity': job_housing_ratio_nongravity            
        }

        return measures_new
    
    def display_performance_measures(self, measures: dict, measures_new: dict):
        """Display performance measures with formatting."""
        logger.info("Performance Measures:")
        logger.info(f"{'':<10} {'2022':>30}")
        #logger.info(f"{'TOTAL':<10} {measures['HU_Tot']:.0f}, {measures['EMP_Tot']:.0f}")
        logger.info(f"{'ACRES':<14} {measures['ACRES_Tot']:>12.1f} {'7179.8':>15}")
        logger.info(f"{'TOD(hu, emp)':<14} {measures['TOD_HU']:>6.1%},{measures['TOD_EMP']:>6.1%} {'19.6%, 38.6%':>15}")
        logger.info(f"{'DT(hu, emp)':<14} {measures['DT_HU']:>6.1%},{measures['DT_EMP']:>6.1%} {'10.6%, 17.9%':>15}")
        logger.info(f"{'MU':<14} {measures['MU']:>12.1%} {'23.6%':>15}")
        logger.info(f"{'RES DEN':<14} {measures['RES_DEN']:>12.2f} {'7.66':>15}")
        logger.info(f"{'MF':<14} {measures['MF']:>12.1%} {'40.2%':>15}")
        logger.info(f"{'FARM AC':<14} {measures['FARM_AC']:>12.1f} {'1955.2':>15}")
        logger.info(f"{'FARM IMP':<14} {measures['FARM_IMP']:>12.1f} {'48.91':>15}")
        logger.info(f"{'Redev%':<14} {measures['Redev']:>12.1%} {'7.5%':>15}")
        logger.info(f"{'Infill (Fresno)':<14} {measures['Infill (Fresno)']:>12.1%} {'50.9%':>15}")
        logger.info(f"{'Infill (non-Fresno)':<14} {measures['Infill (non-Fresno)']:>8.1%} {'NA%':>15}")
        logger.info("\nDevelopment by Geographic Area:")
        logger.info(f"\n{measures['Geo_Area'].to_string(index=False)}")

        logger.info(f"\n{'EMP_Den':<30} {measures_new['EMP_Den']:>12.1f}")
        logger.info(f"{'job_housing_gravity':<30} {measures_new['job_housing_gravity']:>12.2f}")
        # logger.info(f"{'job_housing_nongravity':<30} {measures_new['job_housing_nongravity']:>12.2f}")
        # logger.info(f"{'job_housing_ratio_gravity':<30} {measures_new['job_housing_ratio_gravity']:>12.2f}")
        # logger.info(f"{'job_housing_ratio_nongravity':<30} {measures_new['job_housing_ratio_nongravity']:>12.2f}")

    

def main():
    logger.info("\n--- CALCULATE PERFORMANCE INDICATORS ---")
    logger.info(f"Start time: {datetime.now()}")

    try:
        param_file = sys.argv[1]
        target_year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        calculator = PerformanceIndicatorCalculator(param_file, target_year)

        # Load data
        base_maz, maz_new, dev_table = calculator.load_data()
        
        # Calculate and display performance measures
        measures = calculator.calculate_performance_measures(base_maz, maz_new, dev_table)
        measures_new = calculator.calculate_new_indicators(base_maz, maz_new, dev_table)
        calculator.display_performance_measures(measures, measures_new)
        
        logger.info("\n--- Script ran successfully! ---")
    except Exception as e:
        logger.error(f"\n--- Error occurred: {str(e)} ---", exc_info=True)
        raise

    logger.info(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()