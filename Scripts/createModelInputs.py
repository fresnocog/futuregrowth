# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 16:52:36 2025

@author: joshi
"""

from datetime import datetime
import pandas as pd
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), 'Setup/Logs/abm_inputs.log'), mode='w')  # Overwrites log file each run
    ]
)
logger = logging.getLogger(__name__)

class ABMInputGenerator:
    def __init__(self, param_file: str, target_year: int = None):
        """Initialize with parameters file and optional target year."""
        self.parameters = pd.read_csv(param_file)
        self.parameters.columns = ['Key', 'Value', 'Notes']
        self.working_dir = self._get_param('WORKING_DIR').strip()
        self.data_dir = os.path.join(self.working_dir, 'Data')
        self.data_abm_dir = os.path.join(self.data_dir, 'ABM')
        self.output_dir = os.path.join(self.working_dir, 'Setup', 'Data')
        self.popsim_dir = os.path.join(self.working_dir, 'Setup', 'Outputs')
        self.target_year = target_year if target_year else int(self._get_param('targetYear').strip())

    def _get_param(self, key: str) -> str:
        """Helper method to get parameter value."""
        return self.parameters[self.parameters.Key == key]['Value'].item()
    
    def load_data(self) -> tuple:
        """Load new MAZ and TAZ data along with base ABM inputs."""
        logger.info("Loading input data")
        try:
            maz_new = pd.read_csv(os.path.join(self.output_dir, 'maz_new.csv'))
            taz_new = pd.read_csv(os.path.join(self.output_dir, 'taz_new.csv'))
            maz_parks_base = pd.read_csv(os.path.join(self.data_abm_dir, 'maz_2019_parks.csv'))
            se_detail_base = pd.read_csv(os.path.join(self.data_abm_dir, 'FC19_Base_SE_Detail.csv'))
            logger.debug(f"Loaded maz_new: {maz_new.shape}, taz_new: {taz_new.shape}, "
                        f"maz_parks_base: {maz_parks_base.shape}, se_detail_base: {se_detail_base.shape}")
            return maz_new, taz_new, maz_parks_base, se_detail_base
        except Exception as e:
            logger.error(f"Error loading input data: {str(e)}")
            raise
            
    def generate_maz_parks(self, maz_new: pd.DataFrame, maz_parks_base: pd.DataFrame) -> pd.DataFrame:
        """Generate maz_parks file for the target year."""
        logger.info("Generating maz_parks")
        maz_parks = maz_parks_base[['parcelid', 'xcoord_p', 'ycoord_p', 'sqft_p', 'taz_p', 'block_p', 
                                    'parkdy_p', 'parkhr_p', 'ppricdyp', 'pprichrp']].copy()
        maz_parks['MAZ'] = maz_parks['parcelid']
        maz_parks = maz_parks.merge(maz_new, how='left', on='MAZ')
        
        # Rename columns to match ABM format
        rename_dict = {
            'HH': 'hh_p', 'ELEM': 'stugrad_p', 'HS': 'stuhgh_p', 'COLLEGE': 'stuuni_p',
            'EMP_EDU': 'empedu_p', 'EMP_FOO': 'empfoo_p', 'EMP_GOV': 'empgov_p', 
            'EMP_IND': 'empind_p', 'EMP_MED': 'empmed_p', 'EMP_OFC': 'empofc_p', 
            'EMP_RET': 'empret_p', 'EMP_OTH': 'empsvc_p', 'EMP_AGR': 'empoth_p', 'EMP': 'emptot_p'
        }
        maz_parks.rename(columns=rename_dict, inplace=True)
        
        # Filter and reorder columns
        columns = ['parcelid', 'xcoord_p', 'ycoord_p', 'sqft_p', 'taz_p', 'block_p', 'hh_p', 
                  'stugrad_p', 'stuhgh_p', 'stuuni_p', 'empedu_p', 'empfoo_p', 'empgov_p', 
                  'empind_p', 'empmed_p', 'empofc_p', 'empret_p', 'empsvc_p', 'empoth_p', 
                  'emptot_p', 'parkdy_p', 'parkhr_p', 'ppricdyp', 'pprichrp']
        maz_parks = maz_parks[columns]
        
        output_file = os.path.join(self.popsim_dir, f'maz_{self.target_year}_parks.csv')
        maz_parks.to_csv(output_file, index=False)
        logger.info(f"Saved maz_parks to {output_file}")
        logger.debug(f"maz_parks shape: {maz_parks.shape}")
        return maz_parks
    
    def generate_se_detail(self, taz_new: pd.DataFrame, se_detail_base: pd.DataFrame) -> pd.DataFrame:
        """Generate se_detail file for the target year."""
        logger.info("Generating se_detail")
        se_detail = se_detail_base[['; TAZ', 'COUNTY', 'CITY']].copy()
        se_detail['TAZ'] = se_detail['; TAZ']
        se_detail = se_detail.merge(taz_new, how='left', on='TAZ')
        
        # Rename columns to match ABM format
        rename_dict = {
            'HH': 'TOTHH', 'POP': 'TOTPOP', 'EMP': 'TOTEMP', 'EMP_EDU': 'EMPEDU', 
            'EMP_FOO': 'EMPFOO', 'EMP_GOV': 'EMPGOV', 'EMP_IND': 'EMPIND', 
            'EMP_MED': 'EMPMED', 'EMP_OFC': 'EMPOFC', 'EMP_OTH': 'EMPOTH', 
            'EMP_RET': 'EMPRET', 'EMP_AGR': 'EMPAGR'
        }
        se_detail.rename(columns=rename_dict, inplace=True)
        
        # Filter and fill NaN values
        columns = ['; TAZ', 'COUNTY', 'CITY', 'TOTHH', 'TOTPOP', 'TOTEMP', 'EMPEDU', 
                  'EMPFOO', 'EMPGOV', 'EMPIND', 'EMPMED', 'EMPOFC', 'EMPOTH', 'EMPRET', 'EMPAGR']
        se_detail = se_detail[columns].fillna(0)
        
        output_file = os.path.join(self.popsim_dir, f'FC{self.target_year % 100}_Base_SE_Detail.csv')
        se_detail.to_csv(output_file, index=False)
        logger.info(f"Saved se_detail to {output_file}")
        logger.debug(f"se_detail shape: {se_detail.shape}")
        return se_detail

def main():
    logger.info("\n--- GENERATE ABM INPUTS ---")
    logger.info(f"Start time: {datetime.now()}")

    try:
        param_file = sys.argv[1]
        target_year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        generator = ABMInputGenerator(param_file, target_year)

        # Load data
        maz_new, taz_new, maz_parks_base, se_detail_base = generator.load_data()
        
        # Generate ABM inputs
        maz_parks = generator.generate_maz_parks(maz_new, maz_parks_base)
        se_detail = generator.generate_se_detail(taz_new, se_detail_base)
        
        logger.info("\n--- Script ran successfully! ---")
    except Exception as e:
        logger.error(f"\n--- Error occurred: {str(e)} ---", exc_info=True)
        raise

    logger.info(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()
    