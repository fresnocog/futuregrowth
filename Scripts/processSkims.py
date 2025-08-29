# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 08:33:32 2025

@author: joshi

Modified on 8/20/2025: lines with comment #new
"""

from datetime import datetime
import openmatrix as omx
import numpy as np
import numpy.ma as ma
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
        logging.FileHandler(os.path.join(os.getcwd(), 'Setup/Logs/skim_processing.log'), mode='w') # Overwrites log file each run
    ]
)
logger = logging.getLogger(__name__)

class SkimProcessor:
    VISION_YEAR = 2035
    NO_DEV = 9999
    
    def __init__(self, param_file: str, target_year: int = None):
        """Initialize with parameters file and optional target year."""
        self.parameters = pd.read_csv(param_file)
        self.parameters.columns = ['Key', 'Value', 'Notes']
        self.working_dir = self._get_param('WORKING_DIR').strip()
        self.data_dir = os.path.join(self.working_dir, 'Data')
        self.abm_dir = os.path.join(self.data_dir, 'ABM')
        self.output_dir = os.path.join(self.working_dir, 'Setup', 'Data')
        self.base_year = int(self._get_param('baseYear').strip())
        self.target_year = target_year if target_year else int(self._get_param('targetYear').strip())
        self.base_maz = None
    
    def _get_param(self, Key: str) -> str:
        """Helper method to get parameter value"""
        return self.parameters[self.parameters.Key == Key]['Value'].item()
    
    def load_base_data(self) -> None:
        """Load base MAZ data"""
        logger.info("Loading base MAZ data")
        try:
            self.base_maz = pd.read_csv(os.path.join(self.data_dir, "Base_MAZ_2023.csv"))
            logger.debug(f"Base MAZ data loaded with {len(self.base_maz)} rows")
        except Exception as e:
            logger.error(f"Error loading Base_MAZ_2019.csv: {str(e)}")
            raise
    
    def process_maz_skims(self) -> pd.DataFrame:
        """Process MAZ level bike skims."""
        logger.info("Processing MAZ bike skims")
        maz_skims = self.base_maz[['MAZ', 'TAZ', 'SOI']]
        emp_array_maz = self.base_maz['Base_EMP'].to_numpy()
        hu_array_maz = self.base_maz['Base_HU'].to_numpy() #new
        
        # Calculate bike accessibility
        maz_skims['bike_skim'] = 0.0
        maz_skims['bike_skim_emp'] = 0.0 #new
        try:
            with omx.open_file(os.path.join(self.abm_dir, f"FC{str(self.base_year)[-2:]}_BASE_MAZ_SKM_BIKE.omx")) as bike_skim_omx:
                bike_skims = bike_skim_omx['TIME_BIKE'] #['DIST_BIKE']
                for i in maz_skims.index:
                    row = bike_skims[i]
                    mask = ma.masked_where(row <= 30, row).mask  # Distance <= 5 miles (`30 min by bike) #new corrected 5 to 30 to reflect time
                    mask1 = ma.masked_where(row > 0, row).mask
                    skim_val = (mask1 * mask * emp_array_maz).sum()
                    maz_skims.at[i, 'bike_skim'] = skim_val + emp_array_maz[i]
                    
                    skim_val_emp = (mask1 * mask * hu_array_maz).sum() #new
                    maz_skims.at[i, 'bike_skim_emp'] = skim_val_emp + hu_array_maz[i] #new
        except Exception as e:
            logger.error(f"Error processing bike skims: {str(e)}")
            raise
        
        # Calculate bike skim indexes
        logger.info("Calculating bike skim indexes")
        maz_skims['IDX_Bike'] = 0.0 
        valid_bike = maz_skims['bike_skim'] > 0
        maz_skims.loc[valid_bike, 'IDX_Bike'] = maz_skims[valid_bike]['bike_skim'].rank(pct=True)

        maz_skims['IDX_Bike_emp'] = 0.0 #new
        valid_bike_emp = maz_skims['bike_skim_emp'] > 0 #new
        maz_skims.loc[valid_bike_emp, 'IDX_Bike_emp'] = maz_skims[valid_bike_emp]['bike_skim_emp'].rank(pct=True) #new

        logger.debug(f"Bike skim range: min={maz_skims['bike_skim'].min()}, max={maz_skims['bike_skim'].max()}")
        logger.debug(f"Bike skim emp range: min={maz_skims['bike_skim_emp'].min()}, max={maz_skims['bike_skim_emp'].max()}") #new
        
        return maz_skims
    
    def process_taz_skims(self) -> pd.DataFrame:
        """Process TAZ level transit and SOV skims."""
        logger.info("Processing TAZ Skims")
        taz_skims = pd.DataFrame(list(range(1, 3001)), columns=['TAZ'])
        taz_skims = taz_skims.merge(
            self.base_maz.groupby('TAZ').agg({'SOI': 'first', 'Base_EMP': 'sum', 'Base_HU': 'sum'}),
            how='left',
            on='TAZ'
        )
        taz_skims['SOI'] = taz_skims['SOI'].fillna("")
        taz_skims['Base_EMP'] = taz_skims['Base_EMP'].fillna(0)
        emp_array_taz = taz_skims['Base_EMP'].to_numpy()

        taz_skims['Base_HU'] = taz_skims['Base_HU'].fillna(0) #new
        hu_array_taz = taz_skims['Base_HU'].to_numpy() #new
        
        # Calcualte transit accessibility
        try:
            with omx.open_file(os.path.join(self.abm_dir, f"FC{str(self.base_year)[-2:]}_BASE_SKM_PK_TWB.omx")) as transit_skim_omx:
                for i in taz_skims.index:
                    row = (
                        transit_skim_omx['IVTT'][i] + transit_skim_omx['WLK_P'][i] +
                        transit_skim_omx['WLK_A'][i] + transit_skim_omx['WLK_X'][i] +
                        transit_skim_omx['IWAIT'][i] + transit_skim_omx['XWAIT'][i]
                        )
                    mask = ma.masked_where(row <= 60, row).mask # within 60 minutes
                    mask1 = ma.masked_where(row > 0, row).mask
                    skim_val = (mask * mask1 * emp_array_taz).sum()
                    taz_skims.at[i, 'transit_skim'] = skim_val

                    skim_val_emp = (mask * mask1 * hu_array_taz).sum() #new
                    taz_skims.at[i, 'transit_skim_emp'] = skim_val_emp #new

        except Exception as e:
            logger.error(f"Error processing transit skims: {str(e)}")
            raise
        
        # Calculate transit skim indexes
        logger.info("Calculating transit skim indexes")
        taz_skims['IDX_Transit'] = 0.0
        valid_transit = taz_skims['transit_skim'] > 0
        taz_skims.loc[valid_transit, 'IDX_Transit'] = taz_skims[valid_transit]['transit_skim'].rank(pct=True)

        taz_skims['IDX_Transit_emp'] = 0.0 #new
        valid_transit_emp = taz_skims['transit_skim_emp'] > 0 #new
        taz_skims.loc[valid_transit_emp, 'IDX_Transit_emp'] = taz_skims[valid_transit_emp]['transit_skim_emp'].rank(pct=True) #new

        logger.debug(f"Transit skim range: min={taz_skims['transit_skim'].min()}, max={taz_skims['transit_skim'].max()}")
        logger.debug(f"Transit skim emp range: min={taz_skims['transit_skim_emp'].min()}, max={taz_skims['transit_skim_emp'].max()}") #new
        
        # Calculating SOV accessibility
        taz_skims['sov_skim'] = 0.0
        taz_skims['sov_skim_emp'] = 0.0 #new
        try:
            with omx.open_file(os.path.join(self.abm_dir, f"FC{str(self.base_year)[-2:]}_BASE_SKM_PM_D1.omx")) as sov_skim_omx: # FC{str(self.base_year)[-2:]}_BASE_SKM_PK_D1.omx
                sov_skims = sov_skim_omx['TIME_1Veh']
                for i in taz_skims.index:
                    row = sov_skims[i]
                    mask = ma.masked_where(row > 0, row).mask
                    skim_val = (mask * emp_array_taz * np.exp(-0.049601 * row)).sum()
                    taz_skims.at[i, 'sov_skim'] = skim_val + emp_array_taz[i]

                    skim_val_emp = (mask * hu_array_taz * np.exp(-0.049601 * row)).sum() #new
                    taz_skims.at[i, 'sov_skim_emp'] = skim_val + hu_array_taz[i] #new

        except Exception as e:
            logger.error(f"Error processing SOV skims: {str(e)}")
            raise
        
        # Calculate SOV skim indexes
        logger.info("Calculating SOV skim indexes")
        taz_skims['IDX_SOV'] = 0.0
        valid_sov = taz_skims['sov_skim'] > 0
        taz_skims.loc[valid_sov, 'IDX_SOV'] = taz_skims[valid_sov]['sov_skim'].rank(pct=True)

        taz_skims['IDX_SOV_emp'] = 0.0 #new
        valid_sov_emp = taz_skims['sov_skim_emp'] > 0 #new
        taz_skims.loc[valid_sov, 'IDX_SOV_emp'] = taz_skims[valid_sov_emp]['sov_skim_emp'].rank(pct=True) #new

        logger.debug(f"SOV skim range: min={taz_skims['sov_skim'].min()}, max={taz_skims['sov_skim'].max()}")
        logger.debug(f"SOV skim emp range: min={taz_skims['sov_skim_emp'].min()}, max={taz_skims['sov_skim_emp'].max()}") #new
        
        return taz_skims
        
    def save_outputs(self, maz_skims: pd.DataFrame, taz_skims: pd.DataFrame) -> None:
        """Save skim results to CSV files."""
        logger.info("Saving output files")
        try:
            maz_skims.to_csv(os.path.join(self.output_dir, "skims_maz.csv"), index=False)
            logger.info("Saved skims_maz.csv")
        except Exception as e:
            logger.error(f"Error saving skims_maz.csv: {str(e)}")
            raise
        
        try:
            taz_skims.to_csv(os.path.join(self.output_dir, "skims_taz.csv"), index=False)
            logger.info("Saved skims_taz.csv")
        except Exception as e:
            logger.error(f"Error saving skims_taz.csv: {str(e)}")
            raise

def main():
    logger.info("\n--- SKIM PROCESSING ---")
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        # Initialize processor with parameters file and optional target year
        param_file = sys.argv[1]
        target_year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        processor = SkimProcessor(param_file, target_year)
        
        logger.info(f"--- YEAR {processor.target_year} ---")
        processor.load_base_data()
        
        # Process skims
        maz_skims = processor.process_maz_skims()
        taz_skims = processor.process_taz_skims()
        
        # Save results
        processor.save_outputs(maz_skims, taz_skims)
        
        logger.info("\n--- Script ran successfully! ---")
    except Exception as e:
        logger.error(f"\n--- Error occurred: {str(e)} ---", exc_info=True)
        raise
    
    logger.info(f"End time: {datetime.now()}")
    
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        