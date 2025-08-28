# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:41:42 2025

@author: joshi
"""

from datetime import datetime
import numpy as np
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
        logging.FileHandler(os.path.join(os.getcwd(), 'Setup/Logs/calcDevScores.log'), mode='w')  # Overwrites log file each run
    ]
)
logger = logging.getLogger(__name__)

class DevScoreCalculator:
    VISION_YEAR = 2035
    NO_DEV = 9999
    
    def __init__(self, param_file: str, target_year: int = None, override_dev: str = None):
        """Initialize with parameters file, optional target year, and override development flag"""
        self.parameters = pd.read_csv(param_file)
        self.parameters.columns = ['Key', 'Value', 'Notes']
        self.working_dir = self._get_param('WORKING_DIR').strip()
        self.data_dir = os.path.join(self.working_dir, 'Data')
        self.output_dir = os.path.join(self.working_dir, 'Setup', 'Data')
        self.base_year = int(self._get_param('baseYear').strip())
        self.target_year = target_year if target_year else int(self._get_param('targetYear').strip())
        self.keep_devtable = self.target_year != self.VISION_YEAR if override_dev is None else not bool(override_dev)
        
        # Load parameters
        self.config_params = {
            'Cube_P': float(self._get_param('Cube_P')),
            'wtInfill': float(self._get_param('wtInfill')),
            'wtCons': float(self._get_param('wtCons')),
            'wtDensity': float(self._get_param('wtDensity')),
            'wtBike': float(self._get_param('wtBike')),
            'wtTransit': float(self._get_param('wtTransit')),
            'wtSOV': float(self._get_param('wtSOV')),
            'wtVMT': float(self._get_param('wtVMT')),
            'penaltyInfill': float(self._get_param('penaltyInfill')),
            'penaltyRedev': float(self._get_param('penaltyRedev')),
            'penaltyDensity': float(self._get_param('penaltyDensity')),
            'adjSF': float(self._get_param('adjSF')),
            'adjMU': float(self._get_param('adjMU')),
            'adjTOD': float(self._get_param('adjTOD')),
            'adjDT': float(self._get_param('adjDT')),
            'adjResDen': float(self._get_param('adjRESDEN')),
            'adjEmpDen': float(self._get_param('adjEMPDEN')),
            'HiDenPercentile': float(self._get_param('HiDenPercentile')),
            'RedevMinDen': float(self._get_param('RedevMinDen'))
            }
        
    def _get_param(self, key: str) -> str:
        """Helper method to get parameter value."""
        return self.parameters[self.parameters.Key == key]['Value'].item().strip()
    
    def load_data(self) -> tuple:
        """Load base data, forecast, growth controls, and parcels."""
        logger.info("Loading input data")
        try:
            base_maz = pd.read_csv(os.path.join(self.data_dir, "Base_MAZ_2023.csv"))
            forecast = pd.read_csv(os.path.join(self.output_dir, "controls_soi.csv"))
            cube_growth = pd.read_csv(os.path.join(self.output_dir, "controls_taz.csv"))
            
            if self.target_year != self.VISION_YEAR:
                parcels = pd.read_csv(os.path.join(self.output_dir, "parcels.csv"))
            else:
                parcels = pd.read_csv(os.path.join(self.data_dir, "Parcel_Data.csv"))
            logger.debug(f"Loaded data: base_maz={base_maz.shape}, forecast={forecast.shape}, cube_growth={cube_growth.shape}, parcels={parcels.shape}")
            return base_maz, forecast, cube_growth, parcels
        except Exception as e:
            logger.error(f"Error loading input data: {str(e)}")
            raise
    
    def calculate_base_scores(self, parcels: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
        """Calculate static base scores for parcels (first iteration only)."""
        if self.target_year == self.VISION_YEAR:
            logger.info("Calculating base scores for parcels")
            parcels = parcels[parcels['SOI'] != ' ']
            parcels['DEVTYPE'] = parcels['DEVTYPE_MI']
            parcels.loc[parcels['parcelid'] % 100 <= self.config_params['HiDenPercentile'] *100, 'DEVTYPE'] = parcels['DEVTYPE_HI']
            
            # Merge with DevTypes and Forecast
            dev_types = pd.read_csv(os.path.join(self.data_dir, "DevTypes.csv"))
            parcels = parcels.merge(dev_types, how='left', on='DEVTYPE')
            parcels = parcels.merge(forecast[['SOI', 'SOI_OccRate', 'HH_SIZE', 'SCHL_Factor']], how='left', on='SOI')
            
            # Adjust densities and calculate net growth possiblities
            parcels['HU_Den'] = parcels['HU_Den'] * (1 + self.config_params['adjResDen'])
            parcels['EMP_Den'] = parcels['EMP_Den'] * (1 + self.config_params['adjEmpDen'])
            parcels['Net_Density'] = (
                parcels['HU_Den'] + parcels['EMP_Den'] - 
                ((parcels['HU'] + parcels['EMP'] - parcels['EMP_AGR']) / parcels['ACRES']
                ))
            parcels['HU_NET'] = parcels['HU_Den'] * parcels['ACRES'] - parcels['HU']
            parcels['EMP_NET'] = (parcels['EMP_Den'] * parcels['ACRES']) - (parcels['EMP']) + (parcels['EMP_AGR'])
            parcels['NET_GROWTH'] = parcels['HU_NET'] + parcels['EMP_NET']
            parcels['Developed'] = (parcels['Vacant'] + 1) % 2
            parcels['Incorporated'] = 0
            parcels.loc[parcels['SOI'] != 'Unincorporate','Incorporated'] = 1
            
            # Filter parcels based on redevelopment threshold
            parcels = parcels[parcels['Net_Density'] - self.config_params['RedevMinDen'] * parcels['Developed'] > 0]
            
            # Calculate conservation, infill, and other indices
            
            parcels['Cons_GW'] = (1 - parcels['GW_IDX']).clip(0, 1)
            parcels['Cons_F'] = (
                1 - (parcels['FMMP_L'] + parcels['FMMP_P'] + parcels['FMMP_S'] + parcels['FMMP_U']) / 2 -
                (1 - parcels['Incorporated']) * (parcels['FMMP_P'] + parcels['FMMP_S'] + parcels['FMMP_U'])
                ).clip(0, 1)
            parcels['IDX_Cons'] = parcels[['Cons_GW', 'Cons_F']].min(axis=1)
            parcels['IDX_Infill'] = (1 - parcels['Infill'] / 10560).clip(0, 1)
            parcels['IDX_Den'] = (parcels['Net_Density'] / parcels['Net_Density'].max())
            parcels['IDX_MU'] = parcels['MU']
            parcels['IDX_SF'] = parcels['HU_SF_P']
            
            # Calculate Base Score
            parcels['BASE_SCORE'] = (
                parcels['IDX_Cons'] * self.config_params['wtCons'] +
                parcels['IDX_Infill'] * self.config_params['wtInfill'] +
                parcels['IDX_Den'] * self.config_params['wtDensity'] +
                parcels['IDX_VMT'] * self.config_params['wtVMT'] #+
                #0.01 * (1 - parcels['Dist_Col'] / 10560).clip(0, 1)
                )
            logger.info("Base scores calculated")
            self._save_parcels(parcels)
        return parcels
    
    def _save_parcels(self, parcels: pd.DataFrame) -> None:
        """Save intermediate parcels data."""
        try:
            parcels.to_csv(os.path.join(self.output_dir, "parcels.csv"), index=False)
            logger.info("Saved intermediate parcels.csv")
        except Exception as e:
            logger.error(f"Error saving parcels.csv: {str(e)}")
            raise
    
    def merge_skims(self, parcels: pd.DataFrame) -> pd.DataFrame:
        """Merge skim results into parcels."""
        logger.info("Merging skim results")
        try:
            skims_maz = pd.read_csv(os.path.join(self.output_dir, "skims_maz.csv"))[['MAZ', 'IDX_Bike']]
            parcels = parcels.merge(skims_maz, how='left', on='MAZ')
            skims_taz = pd.read_csv(os.path.join(self.output_dir, "skims_taz.csv"))[['TAZ', 'IDX_Transit', 'IDX_SOV']]
            parcels = parcels.merge(skims_taz, how='left', on='TAZ')
            logger.debug(f"Parcels after skim merge: {parcels.shape}")
            return parcels
        except Exception as e:
            logger.error(f"Error merging skims: {str(e)}")
            raise
    
    def calculate_total_scores(self, parcels: pd.DataFrame) -> pd.DataFrame:
        """Claculate final development scores"""
        logger.info("Calculating total development scores")
        parcels['TOTAL_SCORE'] = (
            parcels['BASE_SCORE'] +
            parcels['IDX_Bike'] * self.config_params['wtBike'] + 
            parcels['IDX_Transit'] * self.config_params['wtTransit'] + 
            parcels['IDX_SOV'] * self.config_params['wtSOV']
            )
        
        # Apply geometric adjustments
        parcels['TOTAL_SCORE'] = (
            parcels['TOTAL_SCORE'] * parcels['SCORE_ADJ'] *                 # General score adjustment factor
            (1 - self.config_params['penaltyRedev'] * parcels['Developed']) *  # Penalty for redeveloping parcels
            (1 + self.config_params['adjSF'] * parcels['IDX_SF']) *            # Boost for single-family potential
            (1 + self.config_params['adjMU'] * parcels['IDX_MU']) *             # Boost for mixed-use potential
            (1 + self.config_params['adjTOD'] * parcels['TOD']) *                   # Boost for TOD development
            (1 + self.config_params['adjDT'] * parcels['DT'])                   #Boost for DT parcels
            )
        
        # Apply density penalty for residential density
        parcels.loc[parcels['HU_Den'] > 0, 'TOTAL_SCORE'] *= (
            1 - self.config_params['penaltyDensity'] * parcels['IDX_Den']
            )
        
        # Apply infill penalty for vision year and beyond
        # if self.target_year >= self.VISION_YEAR:
            # parcels.loc[parcels['Infill'] == 0, 'TOTAL_SCORE'] *= (1 - self.config_params['penaltyInfill'])
        
        if self.target_year >= self.VISION_YEAR:
            parcels.loc[(parcels['Infill'] == 0) & (parcels['SOI'] == 'Fresno'), 'TOTAL_SCORE'] *= (1 - self.config_params['penaltyInfill'])
        
        
        # Filter final columns
        parcels = parcels[['parcelid', 'SOI', 'COMMUNITY', 'TAZ', 'HU_NET', 'EMP_NET', 
                          'SOI_OccRate', 'HH_SIZE', 'BASE_SCORE', 'IDX_Bike', 'IDX_Transit', 
                          'IDX_SOV', 'TOTAL_SCORE']]
        logger.debug(f"Total score range: min={parcels['TOTAL_SCORE'].min()}, max={parcels['TOTAL_SCORE'].max()}")
        return parcels
    
    def create_dev_table(self, parcels: pd.DataFrame, cube_growth: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
        """Create development table."""
        logger.info("Creating development table")
        forecast_dev = forecast[['SOI', 'SOI_HU_Target', 'SOI_EMP_Target']]
        
        if not self.keep_devtable:
            dev_table = parcels.merge(cube_growth, how='left', on='TAZ')
            dev_table = dev_table[[
                'parcelid', 'SOI', 'COMMUNITY', 'TAZ', 'HU_NET', 'EMP_NET', 'TOTAL_SCORE',
                'TAZ_HU_Target', 'TAZ_EMP_Target', 'SOI_HU_P', 'SOI_EMP_P'
                ]]
            dev_table['DEV'] = self.NO_DEV
            dev_table['DEV_TAZ'] = 0
            dev_table['DEV_SOI'] = 0
        else:
            dev_table = pd.read_csv(os.path.join(self.output_dir, "devtable.csv"))
            dev_table = dev_table.drop(columns=['TOTAL_SCORE', 'SOI_HU_Target', 'SOI_EMP_Target'], errors='ignore')
            dev_table = dev_table.merge(parcels[['parcelid', 'TOTAL_SCORE']], how='left', on='parcelid')
        
        dev_table = dev_table.merge(forecast_dev, how='left', on='SOI')
        dev_table = dev_table.sort_values(by=['TAZ', 'TOTAL_SCORE'], ascending=['True', 'False']).reset_index(drop=True)
        logger.debug(f"Development table shape: {dev_table.shape}")
        return dev_table
    
    def save_outputs(self, parcels: pd.DataFrame, dev_table: pd.DataFrame) -> None:
        """ Save final outputs."""
        logger.info("Saving output files")
        try:
            parcels.to_csv(os.path.join(self.output_dir, "parcels_Final.csv"), index=False)
            dev_table.to_csv(os.path.join(self.output_dir, "devtable.csv"), index=False)
            logger.info(f"Saved parcels_Final.csv and devtable.csv for year {self.target_year}")
        except Exception as e:
             logger.error(f"Error saving outputs: {str(e)}")
             raise

def main():
    logger.info("\n--- DEVELOPMENT SCORE CALCULATION ---")
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        param_file = sys.argv[1]
        target_year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        override_dev = sys.argv[3] if len(sys.argv) > 3 else None
        calculator = DevScoreCalculator(param_file, target_year, override_dev)
        
        logger.info(f"--- YEAR {calculator.target_year} ---")
        
        # Load data
        base_maz, forecast, cube_growth, parcels = calculator.load_data()
        
        # Calculate scores
        parcels = calculator.calculate_base_scores(parcels, forecast)
        parcels = calculator.merge_skims(parcels)
        parcels = calculator.calculate_total_scores(parcels)
        dev_table = calculator.create_dev_table(parcels, cube_growth, forecast)
        
        # Save results
        calculator.save_outputs(parcels, dev_table)
        
        logger.info("\n--- Script ran successfully! ---")
    except Exception as e:
        logger.error(f"\n--- Error occurred: {str(e)} ---", exc_info=True)
        raise
        
    logger.info(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()