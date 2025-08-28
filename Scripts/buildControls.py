# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:56:56 2025

@author: joshi
"""

from datetime import datetime
import numpy as np
import pandas as pd
import os
import sys
from typing import Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout), # Output to console
        logging.FileHandler(os.path.join(os.getcwd(), 'Setup/Logs/buildControls.log'), mode='w')
        ]
    )
logger = logging.getLogger(__name__)

class GrowthControlsBuilder:
    VISION_YEAR = 2035
    
    def __init__(self):
        self.parameters = None
        self.working_dir = None
        self.data_dir = None
        self.popsim_dir = None
        self.output_dir = None
        self.base_year = None
        self.target_year = None
        self.config_params = {}
        
    def load_parameters(self, param_file: str) -> None:
        """Load and process parameters file"""
        try:
            self.parameters = pd.read_csv(param_file)
            self.parameters.columns = ['Key', 'Value', 'Notes']
            
            # Extract configuration parameters
            self.working_dir = self._get_param('WORKING_DIR').strip()
            self.data_dir = os.path.join(self.working_dir, 'Data')
            self.popsim_dir = os.path.join(self.data_dir, 'PopSim')
            self.output_dir = os.path.join(self.working_dir, 'Setup', 'Data')
            
            self.base_year = int(self._get_param('baseYear').strip())
            self.config_params = {
                'Cube_P': float(self._get_param('Cube_P').strip()),
                'adjPOP': float(self._get_param('adjPOP').strip()),
                'adjEMP': float(self._get_param('adjEMP').strip()),
                'adjVacRate': float(self._get_param('adjVacRate').strip()),
                'adjUrban': float(self._get_param('adjUrban').strip())
                }
            logger.debug(f"Parameters loaded: base_year={self.base_year}, config_params={self.config_params}")
        except Exception as e:
            logger.error(f"Error loading parameters: {str(e)}")
            raise ValueError(f"Error loading parameters file: {str(e)}")
    
    def _get_param(self, key: str) -> str:
        """Helper method to get parameter value"""
        return self.parameters[self.parameters.Key == key]['Value'].item()
    
    def set_target_year(self, argv: list) -> None:
        """Set target year from command line or parameters file"""
        logger.info("Setting target year")
        try:
            self.target_year = int(argv[2]) if len(argv) > 2 else int(self._get_param('targetYear').strip())
            logger.debug(f"Target year set to: {self.target_year}")
        except Exception as e:
            logger.error(f"Error setting target year: {str(e)}")
            raise ValueError(f"Error setting target year: {str(e)}")
    
    def process_base_data(self) -> pd.DataFrame:
        """Process base MAZ data and calculate adjustements"""
        logger.info("Processing base MAZ data")
        base_maz = pd.read_csv(os.path.join(self.data_dir, "Base_MAZ_2023.csv"))
        
        # Calculate averages and adjustments
        hhsize_avg = base_maz['HH_POP'].sum() / base_maz['Base_HH'].sum()
        vacrate_avg = (base_maz['Base_HU'].sum() - base_maz['Base_HH'].sum()) / base_maz['Base_HU'].sum()
        vacrate_adj = np.clip(
            vacrate_avg + self.config_params['adjVacRate'] * 
            (min(self.VISION_YEAR, self.target_year) - self.base_year) / (self.VISION_YEAR - self.base_year),
            0.025, 0.3
            )
        
        # Apply household factor
        hh_factor = base_maz['Base_HU'].sum() * (1 - vacrate_adj) / base_maz['Base_HH'].sum()
        base_maz['Base_HH'] *= hh_factor
        base_maz['HH_POP'] *= hh_factor
        
        # Calculate additional fields
        # base_maz['Base_GQ'] = (base_maz['Base_DORM'] + base_maz['Base_MEDICAL'] + base_maz['Base_PRISON'] + base_maz['Base_MILITARY'])
        base_maz['Base_SCHL'] = base_maz['Base_ELEM'] + base_maz['Base_HS'] + base_maz['Base_COLLEGE']
        
        logger.debug(f"Base data processed: hhsize_avg={hhsize_avg}, vacrate_adj={vacrate_adj}, hh_factor={hh_factor}")
        logger.info("Base MAZ data processing completed")
        return base_maz
        
    def create_soi_base(self, base_maz: pd.DataFrame) -> pd.DataFrame:
        """Create SOI base data with aggregation"""
        logger.info("Creating SOI base data")
        base_soi = base_maz.groupby('SOI', as_index=False).agg({
            'HH_POP': 'sum', 'Base_HH': 'sum', 'Base_HU': 'sum', 'Base_EMP': 'sum',
            'Base_AGR': 'sum',  'Base_SCHL': 'sum'
            }) # 'Base_GQ': 'sum'
        
        # read vacrate from the dempogrpahic forecast csv
        demo_forecast = pd.read_csv(os.path.join(self.data_dir, "Demographic_Forecast.csv"))
        demo_forecast_nonCounty = demo_forecast[(demo_forecast['YEAR'] == self.target_year) & (demo_forecast['SOI'] != 'Fresno County')]
        demo_forecast_nonCounty = demo_forecast_nonCounty[['SOI', 'OccRate']]
        demo_forecast_nonCounty['OccRate_target'] = demo_forecast_nonCounty['OccRate']
        demo_forecast_nonCounty = demo_forecast_nonCounty[['SOI', 'OccRate_target']]
        
        base_soi = base_soi.merge(demo_forecast_nonCounty, on='SOI', how='left')
        # base_soi['VacRate'] = (base_soi['Base_HU'] - base_soi['Base_HH']) / base_soi['Base_HU']
        
        demo_forecast_nonCounty = demo_forecast[(demo_forecast['YEAR'] == self.base_year) & (demo_forecast['SOI'] != 'Fresno County')]
        demo_forecast_nonCounty = demo_forecast_nonCounty[['SOI', 'OccRate']]
        demo_forecast_nonCounty['OccRate_base'] = demo_forecast_nonCounty['OccRate']
        demo_forecast_nonCounty = demo_forecast_nonCounty[['SOI', 'OccRate_base']]
        
        base_soi = base_soi.merge(demo_forecast_nonCounty, on='SOI', how='left')
        
        # Add Fresno County totals
        hhsize_avg = base_soi['HH_POP'].sum() / base_soi['Base_HH'].sum()
        OccRate_adj_target = demo_forecast[(demo_forecast['YEAR'] == self.target_year) & (demo_forecast['SOI'] == 'Fresno County')]['OccRate'].item()
        OccRate_adj_base = demo_forecast[(demo_forecast['YEAR'] == self.base_year) & (demo_forecast['SOI'] == 'Fresno County')]['OccRate'].item()

        # vacrate_adj = (base_soi['Base_HU'].sum() - base_soi['Base_HH'].sum()) / base_soi['Base_HU'].sum()
        fresno_row = pd.DataFrame([{
            'SOI': 'Fresno County',
            'Base_HU': base_soi['Base_HU'].sum(),
            'Base_EMP': base_soi['Base_EMP'].sum(),
            'Base_AGR': base_soi['Base_AGR'].sum(),
            # 'Base_GQ': base_soi['Base_GQ'].sum(),
            'Base_SCHL': base_soi['Base_SCHL'].sum(),
            'Base_HH': base_soi['Base_HH'].sum(),
            'HH_POP': base_soi['HH_POP'].sum(),
            'HH_SIZE': hhsize_avg,
            'OccRate_target': OccRate_adj_target,
            'OccRate_base': OccRate_adj_base,
        }])
        
        result = pd.concat([base_soi, fresno_row], ignore_index=True)
        logger.info("SOI base data created with Fresno County totals")
        return result
    
    def process_forecast(self, base_soi: pd.DataFrame) -> pd.DataFrame:
        """Process demographic forecast data"""
        logger.info("Processing demographic forecast data")
        demo_forecast = pd.read_csv(os.path.join(self.data_dir, "Demographic_Forecast.csv"))
        
        # Base year data
        forecast_base = demo_forecast[demo_forecast['YEAR'] == self.base_year][[
            'SOI', 'YEAR', 'POP_HH', 'POP_GRP', 'POP_SCHL', 'HH_TOT', 'EMP_TOT',
            'EMP_EDU', 'EMP_FOO', 'EMP_GOV', 'EMP_IND', 'EMP_MED', 'EMP_OFC',
            'EMP_OTH', 'EMP_RET', 'EMP_AGR'
        ]].rename(columns={
            'YEAR': 'BASE_YEAR', 'POP_HH': 'HHPOP_BASE', 'POP_GRP': 'GQPOP_BASE',
            'POP_SCHL': 'SCHL_BASE', 'HH_TOT': 'HH_BASE', 'EMP_TOT': 'EMP_BASE',
            'EMP_EDU': 'EDU_BASE', 'EMP_FOO': 'FOO_BASE', 'EMP_GOV': 'GOV_BASE',
            'EMP_IND': 'IND_BASE', 'EMP_MED': 'MED_BASE', 'EMP_OFC': 'OFC_BASE',
            'EMP_OTH': 'OTH_BASE', 'EMP_RET': 'RET_BASE', 'EMP_AGR': 'AGR_BASE'
        })
        
        # Target year data
        forecast = demo_forecast[demo_forecast['YEAR'] == self.target_year][[
            'SOI', 'YEAR', 'POP_HH', 'POP_GRP', 'POP_SCHL', 'POP_TOT', 'RACE_White',
            'HH_TOT', 'EMP_TOT', 'EMP_EDU', 'EMP_FOO', 'EMP_GOV', 'EMP_IND',
            'EMP_MED', 'EMP_OFC', 'EMP_OTH', 'EMP_RET', 'EMP_AGR'
        ]].rename(columns={
            'YEAR': 'TARGET_YEAR', 'POP_HH': 'HHPOP_TARGET', 'POP_GRP': 'GQPOP_TARGET',
            'POP_SCHL': 'SCHL_TARGET', 'POP_TOT': 'POP_TOT_TARGET', 'RACE_White': 'POP_White_TARGET',
            'HH_TOT': 'HH_TARGET', 'EMP_TOT': 'EMP_TARGET', 'EMP_EDU': 'EDU_TARGET',
            'EMP_FOO': 'FOO_TARGET', 'EMP_GOV': 'GOV_TARGET', 'EMP_IND': 'IND_TARGET',
            'EMP_MED': 'MED_TARGET', 'EMP_OFC': 'OFC_TARGET', 'EMP_OTH': 'OTH_TARGET',
            'EMP_RET': 'RET_TARGET', 'EMP_AGR': 'AGR_TARGET'
        })
        
        # Merge and calculate targets
        forecast = forecast.merge(forecast_base, on='SOI').merge(base_soi, on='SOI')
        forecast = self._calculate_forecast_targets(forecast)
        
        logger.info("Demographic forecast processing completed")
        return forecast
    
    def _calculate_forecast_targets(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Calculate forecast targets with adjustments"""
        logger.debug("Calculating forecast targets")
        adj_pop = 1 + self.config_params['adjPOP']
        adj_emp = 1 + self.config_params['adjEMP']
        
        forecast['SOI_HH_Target'] = adj_pop * (forecast['HH_TARGET'] - forecast['HH_BASE'])
        forecast['SOI_OccRate'] = forecast['SOI_HH_Target'] / ((forecast['HH_TARGET'] / forecast['OccRate_target']) - (forecast['HH_BASE'] / forecast['OccRate_base']))
        forecast['SOI_HU_Target'] = forecast['SOI_HH_Target'] / forecast['SOI_OccRate']
        forecast['SOI_HHPOP_Target'] = adj_pop * (forecast['HHPOP_TARGET'] - forecast['HHPOP_BASE'])
        forecast['SOI_GQ_Target'] = adj_pop * (forecast['GQPOP_TARGET'] - forecast['GQPOP_BASE'])
        forecast['SOI_SCHL_Target'] = adj_pop * (forecast['SCHL_TARGET'] - forecast['SCHL_BASE'])
        forecast['SCHL_Factor'] = (forecast['SOI_SCHL_Target'] + forecast['Base_SCHL']) / forecast['Base_SCHL']
        forecast['HH_SIZE'] = forecast['SOI_HHPOP_Target'] / forecast['SOI_HH_Target']
        
        for sector in ['EDU', 'FOO', 'GOV', 'IND', 'MED', 'OFC', 'OTH', 'RET', 'AGR']:
            forecast[f'SOI_{sector}_Target'] = adj_emp * (forecast[f'{sector}_TARGET'] - forecast[f'{sector}_BASE'])
        
        forecast['SOI_EMP_Target'] = adj_emp * (forecast['EMP_TARGET'] - forecast['EMP_BASE'] - 
                                    forecast['SOI_AGR_Target'])
        
        return forecast
    
    def process_taz_controls(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Process TAZ-level growth controls"""
        logger.info("Processing TAZ-level growth controls")
        cube_growth = pd.read_csv(os.path.join(self.data_dir, "CubeGrowth_19_35.csv"))
        cube_growth = cube_growth.rename(columns={
            'Cube_HU_TAZ': 'TAZ_HU_Target',
            'Cube_EMP_TAZ': 'TAZ_EMP_Target'
        })
        
        cube_community = pd.read_csv(os.path.join(self.data_dir, "Communities.csv"))[[
            'COMMUNITY', 'SOI_HU_P', 'SOI_EMP_P'
        ]]
        
        # Calculate adjustments
        cube_soi = cube_growth.groupby('SOI', as_index=False).agg({
            'TAZ_HU_Target': 'sum',
            'TAZ_EMP_Target': 'sum'
        }).merge(forecast[['SOI', 'SOI_HU_Target', 'SOI_EMP_Target']], on='SOI')
        
        cube_soi['HU_adj'] = cube_soi['SOI_HU_Target'] / cube_soi['TAZ_HU_Target']
        cube_soi['EMP_adj'] = cube_soi['SOI_EMP_Target'] / cube_soi['TAZ_EMP_Target']
        
        # Apply adjustments
        cube_growth = cube_growth.merge(cube_soi[['SOI', 'HU_adj', 'EMP_adj']], on='SOI')
        cube_growth['TAZ_HU_Target'] = (self.config_params['Cube_P'] * 
                                      cube_growth['TAZ_HU_Target'] * cube_growth['HU_adj'])
        cube_growth['TAZ_EMP_Target'] = (self.config_params['Cube_P'] * 
                                       cube_growth['TAZ_EMP_Target'] * cube_growth['EMP_adj'])
        
        result = cube_growth.merge(cube_community, on='COMMUNITY', how='left')[['TAZ', 'TAZ_HU_Target', 
                                                                'TAZ_EMP_Target', 'SOI_HU_P', 
                                                                'SOI_EMP_P']].fillna(0)
        logger.info("TAZ controls processing completed")
        return result
    
def main():
    logger.info('\n--- BUILD GROWTH CONTROLS ---')
    logger.info(f'Start time: {datetime.now()}')
    
    try:
        builder = GrowthControlsBuilder()
        builder.load_parameters(sys.argv[1])
        builder.set_target_year(sys.argv)
        
        # Process data
        base_maz = builder.process_base_data()
        base_soi = builder.create_soi_base(base_maz)
        forecast = builder.process_forecast(base_soi)
        taz_controls = builder.process_taz_controls(forecast)
        
        # Save outputs
        logger.info("Saving output files")
        forecast.to_csv(os.path.join(builder.output_dir, "controls_soi.csv"), index=False)
        taz_controls.to_csv(os.path.join(builder.output_dir, "controls_taz.csv"), index=False)
        
        logger.info('\n--- Script ran successfully! ---')
    except Exception as e:
        logger.error(f'\n--- Error occurred: {str(e)} ---', exc_info=True)
        raise
    
    logger.info(f'End time: {datetime.now()}')

if __name__ == "__main__":
    main()