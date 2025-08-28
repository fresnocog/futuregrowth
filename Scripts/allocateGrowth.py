# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:58:33 2025

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
        logging.FileHandler(os.path.join(os.getcwd(), 'Setup/Logs/allocateGrowth.log'), mode='w')  # Overwrites log file each run
    ]
)
logger = logging.getLogger(__name__)

class LandUseAllocator:
    VISION_YEAR = 2035
    NO_DEV = 9999
    
    def __init__(self, param_file: str, target_year: int = None):
        """Initialize with parameters file and optional target year."""
        self.parameters = pd.read_csv(param_file)
        self.parameters.columns = ['Key', 'Value', 'Notes']
        self.working_dir = self._get_param('WORKING_DIR').strip()
        self.data_dir = os.path.join(self.working_dir, 'Data')
        self.data_popsim_dir = os.path.join(self.data_dir, 'PopSim')
        self.output_dir = os.path.join(self.working_dir, 'Setup', 'Data')
        self.popsim_dir = os.path.join(self.working_dir, 'Setup', 'Outputs')
        self.base_year = int(self._get_param('baseYear').strip())
        self.target_year = target_year if target_year else int(self._get_param('targetYear').strip())
        self.cube_p = float(self._get_param('Cube_P').strip())
        
    def _get_param(self, key: str) -> str:
        """Helper method to get parameter value."""
        return self.parameters[self.parameters.Key == key]['Value'].item()
    
    def load_data(self) -> tuple:
        """Load base MAZ, forecast, and development table data."""
        logger.info("Loading input data")
        try:
            base_maz = pd.read_csv(os.path.join(self.data_dir, "Base_MAZ_2023.csv"))
            forecast = pd.read_csv(os.path.join(self.output_dir, "controls_soi.csv"))
            dev_table = pd.read_csv(os.path.join(self.output_dir, "devtable.csv"))
            gq_maz = pd.read_csv(os.path.join(self.data_dir, "gq_maz_2023.csv"))
            logger.debug(f"Loaded base_maz: {base_maz.shape}, forecast: {forecast.shape}, dev_table: {dev_table.shape}, gq_maz: {gq_maz.shape}")
            return base_maz, forecast, dev_table, gq_maz
        except Exception as e:
            logger.error(f"Error loading input data: {str(e)}")
            raise
    
    def calculate_scaling_factors(self, base_maz: pd.DataFrame, forecast: pd.DataFrame) -> tuple:
        """Calculate regional scaling factors for group quarters, EMP_AGR)."""
        logger.info("Calculating scaling factors")
        # gq_base = (base_maz['Base_DORM'] + base_maz['Base_MEDICAL'] +
        #            base_maz['Base_PRISON'] + base_maz['Base_MILITARY']).sum()
        # gq_factor = (forecast['SOI_GQ_Target'].sum() + gq_base) / gq_base if gq_base else 1.0
        gq_factor = (forecast['SOI_GQ_Target'][1:].sum() + forecast['GQPOP_BASE'][1:].sum()) / forecast['GQPOP_BASE'][1:].sum() if forecast['GQPOP_BASE'].sum() else 1.0
        agr_factor = (forecast['SOI_AGR_Target'][1:].sum() + base_maz['Base_AGR'][1:].sum()) / base_maz['Base_AGR'][1:].sum() if base_maz['Base_AGR'].sum() else 1.0
        minority_p = 1- forecast['POP_White_TARGET'].sum()/forecast['POP_TOT_TARGET'].sum()
        
        logger.info(f"Group quarters scaling factor: {gq_factor}")
        logger.info(f"Ag employment scaling factor: {agr_factor}")
        return gq_factor, agr_factor, minority_p
    
    def run_cube_allocation(self, dev_table: pd.DataFrame) -> pd.DataFrame:
        """Run TAZ-level allocation for Vision Year if Cube_P > 0."""
        if self.target_year == self.VISION_YEAR and self.cube_p > 0:
            logger.info("Running TAZ allocation")
            taz = 0
            taz_hu = 0
            taz_emp = 0
            
            for i in range(len(dev_table)):
                row = dev_table.iloc[i]
                if taz != row['TAZ']:
                    taz = row['TAZ']
                    taz_hu = 0
                    taz_emp = 0
                
                if (dev_table.at[i, 'DEV'] == self.NO_DEV and 
                    (taz_hu + row['HU_NET']) <= row['TAZ_HU_Target'] and 
                    (taz_emp + row['EMP_NET']) <= row['TAZ_EMP_Target']):
                    dev_table.at[i, 'DEV'] = self.target_year
                    dev_table.at[i, 'DEV_TAZ'] = 1
                    taz_hu += row['HU_NET']
                    taz_emp += row['EMP_NET']
            logger.info("TAZ allocation complete")
        return dev_table
    
    def run_agency_allocation(self, dev_table: pd.DataFrame) -> tuple:
        """Run agency-level (SOI and COMMUNITY) allocation."""
        logger.info("Running agency allocation.")
        dev_table = dev_table.sort_values(by=['SOI', 'COMMUNITY', 'TOTAL_SCORE'], ascending=[True, True, False]).reset_index(drop=True)
        dev_years = dev_table.groupby('DEV', as_index = False).agg({'HU_NET': 'sum', 'EMP_NET': 'sum'})
        next_year = dev_years[dev_years['DEV'] > self.target_year]['DEV'].min()
        logger.info(f"Sampling growth from year {next_year}")
        
        cycle = 0
        while cycle < 1:
            dev_soi = dev_table[dev_table['DEV'] <= self.target_year].groupby('SOI').agg({'HU_NET': 'sum', 'EMP_NET': 'sum'}).reset_index()
            dev_community = dev_table[dev_table['DEV'] <= self.target_year].groupby('COMMUNITY').agg({'HU_NET': 'sum', 'EMP_NET': 'sum'}).reset_index()
            dev_taz_soi = dev_table.groupby('SOI').agg({'SOI_HU_Target': 'first', 'SOI_EMP_Target': 'first'}).reset_index().merge(dev_soi, how='left', on='SOI').fillna(0)
            dev_taz_community = dev_table.groupby('COMMUNITY').agg({'SOI_HU_Target': 'first', 'SOI_EMP_Target': 'first', 'SOI_HU_P': 'first', 'SOI_EMP_P': 'first'}).reset_index().merge(dev_community, how='left', on='COMMUNITY')
            
            soi, community, soi_hu, soi_emp, soi_hu_target, soi_emp_target = '', '', 0, 0, 0, 0
            for i in range(len(dev_table)):
                row = dev_table.iloc[i]
                if soi != row['SOI']:
                    soi = row['SOI']
                    soi_hu, soi_emp = 0, 0
                    soi_hu_target = row['SOI_HU_Target'] - dev_taz_soi.loc[dev_taz_soi['SOI'] == soi]['HU_NET'].sum()
                    soi_emp_target = row['SOI_EMP_Target'] - dev_taz_soi[dev_taz_soi['SOI'] == soi]['EMP_NET'].sum()
                    
                if soi == 'Unincorporate' and row['COMMUNITY'] and community != row['COMMUNITY']:
                    community = row['COMMUNITY']
                    soi_hu, soi_emp = 0, 0
                    soi_hu_target = row['SOI_HU_P'] * (row['SOI_HU_Target'] - dev_taz_soi.loc[dev_taz_soi['SOI'] == soi, 'HU_NET'].sum())
                    soi_emp_target = row['SOI_EMP_P'] * (row['SOI_EMP_Target'] - dev_taz_soi.loc[dev_taz_soi['SOI'] == soi, 'EMP_NET'].sum())
                    
                if (soi != 'Unincoirporate' or community) and row['DEV'] == next_year and ((soi_hu + row['HU_NET']) <= max(soi_hu_target, 0)) and ((soi_emp + row['EMP_NET']) <= max(soi_emp_target, 0)):
                            dev_table.at[i, 'DEV'] = self.target_year
                            dev_table.at[i, 'DEV_SOI'] = 1
                            soi_hu += row['HU_NET']
                            soi_emp += row['EMP_NET']
            cycle =+ 1
            
        dev_table.to_csv(os.path.join(self.output_dir, "devtable.csv"), index=False)
        dev_table_final = dev_table[dev_table['DEV'] <= self.target_year]
        dev_soi = dev_table_final.groupby('SOI').agg({'HU_NET': 'sum', 'SOI_HU_Target': 'first', 'EMP_NET': 'sum', 'SOI_EMP_Target': 'first'}).reset_index()
        dev_community = dev_table_final[dev_table_final['SOI'] == 'Unincorporate'].groupby('COMMUNITY').agg({'HU_NET': 'sum', 'SOI_HU_Target': 'first', 'EMP_NET': 'sum', 'SOI_EMP_Target': 'first'}).reset_index()
        logger.info("Agency allocation complete")
        logger.debug(f"SOI allocation results:\n{dev_soi.to_string()}")
        logger.debug(f"Community allocation results:\n{dev_community.to_string()}")
        return dev_table_final, dev_soi, dev_community
    
    def calculate_parcel_growth(self, dev_table: pd.DataFrame) -> pd.DataFrame:
        """Calculate parcel-level growth values."""
        logger.info("Calculating parcel growth values")
        parcels_dev = dev_table[['parcelid']].merge(pd.read_csv(os.path.join(self.output_dir, "parcels.csv")), how='left', on='parcelid')
        
        parcels_dev['HH_NET'] = parcels_dev['HU_NET'] * (parcels_dev['SOI_OccRate'])
        parcels_dev['POP_NET'] = parcels_dev['HH_NET'] * parcels_dev['HH_SIZE']
        parcels_dev['HU_SF_NET'] = parcels_dev['ACRES'] * parcels_dev['HU_Den'] * parcels_dev['HU_SF_P'] - parcels_dev['HU_SF']
        parcels_dev['HU_MF_NET'] = parcels_dev['ACRES'] * parcels_dev['HU_Den'] * parcels_dev['HU_MF_P'] - parcels_dev['HU_MF']
        parcels_dev['HU_OTH_NET'] = parcels_dev['ACRES'] * parcels_dev['HU_Den'] * parcels_dev['HU_OTH_P'] - parcels_dev['HU_OTH']
        parcels_dev['EDU_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['EDU_P'] - parcels_dev['EMP_EDU']
        parcels_dev['FOO_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['FOO_P'] - parcels_dev['EMP_FOO']
        parcels_dev['GOV_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['GOV_P'] - parcels_dev['EMP_GOV']
        parcels_dev['IND_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['IND_P'] - parcels_dev['EMP_IND']
        parcels_dev['MED_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['MED_P'] - parcels_dev['EMP_MED']
        parcels_dev['OFC_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['OFC_P'] - parcels_dev['EMP_OFC']
        parcels_dev['OTH_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['OTH_P'] - parcels_dev['EMP_OTH']
        parcels_dev['RET_NET'] = parcels_dev['ACRES'] * parcels_dev['EMP_Den'] * parcels_dev['RET_P'] - parcels_dev['EMP_RET']
        parcels_dev['AGR_NET'] = -parcels_dev['EMP_AGR']
        parcels_dev['EMP_NET'] = (parcels_dev['EDU_NET'] + parcels_dev['FOO_NET'] + parcels_dev['GOV_NET'] + 
                                  parcels_dev['IND_NET'] + parcels_dev['MED_NET'] + parcels_dev['OFC_NET'] + 
                                  parcels_dev['OTH_NET'] + parcels_dev['RET_NET'] + parcels_dev['AGR_NET'])
        
        parcels_dev.to_csv(os.path.join(self.output_dir, "parcels_dev.csv"), index=False)
        logger.debug(f"Parcel growth calculated: {parcels_dev.shape}")
        return parcels_dev
    
    def generate_new_base_files(self, base_maz: pd.DataFrame, parcels_dev: pd.DataFrame, forecast: pd.DataFrame, gq_factor: float, agr_factor: float) -> tuple:
        """Generate new MAZ and TAZ base files."""
        logger.info("Generating new MAZ and TAZ files")
        
        # Aggregate to MAZ level
        maz_growth = parcels_dev.groupby('MAZ').agg({
            'HH_NET': 'sum', 'POP_NET': 'sum', 'HU_NET': 'sum', 'EMP_NET': 'sum', 'HU_SF_NET': 'sum', 
            'HU_OTH_NET': 'sum', 'HU_MF_NET': 'sum', 'EDU_NET': 'sum', 'FOO_NET': 'sum', 'GOV_NET': 'sum', 
            'IND_NET': 'sum', 'MED_NET': 'sum', 'OFC_NET': 'sum', 'OTH_NET': 'sum', 'RET_NET': 'sum', 
            'AGR_NET': 'sum', 'SCHL_Factor': 'first'
            }).reset_index()
        
        # Rural scaling factors
        rural_hu_base = base_maz[base_maz['SOI'] == 'Unincorporate']['Base_HU'].sum()
        rural_emp_base = base_maz[base_maz['SOI'] == 'Unincorporate']['Base_EMP'].sum()
        rural_hu_target = forecast[forecast['SOI'] == 'Unincorporate']['SOI_HU_Target'].sum()
        rural_emp_target = forecast[forecast['SOI'] == 'Unincorporate']['SOI_EMP_Target'].sum()
        dev_community_hu = parcels_dev[parcels_dev['SOI'] == 'Unincorporate'].groupby('COMMUNITY')['HU_NET'].sum().sum()
        dev_community_emp = parcels_dev[parcels_dev['SOI'] == 'Unincorporate'].groupby('COMMUNITY')['EMP_NET'].sum().sum()
        rural_hu_factor = (rural_hu_base + rural_hu_target - dev_community_hu) / rural_hu_base if rural_hu_base else 1.0
        rural_emp_factor = (rural_emp_base + rural_emp_target - dev_community_emp) / rural_emp_base if rural_emp_base else 1.0
        logger.info(f"Rural HU factor: {rural_hu_factor}, Rural EMP factor: {rural_emp_factor}")
        
        # Generate new MAZ file
        maz_new = base_maz.merge(maz_growth, how='left', on='MAZ').fillna(0)
        maz_new.loc[maz_new['SCHL_Factor'] == 0,'SCHL_Factor'] = 1
        maz_new['HU_Factor'] = 1
        maz_new.loc[maz_new['PLANNED'] == 0, 'HU_Factor'] = rural_hu_factor
        maz_new['EMP_Factor'] = 1
        maz_new.loc[maz_new['PLANNED'] == 0, 'EMP_Factor'] = rural_emp_factor
        
        maz_new['POP'] = (maz_new['HH_POP'] * maz_new['HU_Factor'] + maz_new['POP_NET']).clip(lower=0)
        maz_new['HH'] = (maz_new['Base_HH'] * maz_new['HU_Factor'] + maz_new['HH_NET']).clip(lower=0)
        maz_new['HU'] = (maz_new['Base_HU'] * maz_new['HU_Factor'] + maz_new['HU_NET']).clip(lower=0)
        maz_new['HU_SF'] = (maz_new['HU_SF'] * maz_new['HU_Factor'] + maz_new['HU_SF_NET']).clip(lower=0)
        maz_new['HU_MF'] = (maz_new['HU_MF'] * maz_new['HU_Factor'] + maz_new['HU_MF_NET']).clip(lower=0)
        maz_new['HU_OTH'] = (maz_new['HU_OTH'] * maz_new['HU_Factor'] + maz_new['HU_OTH_NET']).clip(lower=0)
        maz_new['HU'] = maz_new['HU_SF'] + maz_new['HU_MF'] + maz_new['HU_OTH']
        maz_new['EMP_EDU'] = (maz_new['Base_EDU'] * maz_new['EMP_Factor'] + maz_new['EDU_NET']).clip(lower=0)
        maz_new['EMP_FOO'] = (maz_new['Base_FOO'] * maz_new['EMP_Factor'] + maz_new['FOO_NET']).clip(lower=0)
        maz_new['EMP_GOV'] = (maz_new['Base_GOV'] * maz_new['EMP_Factor'] + maz_new['GOV_NET']).clip(lower=0)
        maz_new['EMP_IND'] = (maz_new['Base_IND'] * maz_new['EMP_Factor'] + maz_new['IND_NET']).clip(lower=0)
        maz_new['EMP_MED'] = (maz_new['Base_MED'] * maz_new['EMP_Factor'] + maz_new['MED_NET']).clip(lower=0)
        maz_new['EMP_OFC'] = (maz_new['Base_OFC'] * maz_new['EMP_Factor'] + maz_new['OFC_NET']).clip(lower=0)
        maz_new['EMP_OTH'] = (maz_new['Base_OTH'] * maz_new['EMP_Factor'] + maz_new['OTH_NET']).clip(lower=0)
        maz_new['EMP_RET'] = (maz_new['Base_RET'] * maz_new['EMP_Factor'] + maz_new['RET_NET']).clip(lower=0)
        maz_new['EMP_AGR'] = (maz_new['Base_AGR'] * agr_factor + maz_new['AGR_NET']).clip(lower=0)
        maz_new['EMP'] = (maz_new['EMP_EDU'] + maz_new['EMP_FOO'] + maz_new['EMP_GOV'] + maz_new['EMP_IND'] + 
                          maz_new['EMP_MED'] + maz_new['EMP_OFC'] + maz_new['EMP_OTH'] + maz_new['EMP_RET'] + maz_new['EMP_AGR'])
        # maz_new['DORM'] = maz_new['Base_DORM'] * gq_factor   # group quarter is needed for populationsim and is calculated in generate_popsim_inputs method below
        # maz_new['MEDICAL'] = maz_new['Base_MEDICAL'] * gq_factor
        # maz_new['PRISON'] = maz_new['Base_PRISON'] * gq_factor
        # maz_new['MILITARY'] = maz_new['Base_MILITARY'] * gq_factor
        maz_new['ELEM'] = maz_new['Base_ELEM'] * maz_new['SCHL_Factor']
        maz_new['HS'] = maz_new['Base_HS'] * maz_new['SCHL_Factor']
        maz_new['COLLEGE'] = maz_new['Base_COLLEGE'] * maz_new['SCHL_Factor']
        
        # Round and convert to integers
        for col in ['HH', 'POP', 'ELEM', 'HS', 'COLLEGE']:
            maz_new[col] = maz_new[col].round(0).astype(int)
        # for col in ['HH', 'POP', 'DORM', 'MEDICAL', 'PRISON', 'MILITARY', 'ELEM', 'HS', 'COLLEGE']:
        #     maz_new[col] = maz_new[col].round(0).astype(int)
        
        
        maz_new = maz_new[['MAZ', 'TAZ', 'SOI', 'POP', 'HH', 'HU', 'HU_SF', 'HU_MF', 'HU_OTH', 'EMP', 
                           'EMP_EDU', 'EMP_FOO', 'EMP_GOV', 'EMP_IND', 'EMP_MED', 'EMP_OFC', 'EMP_OTH', 
                           'EMP_RET', 'EMP_AGR', 'ELEM', 'HS', 'COLLEGE', 'SCHL_Factor']] # 'DORM', 'MEDICAL', 'PRISON', 'MILITARY', 
        maz_new = maz_new.sort_values(by='MAZ').reset_index(drop=True)
        maz_new.to_csv(os.path.join(self.output_dir, "maz_new.csv"), index=False)

        # Generate new TAZ file
        taz_new = maz_new.groupby('TAZ').agg({
            'SOI': 'first', 'POP': 'sum', 'HH': 'sum', 'HU': 'sum', 'HU_SF': 'sum', 'HU_MF': 'sum', 'HU_OTH': 'sum',
            'EMP': 'sum', 'EMP_EDU': 'sum', 'EMP_FOO': 'sum', 'EMP_GOV': 'sum', 'EMP_IND': 'sum', 'EMP_MED': 'sum',
            'EMP_OFC': 'sum', 'EMP_OTH': 'sum', 'EMP_RET': 'sum', 'EMP_AGR': 'sum', 'ELEM': 'sum', 'HS': 'sum', 'COLLEGE': 'sum'
        }).reset_index() # 'DORM': 'sum', 'MEDICAL': 'sum', 'PRISON': 'sum', 'MILITARY': 'sum', 
        taz_new.to_csv(os.path.join(self.output_dir, "taz_new.csv"), index=False)
        
        logger.debug(f"New MAZ file: {maz_new.shape}, New TAZ file: {taz_new.shape}")
        return maz_new, taz_new
    
    def generate_popsim_inputs(self, maz_new: pd.DataFrame, taz_new: pd.DataFrame, gq_maz: pd.DataFrame, gq_factor: float, minority_p: float) -> None:
        """Generate input files for PopSim."""
        logger.info("Generating PopSim input files")
        
        # mazData
        maz_data = maz_new[['MAZ', 'HH']].rename(columns={'HH': '2014 HH'})
        maz_data.to_csv(os.path.join(self.popsim_dir, "mazData.csv"), index=False)
        
        # tazData
        taz_data = taz_new[['TAZ', 'HU_SF', 'HU_MF', 'HU_OTH']]
        taz_data.to_csv(os.path.join(self.popsim_dir, "tazData.csv"), index=False)
        
        # gq_maz
        # base_gq_noninst = pd.read_csv(os.path.join(self.data_popsim_dir, "gq_maz.csv"))['othnon14'].sum()
        # gq_maz = maz_new[['MAZ', 'DORM', 'MILITARY', 'MEDICAL']].rename(columns={'DORM': 'univ14', 'MILITARY': 'mil14', 'MEDICAL': 'othnon14'})
        # gq_maz['othnon14'] = (gq_maz['othnon14'] * base_gq_noninst / maz_new['MEDICAL'].sum()).round(0).astype(int) if maz_new['MEDICAL'].sum() else 0
        # Apply group quarter population growth factor to the base file
        gq_maz[['univ14', 'mil14', 'othnon14']] = gq_maz[['univ14', 'mil14', 'othnon14']] * gq_factor
        # Round and convert to integers
        for col in ['univ14', 'mil14', 'othnon14']:
            gq_maz[col] = gq_maz[col].round(0).astype(int)
        gq_maz.to_csv(os.path.join(self.popsim_dir, "gq_maz.csv"), index=False)
        
        # countyData
        county_data = pd.DataFrame({'county': ['FRESNO'], 'hhpop': [maz_new['POP'].sum()], 'minp': [minority_p]})
        county_data.to_csv(os.path.join(self.popsim_dir, "countyData.csv"), index=False)
        
        logger.info(f"PopSim files generated: mazData, tazData, gq_maz, countyData")
        
def main():
    logger.info("\n--- LAND USE ALLOCATION ---")
    logger.info(f"Start time: {datetime.now()}")
    
    try:
        param_file = sys.argv[1]
        target_year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        allocator = LandUseAllocator(param_file, target_year)
        
        logger.info(f"--- YEAR {allocator.target_year}")
        
        # Load data
        base_maz, forecast, dev_table, gq_maz = allocator.load_data()
        
        # Calculate scaling factors
        gq_factor, agr_factor, minority_p = allocator.calculate_scaling_factors(base_maz, forecast)
        
        # Run allocations
        dev_table = allocator.run_cube_allocation(dev_table)
        dev_table_final, dev_soi, dev_community = allocator.run_agency_allocation(dev_table)
        
        # Calculate parcel growth
        parcels_dev = allocator.calculate_parcel_growth(dev_table_final)
        
        # Generate new base files
        maz_new, taz_new = allocator.generate_new_base_files(base_maz, parcels_dev, forecast, gq_factor, agr_factor)
        
        # Generate PopSim inputs
        allocator.generate_popsim_inputs(maz_new, taz_new, gq_maz, gq_factor, minority_p)
        
        logger.info("\n--- Script ran sucessfully! ---")
    except Exception as e:
        logger.error(f"\n--- Error occured: {str(e)}")
        raise
    
    logger.info(f"End time: {datetime.now()}")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    