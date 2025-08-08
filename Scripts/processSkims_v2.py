from datetime import datetime
import openmatrix as omx
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import sys

#############################################################################
##   Load files
#############################################################################
VISION_YEAR = 2035
NO_DEV = 9999

parameters_file = sys.argv[1]
parameters = pd.read_csv(parameters_file)
parameters.columns = ['Key', 'Value', 'Notes']

WORKING_DIR = parameters[parameters.Key == 'WORKING_DIR']['Value'].item().strip(' ')
dataDir = os.path.join(WORKING_DIR, 'Data')
dataPopSim_Dir = os.path.join(dataDir, 'PopSim')
dataABM_Dir = os.path.join(dataDir, 'ABM')
outputDir = os.path.join(WORKING_DIR, 'Setup', 'Data')
popsimDir = os.path.join(WORKING_DIR, 'Setup', 'Outputs')

#####Pankaj: read FresnoABM Directory
FresnoABM_DIR = parameters[parameters.Key == 'FresnoABM_DIR']['Value'].item().strip(' ')

baseYear = int(parameters[parameters.Key == 'baseYear']['Value'].item().strip(' '))
try:
    targetYear = int(sys.argv[2])
except:
    targetYear = int(parameters[parameters.Key == 'targetYear']['Value'].item().strip(' '))

print('\r\n--- SKIM PROCESSING ---')
print('Start time '+str(datetime.now()))
print("\r\n--- YEAR " + str(targetYear) + " ---\r\n")

Base_MAZ = pd.read_csv(os.path.join(dataDir, "Base_MAZ_2019.csv"))


#############################################################################
##   Process MAZ skims (bike)
#############################################################################
maz_skims = Base_MAZ.filter(items=['MAZ','TAZ','SOI'])
#maz_skims.set_index('MAZ',inplace=True)
emp_array_maz = np.array(Base_MAZ['Base_EMP'])
HU_array_maz = np.array(Base_MAZ['Base_HU'])

#####Pankaj: update baseYear if targetYear is not 2025 and create variables for respective directory for skims update dataABM_dir to skim_Dir in the subsequent codes

if targetYear == 2025:
    skim_Dir = os.path.join(FresnoABM_DIR, "FC" + str(baseYear)[-2:] + "_BASE", "01_Skims")
else:
    baseYear = targetYear - 5
    skim_Dir = os.path.join(FresnoABM_DIR, "FC" + str(baseYear)[-2:] + "_BASE", "01_Skims")

# Calculate bike accessibility by MAZ
maz_skims['bike_skim'] = 0.0
try:
            with omx.open_file(os.path.join(self.abm_dir, f"FC{str(self.base_year)[-2:]}_BASE_MAZ_SKM_BIKE.omx")) as bike_skim_omx:
                bike_skims = bike_skim_omx['TIME_BIKE'] #['DIST_BIKE']
                for i in maz_skims.index:
                    row = bike_skims[i]
                    mask = ma.masked_where(row <= 5, row).mask  # Distance <= 5 miles (`30 min by bike)
                    mask1 = ma.masked_where(row > 0, row).mask
                    skim_val = (mask1 * mask * emp_array_maz).sum()
                    maz_skims.at[i, 'bike_skim'] = skim_val + emp_array_maz[i]     
                    skim_val_EMP = (mask1 * mask * HU_array_maz).sum()                 # for emp allocation scoring 
                    maz_skims.at[i, 'bike_skim_EMP'] = skim_val_EMP + HU_array_maz[i]   # for emp allocation scoring 
        except Exception as e:
            logger.error(f"Error processing bike skims: {str(e)}")
            raise
#### Old script start ####
#bikeSkim_omx = omx.open_file(os.path.join(skim_Dir, "FC" + str(baseYear)[-2:] + "_BASE_MAZ_SKM_BIKE.omx"))  # This should be updated to "last year" instead of "base year"
#bikeSkims = bikeSkim_omx['DIST_BIKE']
#
#for i in maz_skims.index:
#    row = bikeSkims[i]
#    mask = ma.masked_where(row>0, row).mask
#    skim_val = (mask*emp_array_maz/(row+0.01)).sum()
#    maz_skims.at[i,'bike_skim'] = skim_val
#
#del bikeSkims
#bikeSkim_omx.close()
#### Old script end ####

# Calculate bike skim indexes
logger.info("Calculating bike skim indexes")
maz_skims['IDX_Bike'] = 0.0 
valid_bike = maz_skims['bike_skim'] > 0
maz_skims.loc[valid_bike, 'IDX_Bike'] = maz_skims[valid_bike]['bike_skim'].rank(pct=True)
maz_skims['IDX_Bike_EMP'] = 0.0                                                                                   # for emp allocation scoring 
valid_bike_EMP = maz_skims['bike_skim_EMP'] > 0                                                                   # for emp allocation scoring    
maz_skims.loc[valid_bike_EMP, 'IDX_Bike_EMP'] = maz_skims[valid_bike_EMP]['bike_skim_EMP'].rank(pct=True)         # for emp allocation scoring 
logger.debug(f"Bike skim range: min={maz_skims['bike_skim'].min()}, max={maz_skims['bike_skim'].max()}")  #

return maz_skims
#### Old script ###############       
#maz_skims['IDX_Bike'] = 0.0
#maz_skims.loc[maz_skims['bike_skim']>0,'IDX_Bike'] = maz_skims[maz_skims['bike_skim']>0].bike_skim.rank(pct = True)
##maz_skims['IDX_Bike'] = maz_skims.bike_skim.rank(pct = True)
## need a mask for distance <= 5 miles (30 minuts by bike) 'DIST_BIKE' < = 5 (<double check the latest script is using DIST>)
##maz_skims['IDX_Bike'] = 0.0
##bike_max = maz_skims['bike_skim'].max()
##bike_min = maz_skims['bike_skim'].min()
##maz_skims['IDX_Bike'] = ((maz_skims['bike_skim']-bike_min)/(bike_max-bike_min))#.clip(0,1)
##print("Bike skim range    >> ", bike_min, bike_max)
#### Old script ###############       

# Export MAZ skims
try:
    maz_skims.to_csv(os.path.join(outputDir,"skims_maz.csv"), index = False)
except:
    print("ERROR:",os.path.join(outputDir,"skims_maz.csv"),"could not be created")

#############################################################################
##   Process TAZ skims (Transit, SOV)
#############################################################################
taz_skims = pd.DataFrame(list(range(1,3001)),columns=['TAZ'])
taz_skims = taz_skims.merge(Base_MAZ.groupby(['TAZ']).agg({'SOI':'first','Base_EMP':'sum'}), how = 'left', on = 'TAZ')
#taz_skims.set_index('TAZ',inplace=True)
taz_skims['SOI'].fillna("",inplace=True)
taz_skims['Base_EMP'].fillna(0,inplace=True)
taz_skims['Base_HU'].fillna(0,inplace=True)
emp_array_taz = taz_skims['Base_EMP'].to_numpy()
HU_array_taz = taz_skims['Base_HU'].to_numpy()

# Calculate transit accessibility by TAZ
#### Old script begin#######
#taz_skims['transit_skim'] = 0.0
#transitSkim_omx = omx.open_file(os.path.join(skim_Dir, "FC" + str(baseYear)[-2:] + "_BASE_SKM_PK_TWB.omx"))  # This should be updated to "last year" instead of "base year"
#transitSkims = transitSkim_omx['IVTT']  # In-vehicle travel time
#
#for i in taz_skims.index:
#    row = transitSkims[i]
#    mask = ma.masked_where(row>0, row).mask
#    skim_val =(mask*emp_array_taz/(row+.01)).sum()
#    taz_skims.at[i,'transit_skim'] = skim_val
#    #print(i,row,skim_val)
#
#del transitSkims
#transitSkim_omx.close()
#### Old script end#######
# SF: Need a mask exclude where  sum('IVTT', 'WLK_P', 'WLK_A', 'WLK_X','IWAIT','XWAIT' ) < 60, total time of access to transit, wait time, time on transit < 60 minutes
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
                    skim_val_EMP = (mask * mask1 * HU_array_taz).sum()     # for emp allocation scoring 
                    taz_skims.at[i, 'transit_skim'] = skim_val
                    taz_skims.at[i, 'transit_skim_EMP'] = skim_val_EMP     # for emp allocation scoring 
        except Exception as e:
            logger.error(f"Error processing transit skims: {str(e)}")
            raise

# Calculate transit skim indexes
logger.info("Calculating transit skim indexes")
taz_skims['IDX_Transit'] = 0.0
valid_transit = taz_skims['transit_skim'] > 0
taz_skims.loc[valid_transit, 'IDX_Transit'] = taz_skims[valid_transit]['transit_skim'].rank(pct=True)
taz_skims['IDX_Transit_EMP'] = 0.0                                                                                              # for emp allocation scoring 
valid_transit_EMP = taz_skims['transit_skim_EMP'] > 0                                                                           # for emp allocation scoring 
taz_skims.loc[valid_transit_EMP, 'IDX_Transit_EMP'] = taz_skims[valid_transit_EMP]['transit_skim_EMP'].rank(pct=True)           # for emp allocation scoring 
logger.debug(f"Transit skim range: min={taz_skims['transit_skim'].min()}, max={taz_skims['transit_skim'].max()}")
        
#taz_skims.loc[taz_skims['transit_skim']>0,'IDX_Transit'] = taz_skims[taz_skims['transit_skim']>0].transit_skim.rank(pct = True)
#taz_skims['IDX_Transit'] = taz_skims.transit_skim.rank(pct = True)

#taz_skims['IDX_Transit'] = 0.0
#transit_max = taz_skims['transit_skim'].max()
#transit_min = taz_skims['transit_skim'].min()
#taz_skims['IDX_Transit'] = ((taz_skims['transit_skim']-transit_min)/(transit_max-transit_min))#.clip(0,1)
#print("Transit skim range    >> ", transit_min, transit_max)


# Calculate SOV accessibility by TAZ
taz_skims['sov_skim'] = 0.0
try:
            with omx.open_file(os.path.join(self.abm_dir, f"FC{str(self.base_year)[-2:]}_BASE_SKM_PM_D1.omx")) as sov_skim_omx: # FC{str(self.base_year)[-2:]}_BASE_SKM_PK_D1.omx
                sov_skims = sov_skim_omx['TIME_1Veh']
                for i in taz_skims.index:
                    row = sov_skims[i]
                    mask = ma.masked_where(row > 0, row).mask
                    skim_val = (mask * emp_array_taz * np.exp(-0.049601 * row)).sum()
                    skim_val_EMP = (mask * HU_array_taz * np.exp(-0.049601 * row)).sum()        # for emp allocation scoring 
                    taz_skims.at[i, 'sov_skim'] = skim_val + emp_array_taz[i]
                    taz_skims.at[i, 'sov_skim_EMP'] = skim_val_EMP + HU_array_taz[i]            # for emp allocation scoring 

        except Exception as e:
            logger.error(f"Error processing SOV skims: {str(e)}")
            raise
#### Old script start ####
#sovSkim_omx = omx.open_file(os.path.join(skim_Dir, "FC" + str(baseYear)[-2:] + "_BASE_SKM_PK_D1.omx"))  # This should be updated to "last year" instead of "base year"
#sovSkims = sovSkim_omx['GENTIME_1Veh']  # In-vehicle travel time
#
#for i in taz_skims.index:
#    row = sovSkims[i]
#    mask = ma.masked_where(row>0, row).mask
#    skim_val =(mask*emp_array_taz/(row+.01)).sum()
#    taz_skims.at[i,'sov_skim'] = skim_val
#    #print(i,row,skim_val)
#
#del sovSkims
#sovSkim_omx.close()
#### Old script end ####
# Calculate sov skim indexes
logger.info("Calculating SOV skim indexes")
taz_skims['IDX_SOV'] = 0.0
valid_sov = taz_skims['sov_skim'] > 0
taz_skims.loc[valid_sov, 'IDX_SOV'] = taz_skims[valid_sov]['sov_skim'].rank(pct=True)
taz_skims['IDX_SOV_EMP'] = 0.0                                                                                    # for emp allocation scoring 
valid_sov_EMP = taz_skims['sov_skim_EMP'] > 0                                                                     # for emp allocation scoring 
taz_skims.loc[valid_sov_EMP, 'IDX_SOV_EMP'] = taz_skims[valid_sov_EMP]['sov_skim_EMP'].rank(pct=True)             # for emp allocation scoring 
logger.debug(f"SOV skim range: min={taz_skims['sov_skim'].min()}, max={taz_skims['sov_skim'].max()}")
        
return taz_skims
#### Old script ####
#taz_skims['IDX_SOV'] = 0.0
#taz_skims.loc[taz_skims['sov_skim']>0,'IDX_SOV'] = taz_skims[taz_skims['sov_skim']>0].sov_skim.rank(pct = True)
## update to the latest version: mask * emp_array_taz * np.exp(-0.049601 * 'TIME_1Veh')
##taz_skims['IDX_SOV'] = taz_skims.sov_skim.rank(pct = True)
#
##taz_skims['IDX_SOV'] = 0.0
##sov_max = taz_skims['sov_skim'].max()
##sov_min = taz_skims['sov_skim'].min()
##taz_skims['IDX_SOV'] = ((taz_skims['sov_skim']-sov_min)/(sov_max-sov_min))#.clip(0,1)
##print("SOV skim range    >> ", sov_min, sov_max)
#### Old script ####

# Export TAZ skims
try:
    taz_skims.to_csv(os.path.join(outputDir,"skims_taz.csv"), index = False)
except:
    print("ERROR:",os.path.join(outputDir,"skims_taz.csv"),"could not be created")


#############################################################################
##   End of script
#############################################################################

print('\r\n--- Script ran successfully! ---\r\n')
print('End time '+str(datetime.now()))



