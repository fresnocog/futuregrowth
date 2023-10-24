from datetime import datetime
import openmatrix as omx
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import sys
import math
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
emp_array_maz = Base_MAZ['Base_EMP'].to_numpy()

# Calculate bike accessibility by MAZ
maz_skims['bike_skim'] = 0.0
bikeSkim_omx = omx.open_file(os.path.join(dataABM_Dir, "FC" + str(baseYear)[-2:] + "_BASE_MAZ_SKM_BIKE.omx"))  # This should be updated to "last year" instead of "base year"
bikeSkims = bikeSkim_omx['DIST_BIKE']

for i in maz_skims.index:
    row = bikeSkims[i]
    mask = ma.masked_where(row<=5, row).mask                 # select where distance <= 5miles  about 30 minutes by bike               #mask = ma.masked_where(row>0, row).mask
    mask1 = ma.masked_where(row>0, row).mask
    skim_val = (mask1 * mask*emp_array_maz).sum()                    # Calculate cumulative oppotunities in 30 minutes by bike                 # skim_val = (mask*emp_array_maz/(row+0.01)).sum()
    maz_skims.at[i,'bike_skim'] = skim_val + emp_array_taz[i]

del bikeSkims
bikeSkim_omx.close()

# Calculate bike skim indexes
maz_skims['IDX_Bike'] = 0.0
maz_skims.loc[maz_skims['bike_skim']>0,'IDX_Bike'] = maz_skims[maz_skims['bike_skim']>0].bike_skim.rank(pct = True)
#maz_skims['IDX_Bike'] = maz_skims.bike_skim.rank(pct = True)

#maz_skims['IDX_Bike'] = 0.0
#bike_max = maz_skims['bike_skim'].max()
#bike_min = maz_skims['bike_skim'].min()
#maz_skims['IDX_Bike'] = ((maz_skims['bike_skim']-bike_min)/(bike_max-bike_min))#.clip(0,1)
#print("Bike skim range    >> ", bike_min, bike_max)

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
emp_array_taz = taz_skims['Base_EMP'].to_numpy()

# Calculate transit accessibility by TAZ
taz_skims['transit_skim'] = 0.0
transitSkim_omx = omx.open_file(os.path.join(dataABM_Dir, "FC" + str(baseYear)[-2:] + "_BASE_SKM_PK_TWB.omx"))  # This should be updated to "last year" instead of "base year"
transitSkims = transitSkim_omx['IVTT'] + transitSkim_omx['WLK_P'] + transitSkim_omx['WLK_A'] + transitSkim_omx['WLK_X'] + transitSkim_omx['IWAIT'] + transitSkim_omx['XWAIT']  # In-vehicle travel time

for i in taz_skims.index:
    row = transitSkims[i]
    mask = ma.masked_where(row<=60, row).mask                 # mask = ma.masked_where(row>0, row).mask
    mask1 = ma.masked_where(row>0, row).mask                 # mask = ma.masked_where(row>0, row).mask

    skim_val =(mask*mask1*emp_array_taz).sum()          # cumulative job opportunities within 60 minutes. (incude wait time and walk to bus stop time)   # skim_val =(mask*emp_array_taz/(row+.01)).sum()
    taz_skims.at[i,'transit_skim'] = skim_val + emp_array_taz[i]
    #print(i,row,skim_val)

del transitSkims
transitSkim_omx.close()

# Calculate transit skim indexes
taz_skims['IDX_Transit'] = 0.0
taz_skims.loc[taz_skims['transit_skim']>0,'IDX_Transit'] = taz_skims[taz_skims['transit_skim']>0].transit_skim.rank(pct = True)
#taz_skims['IDX_Transit'] = taz_skims.transit_skim.rank(pct = True)

#taz_skims['IDX_Transit'] = 0.0
#transit_max = taz_skims['transit_skim'].max()
#transit_min = taz_skims['transit_skim'].min()
#taz_skims['IDX_Transit'] = ((taz_skims['transit_skim']-transit_min)/(transit_max-transit_min))#.clip(0,1)
#print("Transit skim range    >> ", transit_min, transit_max)


# Calculate SOV accessibility by TAZ
taz_skims['sov_skim'] = 0.0
sovSkim_omx = omx.open_file(os.path.join(dataABM_Dir, "FC" + str(baseYear)[-2:] + "_BASE_SKM_PK_D1.omx"))  # This should be updated to "last year" instead of "base year"
sovSkims = sovSkim_omx['TIME_1Veh']  # In-vehicle travel time   # GENTIME_1Veh: Generalized time for zone pair for medium value-of-time group; TIME_1Veh:Congested travel time (mins) for zone pair for medium value-oftime group

for i in taz_skims.index:
    row = sovSkims[i]
    mask = ma.masked_where(row>0, row).mask
    skim_val =  (mask * math.exp(-0.049601 * emp_arry_taz)).sum        # gravity model,  expotential decay weights           #before: inverse power beta = -1, skim_val =(mask*emp_array_taz/(row+.01)).sum()
    taz_skims.at[i,'sov_skim'] = skim_val  + emp_array_taz[i]
    #print(i,row,skim_val)

del sovSkims
sovSkim_omx.close()

# Calculate sov skim indexes
taz_skims['IDX_SOV'] = 0.0
taz_skims.loc[taz_skims['sov_skim']>0,'IDX_SOV'] = taz_skims[taz_skims['sov_skim']>0].sov_skim.rank(pct = True)
#taz_skims['IDX_SOV'] = taz_skims.sov_skim.rank(pct = True)

#taz_skims['IDX_SOV'] = 0.0
#sov_max = taz_skims['sov_skim'].max()
#sov_min = taz_skims['sov_skim'].min()
#taz_skims['IDX_SOV'] = ((taz_skims['sov_skim']-sov_min)/(sov_max-sov_min))#.clip(0,1)
#print("SOV skim range    >> ", sov_min, sov_max)

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
