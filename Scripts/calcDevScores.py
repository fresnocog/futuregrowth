from datetime import datetime
import numpy as np
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

baseYear = int(parameters[parameters.Key == 'baseYear']['Value'].item().strip(' '))
Cube_P = float(parameters[parameters.Key == 'Cube_P']['Value'].item().strip(' '))
wtInfill = float(parameters[parameters.Key == 'wtInfill']['Value'].item().strip(' '))
wtCons = float(parameters[parameters.Key == 'wtCons']['Value'].item().strip(' '))
wtDensity = float(parameters[parameters.Key == 'wtDensity']['Value'].item().strip(' '))
wtBike = float(parameters[parameters.Key == 'wtBike']['Value'].item().strip(' '))
wtTransit = float(parameters[parameters.Key == 'wtTransit']['Value'].item().strip(' '))
wtSOV = float(parameters[parameters.Key == 'wtSOV']['Value'].item().strip(' '))
wtVMT = float(parameters[parameters.Key == 'wtVMT']['Value'].item().strip(' '))
penaltyInfill = float(parameters[parameters.Key == 'penaltyInfill']['Value'].item().strip(' '))
penaltyRedev = float(parameters[parameters.Key == 'penaltyRedev']['Value'].item().strip(' '))
penaltyDensity = float(parameters[parameters.Key == 'penaltyDensity']['Value'].item().strip(' '))
adjSF = float(parameters[parameters.Key == 'adjSF']['Value'].item().strip(' '))
adjMU = float(parameters[parameters.Key == 'adjMU']['Value'].item().strip(' '))
adjTOD = float(parameters[parameters.Key == 'adjTOD']['Value'].item().strip(' '))
adjDT = float(parameters[parameters.Key == 'adjDT']['Value'].item().strip(' '))
adjResDen = float(parameters[parameters.Key == 'adjRESDEN']['Value'].item().strip(' '))
adjEmpDen = float(parameters[parameters.Key == 'adjEMPDEN']['Value'].item().strip(' '))
HiDenPercentile = float(parameters[parameters.Key == 'HiDenPercentile']['Value'].item().strip(' '))
RedevMinDen = float(parameters[parameters.Key == 'RedevMinDen']['Value'].item().strip(' '))

try:
    targetYear = int(sys.argv[2])
    try:
        overrideDev = sys.argv[3]
        keep_devtable = False
    except:
        keep_devtable = True
except:
    targetYear = int(parameters[parameters.Key == 'targetYear']['Value'].item().strip(' '))
    if targetYear == VISION_YEAR:
        keep_devtable = False
    else:
        keep_devtable = True

print('\r\n--- DEVELOPMENT SCORE CALCULATION ---')
print('Start time '+str(datetime.now()))
print("\r\n--- YEAR " + str(targetYear) + " ---\r\n")

Base_MAZ = pd.read_csv(os.path.join(dataDir, "Base_MAZ_2019.csv"))
Forecast = pd.read_csv(os.path.join(outputDir, "controls_soi.csv"))
Cube_Growth = pd.read_csv(os.path.join(outputDir, "controls_taz.csv"))


#############################################################################
##   Calculate Development Score (first iteration only)
#############################################################################
# Calculate static base scores
if targetYear != VISION_YEAR:
    Parcels = pd.read_csv(os.path.join(outputDir, "parcels.csv"))
else:
    Parcels = pd.read_csv(os.path.join(dataDir, "Parcel_Data.csv"))#.dropna()
    Parcels = Parcels[Parcels['SOI']!=' ']
    Parcels['DEVTYPE'] = Parcels['DEVTYPE_MI']
    Parcels.loc[Parcels['parcelid'].isin(random.sample(list(Parcels['parcelid']), k = round(HiDenPercentile*len(Parcels)))) ,'DEVTYPE'] = Parcels['DEVTYPE_HI']    # replace original random process:  Parcels.loc[Parcels['parcelid']%100 <= HiDenPercentile*100,'DEVTYPE'] = Parcels['DEVTYPE_HI'] 
    DevTypes = pd.read_csv(os.path.join(dataDir, "DevTypes.csv"))
    Parcels = Parcels.merge(DevTypes, how = 'left', on = 'DEVTYPE')
    Parcels = Parcels.merge(Forecast.filter(items=['SOI','VacRate','HH_SIZE','SCHL_Factor']), how = 'left', on = 'SOI')
    Parcels['HU_Den'] = Parcels['HU_Den']*(1+adjResDen)
    Parcels['EMP_Den'] = Parcels['EMP_Den']*(1+adjEmpDen)
    Parcels['Net_Density'] = Parcels['HU_Den']+Parcels['EMP_Den']-(Parcels['HU']+Parcels['EMP']-Parcels['EMP_AGR'])/Parcels['ACRES']
    Parcels['HU_NET'] = Parcels['HU_Den']*Parcels['ACRES']-Parcels['HU']
    Parcels['EMP_NET'] = Parcels['EMP_Den']*Parcels['ACRES']-Parcels['EMP']+Parcels['EMP_AGR']
    Parcels['NET_GROWTH'] = Parcels['HU_NET']+Parcels['EMP_NET']
    Parcels['Developed'] = (Parcels['Vacant']+1)%2
    #Parcels['Incorporated'] = (Parcels['SOI'] != 'Unincorporate')
    Parcels['Incorporated'] = 0
    Parcels.loc[Parcels['SOI'] != 'Unincorporate','Incorporated'] = 1
    #Parcels[Parcels['SOI']!='Unincorporate']['Incorporated'] = 1

    Parcels = Parcels[Parcels['Net_Density']-RedevMinDen*Parcels['Developed']>0]
    Parcels['Cons_GW'] = (1-Parcels['GW_IDX']).clip(0,1)
    #Parcels['Cons_F'] = (1-Parcels['FMMP_P']-Parcels['FMMP_L']-Parcels['FMMP_S']-Parcels['FMMP_U']).clip(0,1)
    Parcels['Cons_F'] = (1-(Parcels['FMMP_L']+Parcels['FMMP_P']+Parcels['FMMP_S']+Parcels['FMMP_U'])/2-(1-Parcels['Incorporated'])*(Parcels['FMMP_P']+Parcels['FMMP_S']+Parcels['FMMP_U'])).clip(0,1)

    #Parcels['IDX_Cons'] = (Parcels['Developed'] + Parcels['Incorporated']/2 + Parcels[["Cons_GW", "Cons_F"]].min(axis=1)).clip(0,1)
    Parcels['IDX_Cons'] = Parcels[["Cons_GW", "Cons_F"]].min(axis=1)
    Parcels['IDX_Infill'] = (1-Parcels['Infill']/10560).clip(0,1)
    #Parcels = Parcels[:].filter(items=['parcelid','HU','EMP','ACRES','DEVTYPE_MI']).dropna()
    #Parcels.rename(columns={'DEVTYPE_MI':'DEVTYPE'}, inplace=True)

    Parcels['IDX_Den'] = (Parcels['Net_Density']/Parcels['Net_Density'].max()).clip(0,1)
    Parcels['IDX_MU'] = Parcels['MU']
    Parcels['IDX_SF'] = Parcels['HU_SF_P']

    Parcels['BASE_SCORE'] = Parcels['IDX_Cons']*wtCons+Parcels['IDX_Infill']*wtInfill+Parcels['IDX_Den']*wtDensity+Parcels['IDX_VMT']*wtVMT

    Parcels.to_csv(os.path.join(outputDir,"parcels.csv"), index = False)
    print("Indexes calculated")


#############################################################################
##   Get skim results
#############################################################################
# Merge MAZ skims
Skims_MAZ = pd.read_csv(os.path.join(outputDir, "skims_maz.csv")).filter(items=['MAZ','IDX_Bike'])
Parcels = Parcels.merge(Skims_MAZ, how = 'left', on = 'MAZ')
del Skims_MAZ

# Merge TAZ skims
Skims_TAZ = pd.read_csv(os.path.join(outputDir, "skims_taz.csv")).filter(items=['TAZ','IDX_Transit','IDX_SOV'])
Parcels = Parcels.merge(Skims_TAZ, how = 'left', on = 'TAZ')
del Skims_TAZ


#############################################################################
##   Calculate development scores
#############################################################################
Parcels['TOTAL_SCORE'] = Parcels['BASE_SCORE'] + Parcels['IDX_Bike']*wtBike + Parcels['IDX_Transit']*wtTransit + Parcels['IDX_SOV']*wtSOV
Parcels['TOTAL_SCORE'] = Parcels['TOTAL_SCORE']*Parcels['SCORE_ADJ']*(1-penaltyRedev*Parcels['Developed'])*(1+adjSF*Parcels['IDX_SF'])*(1+adjMU*Parcels['IDX_MU'])*(1+adjTOD*Parcels['TOD'])*(1+adjDT*Parcels['DT'])  # Geometric adjustment
#Parcels['TOTAL_SCORE'] = Parcels['TOTAL_SCORE']*(Parcels['SCORE_ADJ']-penaltyRedev*Parcels['Developed']+adjTOD*Parcels['TOD']+adjDT*Parcels['DT']-adjVMT*Parcels['IDX_VMT'])                # Arithmetic adjustment
Parcels.loc[Parcels['HU_Den'] > 0,'TOTAL_SCORE'] = Parcels['TOTAL_SCORE']*(1-penaltyDensity*Parcels['IDX_Den'])

if targetYear >= VISION_YEAR:
    Parcels.loc[Parcels['Infill'] == 0,'TOTAL_SCORE'] = Parcels['TOTAL_SCORE']*(1-penaltyInfill)
Parcels = Parcels.filter(items=['parcelid','SOI','COMMUNITY','TAZ','HU_NET','EMP_NET','VacRate','HH_SIZE','BASE_SCORE','IDX_Bike','IDX_Transit','IDX_SOV','TOTAL_SCORE'])
Parcels.to_csv(os.path.join(outputDir,"parcels_Final.csv"), index = False)
#print("Final scores calculated")


#############################################################################
##   Create development table
#############################################################################
Forecast_dev = Forecast.filter(items=['SOI','SOI_HU_Target','SOI_EMP_Target'])
#Forecast_dev.rename(columns={'SOI':'AGENCY'}, inplace=True)

if not keep_devtable:
    DevTable = Parcels.merge(Cube_Growth, how = 'left', on = 'TAZ')
    DevTable = DevTable.filter(items=['parcelid','SOI','COMMUNITY','TAZ','HU_NET','EMP_NET','TOTAL_SCORE','TAZ_HU_Target','TAZ_EMP_Target','SOI_HU_Target','SOI_EMP_Target','SOI_HU_P','SOI_EMP_P'])#.dropna()
    DevTable['DEV'] = NO_DEV
    DevTable['DEV_TAZ'] = 0
    DevTable['DEV_SOI'] = 0
else:
    DevTable = pd.read_csv(os.path.join(outputDir, "devtable.csv"))
    del DevTable['TOTAL_SCORE']
    del DevTable['SOI_HU_Target']
    del DevTable['SOI_EMP_Target']
    DevTable = DevTable.merge(Parcels.filter(items=['parcelid','TOTAL_SCORE']), how = 'left', on = 'parcelid')


DevTable = DevTable.merge(Forecast_dev, how = 'left', on = 'SOI')
DevTable = DevTable.sort_values(by=['TAZ', 'TOTAL_SCORE'], ascending=[True, False]).reset_index(drop=True)

DevTable.to_csv(os.path.join(outputDir,"devtable.csv"), index = False)
print('Development table generated for year ' + str(targetYear))


#############################################################################
##   End of script
#############################################################################

print('\r\n--- Script ran successfully! ---\r\n')
print('End time '+str(datetime.now()))
