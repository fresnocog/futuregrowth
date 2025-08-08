from datetime import datetime
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

try:
    targetYear = int(sys.argv[2])
except:
    targetYear = int(parameters[parameters.Key == 'targetYear']['Value'].item().strip(' '))

print('\r\n--- LAND USE ALLOCATION ---')
print('Start time '+str(datetime.now()))
print("\r\n--- YEAR " + str(targetYear) + " ---\r\n")

Base_MAZ = pd.read_csv(os.path.join(dataDir, "Base_MAZ_2019.csv"))
Forecast = pd.read_csv(os.path.join(outputDir, "controls_soi.csv"))
DevTable = pd.read_csv(os.path.join(outputDir, "devtable.csv"))

gq_base = Base_MAZ['Base_DORM'].sum()+Base_MAZ['Base_MEDICAL'].sum()+Base_MAZ['Base_PRISON'].sum()+Base_MAZ['Base_MILITARY'].sum()
gq_factor = (Forecast['SOI_GQ_Target'].sum()+gq_base)/gq_base
agr_factor = (Forecast['SOI_AGR_Target'].sum()+Base_MAZ['Base_AGR'].sum())/Base_MAZ['Base_AGR'].sum()
minority_p = 1- Forecast['POP_White_TARGET'].sum()/Forecast['POP_TOT_TARGET'].sum()
print("Group quarters scaling factor: " + str(gq_factor))
print("Ag employment scaling factor:  " + str(agr_factor))
print("Regional minority percentage:  " + str(minority_p))


#############################################################################
##   Run Cube allocation (Vision Year only)
#############################################################################
if targetYear == VISION_YEAR and Cube_P > 0:
    print('\r\nRunning TAZ allocation...')
    TAZ = 0
    TAZ_HU = 0
    TAZ_EMP = 0

    for i in range(len(DevTable)):
        row = DevTable.iloc[i]
        # Check for new TAZ
        if TAZ != row['TAZ']:
            TAZ = row['TAZ']
            TAZ_HU = 0
            TAZ_EMP = 0

        if DevTable.at[i,'DEV'] == NO_DEV and ((TAZ_HU+row['HU_NET']) <= row['TAZ_HU_Target']) and ((TAZ_EMP+row['EMP_NET'])<=row['TAZ_EMP_Target']):
            DevTable.at[i,'DEV'] = targetYear
            DevTable.at[i,'DEV_TAZ'] = 1
            TAZ_HU = TAZ_HU + row['HU_NET']
            TAZ_EMP = TAZ_EMP + row['EMP_NET']

    #DevTable.to_csv(os.path.join(outputDir,"DevTable.csv"), index = False)
    #print('TAZ allocation complete')

#############################################################################
##   Run agency allocation
#############################################################################
print('\r\nRunning agency allocation...')
#DevTable = DevTable.merge(Forecast_dev, how = 'left', on = 'AGENCY')
DevTable = DevTable.sort_values(by=['SOI', 'COMMUNITY','TOTAL_SCORE'], ascending=[True, True, False]).reset_index(drop=True)
DevTable_EMP = DevTable.sort_values(by=['SOI', 'COMMUNITY','TOTAL_SCORE_EMP'], ascending=[True, True, False]).reset_index(drop=True)

DevYears = DevTable.groupby('DEV', as_index = False).agg({'HU_NET':'sum','EMP_NET':'sum'})
DevYears = DevYears[DevYears['DEV']>targetYear]
NextYear = DevYears['DEV'].min()

DevYears_EMP = DevTable_EMP.groupby('DEV', as_index = False).agg({'EMP_NET':'sum'})
DevYears_EMP = DevYears_EMP[DevYears_EMP['DEV']>targetYear]
NextYear_EMP = DevYears_EMP['DEV'].min()

print("Sampling growth from year", NextYear)

cycle = 0
while cycle < 1:
    DevSOI = DevTable[DevTable['DEV']<=targetYear].groupby('SOI', as_index = False).agg({'HU_NET':'sum','EMP_NET':'sum'})
    DevCOMMUNITY = DevTable[DevTable['DEV']<=targetYear].groupby('COMMUNITY', as_index = False).agg({'HU_NET':'sum','EMP_NET':'sum'})
    DevTAZ_SOI = DevTable.groupby('SOI', as_index = False).agg({'SOI_HU_Target':'first','SOI_EMP_Target':'first'}).merge(DevSOI, how = 'left', on = 'SOI')
    DevTAZ_SOI.fillna(0, inplace = True)
    DevTAZ_COMMUNITY = DevTable.groupby('COMMUNITY', as_index = False).agg({'SOI_HU_Target':'first','SOI_EMP_Target':'first','SOI_HU_P':'first','SOI_EMP_P':'first'}).merge(DevCOMMUNITY, how = 'left', on = 'COMMUNITY')
    DevTAZ_COMMUNITY.fillna(0, inplace = True)
#    print(DevTAZ_SOI)
#    print(DevTAZ_COMMUNITY)

    SOI = ''
    COMMUNITY = ''
    SOI_HU = 0
    SOI_EMP = 0
    SOI_HU_Target = 0
 #   SOI_EMP_Target = 0

    for i in range(len(DevTable)):
        row = DevTable.iloc[i]
        # Check for new SOI and/or community
        if SOI != row['SOI']:
            SOI = row['SOI']
            SOI_HU = 0
            SOI_HU_Target = row['SOI_HU_Target']-DevTAZ_SOI.loc[DevTAZ_SOI['SOI'] == SOI]['HU_NET'].sum()

        if SOI == 'Unincorporate' and row['COMMUNITY'] and COMMUNITY != row['COMMUNITY']:
            COMMUNITY = row['COMMUNITY']
            SOI_HU = 0
            SOI_HU_Target = row['SOI_HU_P']*(row['SOI_HU_Target']-DevTAZ_SOI.loc[DevTAZ_SOI['SOI'] == SOI]['HU_NET'].sum())
        # NEED TO ADD CHECK FOR NEXT HIGHEST YEAR
#        if (SOI != 'Unincorporate' or COMMUNITY) and row['DEV']==NextYear and ((SOI_HU+row['HU_NET']) <= max(SOI_HU_Target,0)) and ((SOI_EMP+row['EMP_NET'])<=max(SOI_EMP_Target,0)):
        if (SOI != 'Unincorporate' or COMMUNITY) and row['DEV']==NextYear and ((SOI_HU+row['HU_NET']) <= max(SOI_HU_Target,0)) :
            DevTable.at[i,'DEV'] = targetYear
            DevTable.at[i,'DEV_SOI'] = 1
            SOI_HU += row['HU_NET']
### EMP allocation start ###
    for i in range(len(DevTable_EMP)):
        row = DevTable_EMP.iloc[i]
        # Check for new SOI and/or community
        if SOI != row['SOI']:
            SOI = row['SOI']
            SOI_EMP = 0
            SOI_EMP_Target = row['SOI_EMP_Target']-DevTAZ_SOI.loc[DevTAZ_SOI['SOI'] == SOI]['EMP_NET'].sum()

        if SOI == 'Unincorporate' and row['COMMUNITY'] and COMMUNITY != row['COMMUNITY']:
            COMMUNITY = row['COMMUNITY']
            SOI_EMP = 0
            SOI_EMP_Target = row['SOI_EMP_P']*(row['SOI_EMP_Target']-DevTAZ_SOI.loc[DevTAZ_SOI['SOI'] == SOI]['EMP_NET'].sum())
        # NEED TO ADD CHECK FOR NEXT HIGHEST YEAR
#        if (SOI != 'Unincorporate' or COMMUNITY) and row['DEV']==NextYear_EMP and ((SOI_HU+row['HU_NET']) <= max(SOI_HU_Target,0)) and ((SOI_EMP+row['EMP_NET'])<=max(SOI_EMP_Target,0)):
        if (SOI != 'Unincorporate' or COMMUNITY) and row['DEV']==NextYear_EMP and ((SOI_EMP+row['EMP_NET'])<=max(SOI_EMP_Target,0)) :
            DevTable_EMP.at[i,'DEV'] = targetYear
            DevTable_EMP.at[i,'DEV_SOI'] = 1
            SOI_EMP += row['EMP_NET']
### EMP allocation end ###


    cycle += 1

#print('Agency allocation complete')
DevTable.to_csv(os.path.join(outputDir,"devtable.csv"), index = False)
DevTable_EMP.to_csv(os.path.join(outputDir,"devtable_emp.csv"), index = False)

DevTable = DevTable[DevTable['DEV']<=targetYear]
Dev_COMMUNITY = DevTable[DevTable['SOI']=='Unincorporate'].groupby('COMMUNITY', as_index = False).agg({'HU_NET':'sum','SOI_HU_Target':'first','EMP_NET':'sum','SOI_EMP_Target':'first'})
Dev_SOI = DevTable.groupby('SOI', as_index = False).agg({'HU_NET':'sum','SOI_HU_Target':'first','EMP_NET':'sum','SOI_EMP_Target':'first'})
print(Dev_SOI)
print(Dev_COMMUNITY)

DevTable_EMP = DevTable_EMP[DevTable_EMP['DEV']<=targetYear]
Dev_COMMUNITY_EMP = DevTable_EMP[DevTable_EMP['SOI']=='Unincorporate'].groupby('COMMUNITY', as_index = False).agg({'EMP_NET':'sum','SOI_EMP_Target':'first'})
Dev_SOI_EMP = DevTable_EMP.groupby('SOI', as_index = False).agg({'EMP_NET':'sum','SOI_EMP_Target':'first'})
print(Dev_SOI_EMP)
print(Dev_COMMUNITY_EMP)

#############################################################################
##   Calculate parcel growth values
#############################################################################
# Calculate new growth for parcels
Parcels_Dev = DevTable.filter(items=['parcelid'])
Parcels_Dev = Parcels_Dev.merge(pd.read_csv(os.path.join(outputDir, "parcels.csv")), how = 'left', on = 'parcelid')
Parcels_Dev['HH_NET']=Parcels_Dev['HU_NET']*(1-Parcels_Dev['VacRate'])
Parcels_Dev['POP_NET']=Parcels_Dev['HH_NET']*Parcels_Dev['HH_SIZE']
Parcels_Dev['HU_SF_NET']=Parcels_Dev['ACRES']*Parcels_Dev['HU_Den']*(Parcels_Dev['HU_SF_P'])-Parcels_Dev['HU_SF']
Parcels_Dev['HU_MF_NET']=Parcels_Dev['ACRES']*Parcels_Dev['HU_Den']*Parcels_Dev['HU_MF_P']-Parcels_Dev['HU_MF']
Parcels_Dev['HU_OTH_NET']=Parcels_Dev['ACRES']*Parcels_Dev['HU_Den']*Parcels_Dev['HU_OTH_P']-Parcels_Dev['HU_OTH']

Parcels_Dev_HU = Parcels_Dev
Parcels_Dev_HU = Parcels_Dev_HU[['parcelid', 'HH_NET', 'POP_NET','HU_SF_NET','HU_MF_NET','HU_OTH_NET']]  # Keep new columns only
Parcels_Dev = DevTable_EMP.filter(items=['parcelid'])
Parcels_Dev = Parcels_Dev.merge(pd.read_csv(os.path.join(outputDir, "parcels.csv")), how = 'left', on = 'parcelid')
Parcels_Dev['EDU_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['EDU_P']-Parcels_Dev['EMP_EDU']
Parcels_Dev['FOO_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['FOO_P']-Parcels_Dev['EMP_FOO']
Parcels_Dev['GOV_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['GOV_P']-Parcels_Dev['EMP_GOV']
Parcels_Dev['IND_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['IND_P']-Parcels_Dev['EMP_IND']
Parcels_Dev['MED_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['MED_P']-Parcels_Dev['EMP_MED']
Parcels_Dev['OFC_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['OFC_P']-Parcels_Dev['EMP_OFC']
Parcels_Dev['OTH_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['OTH_P']-Parcels_Dev['EMP_OTH']
Parcels_Dev['RET_NET']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['RET_P']-Parcels_Dev['EMP_RET']
Parcels_Dev['AGR_NET']=-Parcels_Dev['EMP_AGR']

if False:
    row_region = Forecast_dev.loc[Forecast_dev['SOI'] == 'Fresno County']

    # General industrial
    Parcels_Dev['G_IND']=Parcels_Dev['ACRES']*Parcels_Dev['EMP_Den']*Parcels_Dev['G_IND_P']
    g_ind_total = Parcels_Dev['G_IND'].sum()
    if g_ind_total:
        ind_target = max(0,row_region.iloc[0]['SOI_IND_Target']-Parcels_Dev['IND_NET'].sum())
        oth_target = max(0,row_region.iloc[0]['SOI_OTH_Target']-Parcels_Dev['OTH_NET'].sum())
        if ind_target+oth_target>0:
            ind_p = min(g_ind_total,ind_target+oth_target)*ind_target + B*row_region.iloc[0]['SOI_IND_Target']/(row_region.iloc[0]['SOI_IND_Target']+row_region.iloc[0]['SOI_OTH_Target'])/(ind_target+oth_target)
            oth_p = oth_target/(ind_target+oth_target)
        elif row_region.iloc[0]['SOI_IND_Target']+row_region.iloc[0]['SOI_OTH_Target']>0:
            print("NO")

    # General office
    edu_target = row_region.iloc[0]['SOI_EDU_Target']-Parcels_Dev['EDU_NET'].sum()
    gov_target = row_region.iloc[0]['SOI_GOV_Target']-Parcels_Dev['GOV_NET'].sum()
    med_target = row_region.iloc[0]['SOI_MED_Target']-Parcels_Dev['MED_NET'].sum()
    ofc_target = row_region.iloc[0]['SOI_OFC_Target']-Parcels_Dev['OFC_NET'].sum()

    # General retail
    foo_target = row_region.iloc[0]['SOI_FOO_Target']-Parcels_Dev['FOO_NET'].sum()
    ret_target = row_region.iloc[0]['SOI_RET_Target']-Parcels_Dev['RET_NET'].sum()

Parcels_Dev['EMP_NET']=Parcels_Dev['EDU_NET']+Parcels_Dev['FOO_NET']+Parcels_Dev['GOV_NET']+Parcels_Dev['IND_NET']+Parcels_Dev['MED_NET']+Parcels_Dev['OFC_NET']+Parcels_Dev['OTH_NET']+Parcels_Dev['RET_NET']+Parcels_Dev['AGR_NET']

Parcels_Dev = Parcels_Dev[['parcelid', 'EDU_NET', 'FOO_NET','GOV_NET','IND_NET','MED_NET','OFC_NET','OTH_NET','RET_NET','AGR_NET','EMP_NET']]  # Keep new columns only

Parcels_Dev = Parcels_Dev.merge(Parcels_Dev_HU, how = 'outer', on = 'parcelid') # merge EMP and HU growth allocation
Parcels_Dev = Parcels_Dev.merge(pd.read_csv(os.path.join(outputDir, "parcels.csv")), how = 'left', on = 'parcelid')  # merge parcel csv to get other column back

Parcels_Dev.to_csv(os.path.join(outputDir,"parcels_dev.csv"), index = False)


#############################################################################
##   Generate new base MAZ and TAZ files
#############################################################################
# Aggregate new growth to MAZ level
MAZ_Growth = Parcels_Dev.groupby('MAZ', as_index = False).agg({'HH_NET':'sum','POP_NET':'sum','HU_NET':'sum','EMP_NET':'sum','HU_SF_NET':'sum','HU_OTH_NET':'sum','HU_MF_NET':'sum',
    'EDU_NET':'sum','FOO_NET':'sum','GOV_NET':'sum','IND_NET':'sum','MED_NET':'sum','OFC_NET':'sum','OTH_NET':'sum','RET_NET':'sum','AGR_NET':'sum','SCHL_Factor':'first'})
#MAZ_Growth.to_csv(os.path.join(outputDir,"maz_growth.csv"), index = False)

# Determine scaling factors for rural unincorporated area
rural_hu_factor = (Base_MAZ[Base_MAZ['SOI']=='Unincorporate']['Base_HU'].sum()+Forecast[Forecast['SOI']=='Unincorporate']['SOI_HU_Target'].sum()-Dev_COMMUNITY['HU_NET'].sum())/Base_MAZ[Base_MAZ['SOI']=='Unincorporate']['Base_HU'].sum()
rural_emp_factor = (Base_MAZ[Base_MAZ['SOI']=='Unincorporate']['Base_EMP'].sum()+Forecast[Forecast['SOI']=='Unincorporate']['SOI_EMP_Target'].sum()-Dev_COMMUNITY['EMP_NET'].sum())/Base_MAZ[Base_MAZ['SOI']=='Unincorporate']['Base_EMP'].sum()
print("rural_hu_factor ",rural_hu_factor)
print("rural_emp_factor",rural_emp_factor)

# Generate new MAZ file
MAZ_New = Base_MAZ.merge(MAZ_Growth, how = 'left', on = 'MAZ')
MAZ_New.fillna(0, inplace = True)
MAZ_New.loc[MAZ_New['SCHL_Factor'] == 0,'SCHL_Factor'] = 1
MAZ_New['HU_Factor'] = 1
MAZ_New.loc[MAZ_New['PLANNED']==0,'HU_Factor'] = rural_hu_factor
MAZ_New['EMP_Factor'] = 1
MAZ_New.loc[MAZ_New['PLANNED']==0,'EMP_Factor'] = rural_emp_factor
MAZ_New['POP'] = (MAZ_New['HH_POP']*MAZ_New['HU_Factor']+MAZ_New['POP_NET']).clip(lower=0)
MAZ_New['HH'] = (MAZ_New['Base_HH']*MAZ_New['HU_Factor']+MAZ_New['HH_NET']).clip(lower=0)
MAZ_New['HU'] = (MAZ_New['Base_HU']*MAZ_New['HU_Factor']+MAZ_New['HU_NET']).clip(lower=0)
MAZ_New['HU_SF'] = (MAZ_New['HU_SF']*MAZ_New['HU_Factor']+MAZ_New['HU_SF_NET']).clip(lower=0)
MAZ_New['HU_MF'] = (MAZ_New['HU_MF']*MAZ_New['HU_Factor']+MAZ_New['HU_MF_NET']).clip(lower=0)
MAZ_New['HU_OTH'] = (MAZ_New['HU_OTH']*MAZ_New['HU_Factor']+MAZ_New['HU_OTH_NET']).clip(lower=0)
MAZ_New['HU'] = MAZ_New['HU_SF']+MAZ_New['HU_MF']+MAZ_New['HU_OTH']
#MAZ_New['EMP'] = (MAZ_New['Base_EMP']+MAZ_New['EMP_NET']
MAZ_New['EMP_EDU'] = (MAZ_New['Base_EDU']*MAZ_New['EMP_Factor']+MAZ_New['EDU_NET']).clip(lower=0)
MAZ_New['EMP_FOO'] = (MAZ_New['Base_FOO']*MAZ_New['EMP_Factor']+MAZ_New['FOO_NET']).clip(lower=0)
MAZ_New['EMP_GOV'] = (MAZ_New['Base_GOV']*MAZ_New['EMP_Factor']+MAZ_New['GOV_NET']).clip(lower=0)
MAZ_New['EMP_IND'] = (MAZ_New['Base_IND']*MAZ_New['EMP_Factor']+MAZ_New['IND_NET']).clip(lower=0)
MAZ_New['EMP_MED'] = (MAZ_New['Base_MED']*MAZ_New['EMP_Factor']+MAZ_New['MED_NET']).clip(lower=0)
MAZ_New['EMP_OFC'] = (MAZ_New['Base_OFC']*MAZ_New['EMP_Factor']+MAZ_New['OFC_NET']).clip(lower=0)
MAZ_New['EMP_OTH'] = (MAZ_New['Base_OTH']*MAZ_New['EMP_Factor']+MAZ_New['OTH_NET']).clip(lower=0)
MAZ_New['EMP_RET'] = (MAZ_New['Base_RET']*MAZ_New['EMP_Factor']+MAZ_New['RET_NET']).clip(lower=0)
MAZ_New['EMP_AGR'] = (MAZ_New['Base_AGR']*agr_factor+MAZ_New['AGR_NET']).clip(lower=0)
MAZ_New['EMP'] = MAZ_New['EMP_EDU']+MAZ_New['EMP_FOO']+MAZ_New['EMP_GOV']+MAZ_New['EMP_IND']+MAZ_New['EMP_MED']+MAZ_New['EMP_OFC']+MAZ_New['EMP_OTH']+MAZ_New['EMP_RET']+MAZ_New['EMP_AGR']
MAZ_New['DORM'] = MAZ_New['Base_DORM']*gq_factor
MAZ_New['MEDICAL'] = MAZ_New['Base_MEDICAL']*gq_factor
MAZ_New['PRISON'] = MAZ_New['Base_PRISON']*gq_factor
MAZ_New['MILITARY'] = MAZ_New['Base_MILITARY']*gq_factor
MAZ_New['ELEM'] = MAZ_New['Base_ELEM']*MAZ_New['SCHL_Factor']
MAZ_New['HS'] = MAZ_New['Base_HS']*MAZ_New['SCHL_Factor']
MAZ_New['COLLEGE'] = MAZ_New['Base_COLLEGE']*MAZ_New['SCHL_Factor']

# Scale rural growth


MAZ_New.round({'HH': 0,'POP': 0,'DORM': 0,'MEDICAL': 0,'PRISON': 0,'MILITARY': 0,'ELEM': 0,'HS': 0,'COLLEGE': 0})
MAZ_New['HH'] = MAZ_New['HH'].astype(int)
MAZ_New['POP'] = MAZ_New['POP'].astype(int)
MAZ_New['DORM'] = MAZ_New['DORM'].astype(int)
MAZ_New['MEDICAL'] = MAZ_New['MEDICAL'].astype(int)
MAZ_New['PRISON'] = MAZ_New['PRISON'].astype(int)
MAZ_New['MILITARY'] = MAZ_New['MILITARY'].astype(int)
MAZ_New['ELEM'] = MAZ_New['ELEM'].astype(int)
MAZ_New['HS'] = MAZ_New['HS'].astype(int)
MAZ_New['COLLEGE'] = MAZ_New['COLLEGE'].astype(int)

MAZ_New = MAZ_New.filter(items=['MAZ','TAZ','SOI','POP','HH','HU','HU_SF','HU_MF','HU_OTH',
    'EMP','EMP_EDU','EMP_FOO','EMP_GOV','EMP_IND','EMP_MED','EMP_OFC','EMP_OTH','EMP_RET','EMP_AGR',
    'DORM','MEDICAL','PRISON','MILITARY','ELEM','HS','COLLEGE','SCHL_Factor'])
MAZ_New = MAZ_New.sort_values(by=['MAZ'], ascending=[True]).reset_index(drop=True)
MAZ_New.to_csv(os.path.join(outputDir,"maz_new.csv"), index = False)

# Generate new TAZ file
TAZ_New = MAZ_New.groupby('TAZ', as_index = False).agg({'SOI':'first','POP':'sum','HH':'sum','HU':'sum','HU_SF':'sum','HU_MF':'sum','HU_OTH':'sum',
    'EMP':'sum','EMP_EDU':'sum','EMP_FOO':'sum','EMP_GOV':'sum','EMP_IND':'sum','EMP_MED':'sum','EMP_OFC':'sum','EMP_OTH':'sum','EMP_RET':'sum','EMP_AGR':'sum',
    'DORM':'sum','MEDICAL':'sum','PRISON':'sum','MILITARY':'sum','ELEM':'sum','HS':'sum','COLLEGE':'sum'})
#TAZ_New = TAZ_New.sort_values(by=['TAZ'], ascending=[True]).reset_index(drop=True)
TAZ_New.to_csv(os.path.join(outputDir,"taz_new.csv"), index = False)


#############################################################################
##   Generate PopSim input files
#############################################################################
# Generate mazData
mazData = MAZ_New.filter(items=['MAZ','HH'])
mazData.rename(columns={'HH':'2014 HH'}, inplace=True)
mazData.to_csv(os.path.join(popsimDir,"mazData.csv"), index = False)

# Generate tazData
tazData = TAZ_New.filter(items=['TAZ','HU_SF','HU_MF','HU_OTH'])
tazData.to_csv(os.path.join(popsimDir,"tazData.csv"), index = False)

# gq_maz
base_gq_noninst = pd.read_csv(os.path.join(dataPopSim_Dir, "gq_maz.csv"))['othnon14'].sum()
gq_maz = MAZ_New.filter(items=['MAZ','DORM','MILITARY','MEDICAL'])[['MAZ','DORM','MILITARY','MEDICAL']]
gq_maz.columns=['MAZ','univ14','mil14','othnon14']
gq_maz['othnon14'] = gq_maz['othnon14']*base_gq_noninst/MAZ_New['MEDICAL'].sum()
gq_maz.round({'othnon14': 0})
gq_maz['othnon14'] = gq_maz['othnon14'].astype(int)
gq_maz.to_csv(os.path.join(popsimDir,"gq_maz.csv"), index = False)

# countyData
countyData = pd.DataFrame({'county':['FRESNO'], 'hhpop': [MAZ_New['POP'].sum()],'minp':[minority_p]})[['county','hhpop','minp']]
countyData.to_csv(os.path.join(popsimDir,"countyData.csv"), index = False)

#############################################################################
##   End of script
#############################################################################

print('\r\n--- Script ran successfully! ---\r\n')
print('End time '+str(datetime.now()))

