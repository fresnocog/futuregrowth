::-------------------------------------------------------------------------------------------------
:: This batch file runs the land use allocation process for the Fresno region, using
::    growth targets
::
:: Seth Scott, sscott@fresnocog.org, 05/07/2020
::
:: TO DO
:: 1. make sure python call is from the same directory as in the setup folder
::-------------------------------------------------------------------------------------------------

@ECHO OFF
ECHO %startTime%%Time%

:: FLAGS - YES/NO
SET RUN_CONTROLS=YES
SET RUN_SKIMS=NO
SET RUN_SCORING=YES
SET RUN_ALLOCATION=YES
SET RUN_PERFORMANCE=YES
SET RUN_ABM_INPUTS=YES

:: Set Inputs and Directories
SET WORKING_DIR=%~dp0
SET SCENARIO=%1

IF [%SCENARIO%] == [] (SET PARAMETERS_FILE="%WORKING_DIR%Setup\parameters.csv") ELSE (SET PARAMETERS_FILE="%WORKING_DIR%Scenarios\%SCENARIO%\parameters.csv")

IF NOT EXIST %PARAMETERS_FILE% (
	ECHO ERROR - Could not find parameter file
	ECHO %PARAMETERS_FILE%
	EXIT /B
)

::-------------------------------------------------------------------------------------------------

:: Create growth controls
IF %RUN_CONTROLS%==YES (
	ECHO %startTime%%Time%: Creating growth controls...
	CALL python Scripts\buildControls.py %PARAMETERS_FILE%>> RunAll.log 2>&1
)

:: Process skims
IF %RUN_SKIMS%==YES (
	ECHO %startTime%%Time%: Processing skims...
	CALL python scripts\processSkims.py %PARAMETERS_FILE%>> RunAll.log 2>&1
)
	
:: Calculate development scores
IF %RUN_SCORING%==YES (
	ECHO %startTime%%Time%: Calculating development scores...
	CALL python scripts\calcDevScores.py %PARAMETERS_FILE%>> RunAll.log 2>&1
)

:: Allocate new growth
IF %RUN_ALLOCATION%==YES (
	ECHO %startTime%%Time%: Allocating new growth...
	CALL python scripts\allocateGrowth.py %PARAMETERS_FILE%>> RunAll.log 2>&1
)

:: Calculate performance indicators
IF %RUN_PERFORMANCE%==YES (
	ECHO %startTime%%Time%: Calculating performance indicators...
	CALL python scripts\performance.py %PARAMETERS_FILE%>> RunAll.log 2>&1
)

:: Generate ABM input files
IF %RUN_ABM_INPUTS%==YES (
	ECHO %startTime%%Time%: Generating ABM input files...
	CALL python scripts\createModelInputs.py %PARAMETERS_FILE%>> RunAll.log 2>&1
)

ECHO %startTime%%Time%: Batch complete