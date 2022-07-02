# JWST-NIRCam-Photometry-Tests

Included Here are a series of JWST NIRCam Simulation Tests for exoplanet photometry with the JWST Science Calibration Pipeline and the Time Series Helper & Integration Reduction Tool (tshirt). 

Documentation Pages: 
- JWST Pipeline: https://jwst-pipeline.readthedocs.io/en/latest/jwst/introduction.html 
- tshirt Pipeline: https://tshirt.readthedocs.io/en/latest/

To Run these notebooks:
1. Create a new virtual enviorment \
  `conda create -n <env_name> python = 3.8` \
  `conda activate <env_name>`
2. Install all required packages from the requirements file \
  `pip install -r requirements.txt`
3. Download Simulation or Real Data to Test (e.g. from MAST)
4. Run the Notebooks!

## To Run the JWST & tshirt Pipeline Step by Step, refer to the notebooks in the folders `GJ3470b`, `HATP14b`, `HD189733b`, `Original_Data_Set_NRCA3_DataChallengeSimulation`, `WASP39b`, `WASP80b`. 
- ROEBA Notebooks: These notebooks use a modified STAGE 1 from the JWST pipeline. Rather than preforming a mean background subtraction on the data, these notebook preform a Row-by-row, Odd/even by amplifier correction to the background subtraction. Refer to: https://tshirt.readthedocs.io/en/latest/specific_modules/ROEBA.html?highlight=ROEBA

## For a ONE Step Photometry Function on the Data refer to the notebook and code in the folder `Photometry_Pipeline_Code`:
- Current Usage: CYCLE1 
Refer the the example notebook in this folder to see how to run this code and what the parameters in the functions stand for. 

Note: CYCLE1 will focus on obtaining grism time-series. However, with NIRCam simultaneous observations can be taken in the long and short wavelength (SW) channels and we can perform photometry in the SW detectors. In this observing mode, the observations in both the LW and SW channels must use the same SUBGRISM array and readout. So, the LW detector will determine where the target is located in the SW detector. Therefore, to successfully run this code, one must download all data related to the target (e.g. the long-wavelength channel and the short-wavelength channels). This is required because in real observations (at least for CYCLE1), simultaneous observations on various detector are occuring at the same time (grism time-series and time-series imaging) and the one will not know where the target is located to do photometry. During the grism time-series target aquisition, depending on the selected filter, the target can placed at either of 2 locations (either on NRCA1 or NRCA3). When F277W, F322W2, or F356W are used, TA places the target towards the right of the long wavelength detector (NRCA1). When F444W is used, TA places the target on NRCA3. 

For More Information: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-operations/nircam-target-acquisition/nircam-grism-time-series-target-acquisition
