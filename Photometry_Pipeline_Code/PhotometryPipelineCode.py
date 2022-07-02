#import library
import os

#Import If Needed:
#os.environ['CRDS_PATH'] = '/fenrirdata1/kg_data/crds_cache/' #These pathways should be defined in your ~./bash profile. If not, you can set them within the notebook.
#os.environ['CRDS_SERVER_URL']= 'https://jwst-crds.stsci.edu'
#os.environ['CRDS_CONTEXT']='jwst_0785.pmap' #Occasionally, the JWST CRDS pmap will be updated. Updates may break existing code. Use this command to revert to an older working verison until the issue is fixed. 

#JWST Pipeline Imports
import jwst
print("The Version of the JWST Pipeline is "+ str(jwst.__version__)) #Print what version of the pipeline you are using.

from jwst.pipeline.calwebb_detector1 import Detector1Pipeline #Stage 1
from jwst.pipeline.calwebb_image2 import Image2Pipeline #Stage 2
from jwst.pipeline.calwebb_tso3 import Tso3Pipeline #Stage 3
from jwst.associations.asn_from_list import asn_from_list #Association file imports
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase

# Individual steps that make up calwebb_detector1
from jwst.dq_init import DQInitStep
from jwst.saturation import SaturationStep
from jwst.superbias import SuperBiasStep
from jwst.ipc import IPCStep                                                                                    
from jwst.refpix import RefPixStep                                                                
from jwst.linearity import LinearityStep
from jwst.persistence import PersistenceStep
from jwst.dark_current import DarkCurrentStep
from jwst.jump import JumpStep
from jwst.ramp_fitting import RampFitStep
from jwst import datamodels

#General Imports
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from configparser import ConfigParser
import csv
import numpy as np
import asdf
import glob
import pdb
import random
import time
import yaml
import tqdm
from copy import deepcopy
import pandas as pd
from astropy.table import Table

## ES tshirt custom pipeline
from tshirt.pipeline import phot_pipeline, analysis #tshirt specific imports
from tshirt.pipeline.instrument_specific import rowamp_sub
from splintegrate import splintegrate

#Modeling; will develop later
#import batman
#from scipy.optimize import curve_fit

#Style Choice
class style:
   BOLD = '\033[1m'
   END = '\033[0m'
    
    
def log_output(TargetName, output_dir):
    #Get the configparser object
    config_object = ConfigParser()
    
    #Required sections
    config_object["*"] = {"handler": "append:{}_pipeline.log".format(TargetName), "level": "INFO"}
    
    #Write the above sections to stpipe-log.cfg file
    pwd = os.getcwd() #current working directory
    
    #Files must be written to both the working directory and output_dir
    with open(pwd+'/stpipe-log.cfg', 'w') as conf:
        config_object.write(conf)
        
    print("Configuration file and output file for the JWST Pipeline Log has been created in the working directory: {}_stpipe-log.cfg & {}_pipeline.log".format(TargetName, TargetName))


def PhotometryPipeline(TargetName, rawFilesDirectory, output_dir, STAGE3='JWST', radii_values=[None, None, None], simulation = False, sweep =False, recalculate = False, showPlot = False, add_noutputs_keyword=False, useMultiprocessing = False):
    """
    The JWST Pipeline:
    ------------------
    
    Stage 1: Detector-level corrections and ramp fitting for individual exposures. 
    
        Basic detector-level connections are applied to all exposure types (imaging, spectroscopic, coronagraphic, etc.). 
        It is applied to one exposure at a time. It is sometimes referred to as “ramps-to-slopes” processing, because the 
        input raw data are in the form of one or more ramps (integrations) containing accumulating counts from the 
        non-destructive detector readouts and the output is a corrected countrate (slope) image, which contains the 
        original raw data from all of the detector readouts in the exposure (ncols x nrows x ngroups x nintegrations).
        The list of basic detector-level connections are:
        
        - The 𝐢𝐩𝐜 step corrects a JWST exposure for interpixel capacitance by convolving with an IPC reference image.
        - The 𝐩𝐞𝐫𝐬𝐢𝐬𝐭𝐞𝐧𝐜𝐞 step. Based on a model, this step computes the number of traps that are expected to have captured 
            or released a charge during an exposure. The released charge is proportional to the persistence signal, and 
            this will be subtracted (group by group) from the science data.
        - The 𝐝𝐪_𝐢𝐧𝐢𝐭 Data Quality (DQ) initialization step in the calibration pipeline populates the DQ mask for the input 
            dataset. The MASK reference file contains pixel-by-pixel DQ flag values that indicate problem conditions.
        - The 𝐬𝐚𝐭𝐮𝐫𝐚𝐭𝐢𝐨𝐧 step flags pixels at or below the A/D floor or above the saturation threshold.
        - The 𝐬𝐮𝐩𝐞𝐫𝐛𝐢𝐚𝐬 superbias subtraction step removes the fixed detector bias from a science data set by subtracting 
            a superbias reference image.
        - The 𝐫𝐞𝐟𝐩𝐢𝐱 step corrects for these readout signal drifts by using the reference pixels.
        - The 𝐥𝐢𝐧𝐞𝐚𝐫𝐢𝐭𝐲 step applies the “classic” linearity correction adapted from the HST WFC3/IR linearity correction 
            routine, correcting science data values for detector non-linearity. The correction is applied pixel-by-pixel, 
            group-by-group, integration-by-integration within a science exposure.
        - The 𝐝𝐚𝐫𝐤_𝐜𝐮𝐫𝐫𝐞𝐧𝐭 step removes dark current from an exposure by subtracting dark current data stored in a 
            dark reference file in CRDS.
        - The 𝐣𝐮𝐦𝐩 routine detects jumps in an exposure by looking for outliers in the up-the-ramp signal for each pixel 
            in each integration within an input exposure.
        - The 𝐫𝐚𝐦𝐩_𝐟𝐢𝐭𝐭𝐢𝐧𝐠 step determines the mean count rate, in units of counts per second, for each pixel by performing
            a linear fit to the data in the input file. The fit is done using the “ordinary least squares” method.
        
        Output: 
        The output of stage 1 processing is a countrate image (DN/s) per exposure, or per integration for some modes. 
        All types of inputs result in a 2D countrate product, based on averaging over all of the integrations within 
        the exposure. The output file will be of type “_rate” (0_rampfitstep). A 3D countrate product is created that 
        contains the individual results of each integration. The 2D countrate images for each integration are stacked 
        along the 3rd axis of the data cubes (ncols x nrows x nints). This output file will be of type “_rateints” 
        (1_rampfitstep). 
    
    Stage 2: Instrument-mode calibrations for individual exposures
    
        There are two main Stage 2 pipelines: one for imaging and one for spectroscopy. Stage 2 processing consists of 
        additional instrument-level and observing-mode corrections and calibrations to produce fully calibrated exposures. 
        The details differ for imaging and spectroscopic exposures, and there are some corrections that are unique to 
        certain instruments or modes.
    
        𝐈𝐦𝐚𝐠𝐞2𝐏𝐢𝐩𝐞𝐥𝐢𝐧𝐞: Imaging processing applies additional instrumental corrections and calibrations that result in a 
        fully calibrated individual exposure. Imaging TSO data are run through this pipeline. The steps are very similar 
        to those in Spec2Pipeline. WCS information is added, flat fielding and flux calibration are performed, and 
        astrometric distortion is removed from the images. There are two parameter references used to control this pipeline, 
        depending on whether the data are to be treated as Time Series Observation (TSO). The parameter reference is 
        provided by CRDS. For TSO exposures, some steps are set to be skipped by default. The input to Image2Pipeline 
        is a countrate exposure, in the form of either “_rate” or “_rateints” data. A single input file can be processed 
        or an ASN file listing multiple inputs can be used, in which case the processing steps will be applied to each 
        input exposure, one at a time. If “_rateints” products are used as input, each step applies its algorithm to each 
        integration in the exposure, where appropriate.
        
        Output: 
        The output is a fully calibrated, but unrectified, exposure, using the product type suffix “_cal” (0_cal) or 
        “_calints” (1_calints), depending on the type of input.
    
    Stage 3: Combining data from multiple exposures within an observation
    
        Stage 3 processing consists of routines that work with multiple exposures and in most cases produce some kind of
        combined product. There are unique pipeline modules for stage 3 processing of imaging, spectroscopic, coronagraphic, 
        AMI, and TSO observations.
        
        𝐓𝐬𝐨3𝐏𝐢𝐩𝐞𝐥𝐢𝐧𝐞: The Stage 3 TSO pipeline is to be applied to associations of calibrated TSO exposures 
        (e.g. NIRCam TS imaging, NIRCamTS grism, NIRISS SOSS, NIRSpec BrightObj, MIRI LRS Slitless) and is used to produce
        calibrated time-series photometry or spectra of the source object. This is a pipeline customized for TSO data. 
        Grism TSO data undergo outlier detection (essentially a check for any cosmic rays/transient effects that were missed 
        in Detector1Pipeline), background subtraction, spectral extraction, and photometry. Imaging TSO data are run through 
        outlier detection, and photometry is performed. The logic that decides whether to apply the imaging or spectroscopy 
        steps is based on the EXP_TYPE and TSOVISIT keyword values of the input data. Imaging steps are applied if either of
        the following is true: EXP_TYPE = ‘NRC_TSIMAGE’, EXP_TYPE = ‘MIR_IMAGE’ and TSOVISIT = True. The input to 
        𝐓𝐬𝐨3𝐏𝐢𝐩𝐞𝐥𝐢𝐧𝐞 is in the form of an ASN file that lists multiple exposures or exposure segments of a science target. 
        The individual inputs should be in the form of 3D calibrated (“_calints”) products from either Image2Pipeline or 
        Spec2Pipeline processing.

    tshirt Pipeline:
    ----------------
        From: https://tshirt.readthedocs.io/en/latest/phot_pipeline/phot_pipeline.html
        The Time Series Helper & Integration Reduction Tool (tshirt) is a general-purpose tool for time series science. 
        Its main application is transiting exoplanet science. tshirt can: Reduce raw data: flat field, bias subtract, 
        gain correct, etc. Extract Spectroscopy and in our interest extract photometry. This photometric pipeline will 
        take image data that have been reduced (using the JWST pipeline) and calculate lightcurves on the stars/sources 
        in the field. Therefore, we can use this external method and bypass stage 3 of the JWST Science Calibration Pipeline.
        
    Parameters:
    ----------
    TargetName: str
        A unique name for the target
    rawFilesDirectory: dir
        A directory containing raw uncalibrated files only
    output_dir: dir 
        An output directory for “_rate” (0_rampfitstep) and “_rateints” (1_rampfitstep) files
    STAGE3: str
        Support either 'JWST' or 'tshirt'. Directs code what pipeline to use for stage 3. 
    radii_values: array (*optional)
        Array of radii sizes for circular apetures in the form [radius_src, radius_inner, radius_outer]
    recalculate: bool
        Recalculate photometry (all stages)? 
    simulation: bool
        Is this simulation data?
    sweep: bool
        Preform an aperture sweep?
    showPlot: bool
        Show plots? 
    add_noutputs_keyword: bool
        Modify the uncal file NOUTPUTS. This is DANGEROUS. Only use for older mirage simulations that lacked NOUTPUTS keyword.
    """
    print(style.BOLD + "STAGE 1 IN PROGRESS" + style.END)
    pwd = os.getcwd() #current working directory

    if os.path.exists(output_dir+"{}_stpipe-log.cfg".format(TargetName)) == True:
        print("Outputing JWST Pipeline Log to file in working directory: {}_pipeline.log".format(TargetName))
        
    if os.path.exists(output_dir) == False:       #If the output directory does not exist, create it
        os.makedirs(output_dir)
    
    #RECENTERING
    if (simulation==True): #Working with simulation data
        RandFile_long = random.choice(glob.glob(rawFilesDirectory+"/*nrca5_uncal.fits")) #random long-wavelength file
        HDUList_long = fits.open(RandFile_long)
        Target_filter = HDUList_long[0].header['FILTER'] #Filter element
        HDUList_long.close()
        
        #Where detector is the target in? 
        if (Target_filter == 'F277W' or  Target_filter =='F322W2' or Target_filter =='F356W'):
            print("The target is located on detector A3")
            short_Detector = 'nrca3'
        elif (Target_filter == 'F444W'):
            print("The target is located on detector A1")
            short_Detector = 'nrca1'
        else:
            print("Taget Detector Location Undetermined")
            
    else: #Working with real data
        RandFile_long = random.choice(glob.glob(rawFilesDirectory+"/*nrcalong_uncal.fits")) #random long-wavelength fi
        HDUList_long = fits.open(RandFile_long)
        Target_filter = HDUList_long[0].header['FILTER'] #Filter element
        HDUList_long.close()
        
        if (Target_filter == 'F277W' or Target_filter == 'F322W2' or Target_filter == 'F356W'):
            print("The target is located on detector A3")
            short_Detector = 'nrca3'
        elif (Target_filter == 'F444W'):
            print("The target is located on detector A1")
            short_Detector = 'nrca1'
        else:
            print("Taget Detector Location Undetermined")
            
    RandFile_short = random.choice(glob.glob(rawFilesDirectory+"/*{}_uncal.fits".format(short_Detector))) #random short-wavelength file

    HDUList_short = fits.open(RandFile_short)  
    SUBARRAY = HDUList_short[0].header['SUBARRAY'] #What is the subarray
    nints_splintegrate = HDUList_short[0].header['NINTS']
    pupil = HDUList_short[0].header['PUPIL']
    filter_element = HDUList_short[0].header['FILTER']
    date_obs = HDUList_short[0].header['DATE-OBS']
    
    #NIRCam Subarray Origin point
    X_origin_subarray = HDUList_short[0].header['SUBSTRT1']
    Y_origin_subarray = 0
    
    #DS9 origin [0.5,0.5]:
    X_origin_DS9 = 0.5 
    Y_origin_DS9 = 0.5        
            
    if (simulation == True): #Working with simulation data 
        print("Working with Simulation data")
        
        if (SUBARRAY == 'SUBGRISM64'):
            print("Working with SUBGRISM64")
            #Star's DS9 position: Where the gets positioned automatically in the simulation 
            if (short_Detector=="nrca1"):
                X_star_DS9 = 1982.2 
                Y_star_DS9 = 30.2
            elif (short_Detector=="nrca3"):
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            else:
                None        
        elif (SUBARRAY=='SUBGRISM128'):
            print("Working with SUBGRISM128")
            #Star's DS9 position: Where the gets positioned automatically in the simulation
            if (short_Detector=="nrca3"):
                X_star_DS9 = 1052.8 
                Y_star_DS9 = 57.297356
            elif (short_Detector=="nrca1"): 
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            else:
                None     
        
        elif (SUBARRAY=='SUBGRISM256'):
            print("Working with SUBGRISM256")

            #Star's DS9 position: Where the gets positioned automatically in the simulation 
            if (short_Detector=="nrca3"):
                X_star_DS9 = 1056.0 
                Y_star_DS9 = 167.0
            elif (short_Detector=="nrca1"): 
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            else:
                None
        else:
            print("Unsupported Subarray")
    
    else: #Working with real data
        print("Working with Real data")
        if (SUBARRAY == 'SUBGRISM64'):
            print("Working with SUBGRISM64'")
            #Star's DS9 position: Where the gets positioned automatically
            if (short_Detector=="nrca1"):
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            elif (short_Detector=="nrca3"): 
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            else:
                None
                  
        elif (SUBARRAY=='SUBGRISM128'):
            print("Working with SUBGRISM128")
            #Star's DS9 position: Where the gets positioned automatically
            if (short_Detector== "nrca3"):
                #Star's DS9 position: 
                X_star_DS9 = 1060 
                Y_star_DS9 = 56
            elif (short_Detector=="nrca1"): 
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            else:
                None
                  
        elif (SUBARRAY=='SUBGRISM256'):
            print("Working with SUBGRISM256")

            #Star's DS9 position: Where the gets positioned automatically
            if (short_Detector=="nrca1"):
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            elif (short_Detector=="nrca3"): 
                X_star_DS9 = HDUList_short[1].header['XREF_SCI'] #DEFAULT
                Y_star_DS9 = HDUList_short[1].header['YREF_SCI']
            else:
                None
        else:
            print("Unsupported Subarray")
            
    #Match up the origins at [1,0] to get star's position on the detector subarray:
    X_star_subarray = X_star_DS9 + abs(X_origin_subarray - X_origin_DS9)
    Y_star_subarray = Y_star_DS9 - abs(Y_origin_subarray - Y_origin_DS9)
    #Modified reference center for the apertures:
    XREF_SCI = X_star_subarray
    YREF_SCI = Y_star_subarray
    print("The Aperture ReCentered Position: "+ str(XREF_SCI)+"," +str(YREF_SCI))        
        
    #Add Noutputs Keyword? 
    if (add_noutputs_keyword == True):
        all_uncal_files = []                          #List of All Uncalibrated File Names.
        for fitsName in glob.glob(rawFilesDirectory+"*{}_uncal.fits".format(short_Detector)): #Update the Primary header information for 4 detector amplifiers 
            HDUList = fits.open(fitsName, 'update')
            HDUList[0].header['NOUTPUTS'] = (4, 'Number of output amplifiers')
            HDUList.close()
            all_uncal_files.append(fitsName)
   
        all_uncal_files = sorted(all_uncal_files)     #Sort uncalibrated files alphabetically.
    else: 
        all_uncal_files = sorted(glob.glob(rawFilesDirectory+"*{}_uncal.fits".format(short_Detector)))
    
    #STAGE 1 PHTOOMETRY
    max_cores = "none" #Set to none, as to not go over memory limit.
    photParam = {'refStarPos': [[XREF_SCI,YREF_SCI]],'backStart':100,'backEnd': 101, 'FITSextension': 1, 
                 'isCube': True,'cubePlane':0,'procFiles':'*.fits'} #For Stage 1 Corrections. 
    
    stage1_result_files = [] #Empty list to append rateints files
    startTime_stage1 = time.time() #Time how long stage 1 takes
    
    for rawFile in all_uncal_files: #Pre-setting what the rateints files will be named
        filename=os.path.basename(rawFile) #Grab name not directory
        stage1_result_file = filename.replace("uncal", "1_rampfitstep")
        stage1_result_files.append(output_dir+stage1_result_file)
        
        #Determine if the stage 1 was already run and file exists 
        if (os.path.exists(output_dir+stage1_result_file) == True) and (recalculate == False): 
            print(stage1_result_file+ style.BOLD + " EXISTS" + style.END)
        else: 
            print(stage1_result_file+" does't exist or recalculation is TRUE ... processing slopes-to-ramps")
            
            #Using the run() method. Instantiate and set parameters
            #Data Quality Initialization
            dq_init_step = DQInitStep()
            #dq_init_step.logcfg = pwd+"/{}_pipeline.log".format(TargetName)
            dq_init = dq_init_step.run(rawFile)
            
            # Saturation Flagging
            saturation_step = SaturationStep()
            #Call using the the output from the previously-run dq_init step
            saturation = saturation_step.run(dq_init)
            del dq_init # try to save memory
            
            # Superbias Subtraction
            superbias_step = SuperBiasStep()
            #superbias_step.override_superbias = "/usr/local/nircamsuite/cal/Bias/jwst_nircam_superbias_nrca3_subgrism128_bias0.fits" #Override only for the HATP14b real data post lauch
            #Superbias_step.output_dir = output_dir
            #Superbias_step.save_results = True
            #Call using the the output from the previously-run saturation step
            superbias = superbias_step.run(saturation)
            del saturation ## try to save memory
            
            #Reference Pixel Correction
            refpix_step = RefPixStep()
            #refpix_step.output_dir = output_dir
            #refpix_step.save_results = True
            #try using a copy of the bias results as the refpix output
            #refpix = refpix_step.run(superbias)
            #es_refpix = deepcopy(refpix)
            #the old way was to run the refpix and then replace it
            es_refpix = deepcopy(superbias)
                
            ngroups = superbias.meta.exposure.ngroups
            nints = superbias.data.shape[0] ## use the array size because segmented data could have fewer ints
            
            #First, make sure that the aperture looks good.
            phot = phot_pipeline.phot(directParam=photParam)
            #phot.showStamps(showPlot=True,boxsize=200,vmin=0,vmax=1)
            
            #Everything inside the larger blue circle will be masked when doing reference pixel corrections
            for oneInt in tqdm.tqdm(np.arange(nints)):
                for oneGroup in np.arange(ngroups):
                    
                    rowSub, modelImg = rowamp_sub.do_backsub(superbias.data[oneInt,oneGroup,:,:],phot)
                    es_refpix.data[oneInt,oneGroup,:,:] = rowSub
                    
            #Linearity Step
            del superbias # try to save memory
            linearity_step = LinearityStep()
            #Call using the the output from the previously-run refpix step
            linearity = linearity_step.run(es_refpix)
            del es_refpix # try to save memory
            
            #Persistence Step
            persist_step = PersistenceStep()
            #skip for now since ref files are zeros
            persist_step.skip = True
            #Call using the the output from the previously-run linearity step
            persist = persist_step.run(linearity)
            del linearity # try to save memory
            
            #Dark current step
            dark_step = DarkCurrentStep()
            #There was a CRDS error so I'm skipping
            dark_step.skip = True
            #Call using the persistence instance from the previously-run persistence step
            dark = dark_step.run(persist)
            del persist #try to save memory
            
            #Jump Step    
            jump_step = JumpStep()
            #jump_step.output_dir = output_dir
            #jump_step.save_results = True
            jump_step.rejection_threshold = 15
            jump_step.maximum_cores = max_cores
            #Call using the dark instance from the previously-run dark current subtraction step
            jump = jump_step.run(dark)
            del dark # try to save memory
                
            #Ramp Fitting    
            ramp_fit_step = RampFitStep()
            ramp_fit_step.maximum_cores = max_cores
            ramp_fit_step.output_dir = output_dir
            ramp_fit_step.save_results = True
            
            #Let's save the optional outputs, in order
            #to help with visualization later
            #ramp_fit_step.save_opt = True
            
            #Call using the dark instance from the previously-run jump step
            ramp_fit = ramp_fit_step.run(jump)
            del jump # try to save memory
            del ramp_fit # try to save memory
            print(stage1_result_file + style.BOLD + " EXISTS" + style.END)
    executionTime_stage1 = (time.time() - startTime_stage1)
    print(style.BOLD + "STAGE 1 COMPLETE: Execution Time " + str(executionTime_stage1) + " seconds"+ style.END)
    
    #splitegrate
    #Splintegrate splits and combines integrations from the pipeline up.
    #Set flipToDet = False to not flip the x-axis
    #This is a step required if running tshirt
    #This simulation has multiple segments that need to be split for tshirt purposes.
    print(style.BOLD + "SPLINTEGRATE IN PROGRESS" + style.END)
    splintegrate_output_dir = output_dir+'splintegrate'                #Defining a splintegrate directory
    if os.path.exists(splintegrate_output_dir) == False:               #If the splintegrate output directory doesn't exist, create
        os.makedirs(splintegrate_output_dir)
    if (len(glob.glob(splintegrate_output_dir+"/*.fits")) == nints_splintegrate):
        None
    else:
        for rateints_segment in glob.glob(output_dir + '*1_rampfitstep.fits'): #Grabbing stage 1 results
            splint = splintegrate.splint(inFile=rateints_segment, outDir=splintegrate_output_dir, flipToDet=False, overWrite=True)
            splint.split()
    print(style.BOLD + "SPLINTEGRATE COMPLETE" + style.END)
    
    #PLOTTING STAGE 1 RESULTS
    if (showPlot == True): #Plotting the Stage 1 results 
        if (len(stage1_result_files)==1):
            fig, axs = plt.subplots(figsize=(20,15))
            for file in zip(stage1_result_files):
                print(file)
                HDUList_plot = fits.open(file)
                HDUList_plot.info
                image2D = HDUList_plot[1].data[0]
                axs.imshow(image2D, vmin=0, vmax=3000)
            fig.suptitle("Stage 1 2D Countrate Products (" + str(len(stage1_result_files)) + " Exposures)", size=16, y=0.9)

        else: 
            fig, axs = plt.subplots(len(stage1_result_files), 1, figsize=(20,15))
            for file, i in zip(stage1_result_files, range(len(stage1_result_files))):
                HDUList_plot = fits.open(file)
                HDUList_plot.info
                image2D = HDUList_plot[1].data[0]
                axs[i].imshow(image2D, vmin=0, vmax=3000)
            fig.suptitle("Stage 1 2D Countrate Products (" + str(len(stage1_result_files)) + " Exposures)", size=16, y=0.9)
    else:
        None

    #𝐀𝐬𝐬𝐨𝐜𝐢𝐚𝐭𝐢𝐨𝐧 Files; Organizing Stage 1 Output Files: Associations are basically just lists of things, mostly exposures, that are somehow related. An association file is a JSON-format file that contains a list of all the files with the same instrument set-up (filter, observation mode, etc) that might be combined into a single image. Relationships between multiple exposures are captured in an association, which is a means of identifying a set of exposures that belong together and may be dependent upon one another. The association concept permits exposures to be calibrated, archived, retrieved, and reprocessed as a set rather than as individual objects.

    asn_dir = output_dir #Name the association file's directory.
    level2_asn = (os.path.join(asn_dir, '{}_level2_asn.json'.format(short_Detector))) #Name the stage 2 association file and give it a path.
    asn_stage2 = asn_from_list(stage1_result_files,rule=DMSLevel2bBase) #The rateints files; DMSLevel2bBase indicates that a Level2 association is to be created.
    with open(level2_asn, 'w') as fh: #Write an association file.
        fh.write(asn_stage2.dump()[1])
        
    print(style.BOLD + "STAGE 2 IN PROGRESS" + style.END)
    
    stage2_result_files = [] #Calibrated Result Files
    startTime_stage2 = time.time() #Time how long stage 2 takes

    for rateFile in os.listdir(output_dir):
        if rateFile.endswith("_1_rampfitstep.fits"):
            stage2_result_file = rateFile.replace("1_rampfitstep", "1_calints")
            stage2_result_files.append(output_dir+stage2_result_file)
    for stage2_result_file in stage2_result_files:
            if (os.path.exists(stage2_result_file) == True) and (recalculate == False):
                print(stage2_result_file+ style.BOLD + " EXISTS" + style.END)
            elif (os.path.exists(stage2_result_file) == False) :
                print(stage2_result_file+" one file didn't exist or recalculation is TRUE ... calibrating exposures")
                
                #The file to use is the stage 2 association file defined above. 
                #Instantiate the class. Do not provide a configuration file.
                pipeline_stage2 = Image2Pipeline()
        
                #Specify that you want results saved to a file
                pipeline_stage2.save_results = True
                pipeline_stage2.output_dir = output_dir
                
                #Execute the pipeline using the run method
                if (len(stage1_result_files) != 1): #If there is only one raw file, can't make asm file for STAGE 2
                    result_stage2 = pipeline_stage2.run(level2_asn)
                elif (len(stage1_result_files) == 0):
                    print("No STAGE 1 Files Found")
                    break
                else:
                    result_stage2 = pipeline_stage2.run(stage1_result_files[0])
                    break
    executionTime_stage2 = (time.time() - startTime_stage2)
    print(style.BOLD + "STAGE 2 COMPLETE: Execution Time "+ str(executionTime_stage2) + " seconds" + style.END)
    
    
    #Generate an association file required for Stage 3
    level3_asn = (os.path.join(asn_dir, '{}_level3_asn.json'.format(short_Detector))) #Name the stage 3 association file and give it a path.
    asn_stage3 = asn_from_list(stage2_result_files, product_name ='{}_{}_level3_asn'.format(TargetName, short_Detector)) #The rateints files; Name the output.
    with open(level3_asn, 'w') as fh: #Write an association file.
        fh.write(asn_stage3.dump()[1])
    
    startTime_stage3 = time.time() #Time how long this step takes
    
    if (STAGE3 == 'JWST'): #Preform only if for JWST Pipeline
        print(style.BOLD + "STAGE 3 (JWST Pipeline) IN PROGRESS" + style.END)
        
        if (pupil == 'WLP8') or (filter_element == 'WLP4'): #Only works for WLP4&WLP8 for now
            if (radii_values==[None, None, None]): #Use default values from JWST Pipeline
                print("Using Default Radii Parameters for pupil-filter combo: "+ str(pupil)+ "+" + str(filter_element))
                stage3_results_dir = output_dir #Default Values will be saved to output_dir      

                original_tsophot=asdf.open(os.environ['CRDS_PATH']+'/references/jwst/nircam/jwst_nircam_tsophot_0001.asdf') #the original tsophot reference file
                if (os.path.exists(output_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)) == True) and (recalculate == False):
                    print(output_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)+ style.BOLD + " EXISTS" + style.END)
                else:
                    print(output_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)+" does't exist or recalculation is TRUE ... calculating photometry results") 
                    
                    #The file to use is the stage 3 association file defined above. 
                    #Instantiate the class. Do not provide a configuration file.
                    pipeline_stage3 = Tso3Pipeline()
                    pipeline_stage3.outlier_detection.skip = True
                    
                    #Specify that you want results saved to a file
                    pipeline_stage3.save_results = True
                    pipeline_stage3.output_dir = stage3_results_dir
                    
                    #Execute the pipeline using the run method
                    result_stage3 = pipeline_stage3.run(level3_asn) 
                    print(output_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)+ style.BOLD + " EXISTS" + style.END)
                
            else: #Using Input Radii Values
                radius_src = radii_values[0]
                radius_inner = radii_values[1]
                radius_outer = radii_values[2]
                
                stage3_results_dir = output_dir+str(radius_src)+str(radius_inner)+str(radius_outer)+"_radii/"      
                if os.path.exists(stage3_results_dir) == False:       #If the output directory does not exist, create it
                    os.makedirs(stage3_results_dir)
                    
                print("Using Given Radii Values "+ str(radius_src)+str(radius_inner)+str(radius_outer) + " for pupil-filter combo: " + str(pupil)+ "+" + str(filter_element))
                
                original_tsophot=asdf.open(os.environ['CRDS_PATH']+'/references/jwst/nircam/jwst_nircam_tsophot_0001.asdf') #the original tsophot reference file
                original_tsophot.tree #print the original tsophot reference file
                
                #adjust the radii parameters
                original_tsophot.tree['radii'] = [{'pupil': 'WLP8',
                                                   'radius': radius_src,
                                                   'radius_inner': radius_inner,
                                                   'radius_outer': radius_outer}, 
                                                  {'pupil':'ANY','radius':radius_src,'radius_inner':radius_inner,'radius_outer': radius_outer}]
                original_tsophot.write_to(stage3_results_dir+'adjusted_jwst_nircam_tsophot_0001.asdf')
                adjusted_tsophot=asdf.open(stage3_results_dir+'adjusted_jwst_nircam_tsophot_0001.asdf') #the adjusted tsophot reference file
                
                if (os.path.exists(stage3_results_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)) == True) and (recalculate == False):
                    print(stage3_results_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)+ style.BOLD + " EXISTS" + style.END)
                else:
                    print(stage3_results_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)+" does't exist or recalculation is TRUE ... calculating photometry results") 
                    
                    #The file to use is the stage 3 association file defined above. 
                    #Instantiate the class. Do not provide a configuration file.
                    pipeline_stage3 = Tso3Pipeline()
                    pipeline_stage3.outlier_detection.skip = True
                    pipeline_stage3.tso_photometry.override_tsophot = stage3_results_dir+'adjusted_jwst_nircam_tsophot_0001.asdf' #use the modified tso_phot ref file
                    
                    #Specify that you want results saved to a file
                    pipeline_stage3.save_results = True
                    pipeline_stage3.output_dir = stage3_results_dir
                    
                    #Execute the pipeline using the run method
                    result_stage3 = pipeline_stage3.run(level3_asn) 
                    print(stage3_results_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)+ style.BOLD + " EXISTS" + style.END)
                    
        #Standard Deviation Range: We ideally want only segment 001. 
        if (len(glob.glob(output_dir+'splintegrate/*seg001*.fits')) == 0):
            seg01_len = 20
        else:
            seg01_len = len(glob.glob(output_dir+'splintegrate/*seg001*.fits'))
            
        stage3_calculation_file = stage3_results_dir+"{}_{}_JWST_Pipeline_Photometry_Calculations".format(TargetName, short_Detector)
        #Create a table for these error results and save them to the stage3_calculation_file defined previously. 
        dat_stage3 = Table()
            
        #JWST pipeline: net aperture
        dat = ascii.read(stage3_results_dir+'{}_{}_level3_asn_phot.ecsv'.format(TargetName, short_Detector)) #Call the data
        normalized_net_aperture_sum_pipeline = dat['net_aperture_sum'].value/dat['net_aperture_sum'][0].value #norm net aperture sum
        std_net_aperture_sum_pipeline = np.std(normalized_net_aperture_sum_pipeline[0:seg01_len]) #calculated standard deviation
        relative_error_net_aperture_sum_pipeline = (dat['net_aperture_sum_err'].value/dat['net_aperture_sum'].value)
        #MAD*1.48 Calculation: 
        deviation_netap = normalized_net_aperture_sum_pipeline[0:seg01_len] - np.median(normalized_net_aperture_sum_pipeline[0:seg01_len])
        mad_netap = np.median(np.abs(deviation_netap))*1.48
        
        print(style.BOLD+"JWST Pipeline Calculated Net Aperture Sum MAD*1.48 (ppm):"+style.END + " " +str(mad_netap*10**6))
        print(style.BOLD+"JWST Pipeline Calculated Net Aperture Sum std (ppm):"+style.END + " " +str(std_net_aperture_sum_pipeline*10**6))
        print(style.BOLD+"JWST Pipeline Median Relative Error (Theoretical) Net Aperture Sum (ppm):"+style.END + " " +str(np.median(relative_error_net_aperture_sum_pipeline)*10**6)) #ppm

      
        dat_stage3['JWST Pipeline Calculated Net Aperture Sum MAD*1.48 (ppm)'] = [mad_netap*10**6]
        dat_stage3['JWST Pipeline Calculated Net Aperture Sum std (ppm)'] = [std_net_aperture_sum_pipeline*10**6]
        dat_stage3['JWST Pipeline Median Relative Error (Theoretical) Net Aperture Sum (ppm)'] = [np.median(relative_error_net_aperture_sum_pipeline)*10**6]
        
        #JWST pipeline: aperature background
        normalized_aperture_bkg_pipeline = dat['aperture_bkg'].value/dat['aperture_bkg'][0].value #normalized aperture bkg
        std_aperture_bkg_pipeline = np.std(normalized_aperture_bkg_pipeline[0:seg01_len]) #calculated standard deviation
        relative_error_aperture_bkg_pipeline = (dat['aperture_bkg_err'].value/dat['aperture_bkg'].value)
        #MAD*1.48 Calculation: 
        deviation_apbkg = normalized_aperture_bkg_pipeline[0:seg01_len] - np.median(normalized_aperture_bkg_pipeline[0:seg01_len])
        mad_apbkg = np.median(np.abs(deviation_apbkg))*1.48
        
        dat_stage3['JWST Pipeline Calculated Aperture Background MAD*1.48 (ppm)'] = [mad_apbkg*10**6]
        dat_stage3['JWST Pipeline Calculated Aperture Background std (ppm)'] = [std_aperture_bkg_pipeline*10**6]
        dat_stage3['JWST Pipeline Median Relative Error (Theoretical) Aperture Background (ppm)'] = [np.median(relative_error_aperture_bkg_pipeline)*10**6]
        
        #JWST pipeline: annulus mean
        normalized_annulus_mean_pipeline = dat['annulus_mean'].value/dat['annulus_mean'][0].value #normalized annulus mean
        std_annulus_mean_pipeline = np.std(normalized_annulus_mean_pipeline[0:seg01_len]) #calculated standard deviation
        relative_error_annulus_mean_pipeline = (dat['annulus_mean_err'].value/dat['annulus_mean'].value) #theoretical error
        #MAD*1.48 Calculation: 
        deviation_ansmean = normalized_annulus_mean_pipeline[0:seg01_len] - np.median(normalized_annulus_mean_pipeline[0:seg01_len])
        mad_ansmean = np.median(np.abs(deviation_ansmean))*1.48
        
        dat_stage3['JWST Pipeline Calculated Annulus Mean MAD*1.48 (ppm)'] = [mad_ansmean*10**6]
        dat_stage3['JWST Pipeline Calculated Annulus Mean std (ppm)'] = [std_annulus_mean_pipeline*10**6]
        dat_stage3['JWST Pipeline Median Relative Error (Theoretical) Annulus Mean (ppm)'] = [np.median(relative_error_annulus_mean_pipeline)*10**6]
        
        #JWST pipeline: annulus sum
        normalized_annulus_sum_pipeline = dat['annulus_sum'].value/dat['annulus_sum'][0].value #normalized annulus sum
        std_annulus_sum_pipeline = np.std(normalized_annulus_sum_pipeline[0:seg01_len]) #calculated standard deviation
        relative_error_annulus_sum_pipeline = (dat['annulus_sum_err'].value/dat['annulus_sum'].value)
        #MAD*1.48 Calculation: 
        deviation_anssum = normalized_annulus_sum_pipeline[0:seg01_len] - np.median(normalized_annulus_sum_pipeline[0:seg01_len])
        mad_anssum = np.median(np.abs(deviation_anssum))*1.48
        
        dat_stage3['JWST Pipeline Calculated Annulus Sum MAD*1.48 (ppm)'] = [mad_anssum*10**6]
        dat_stage3['JWST Pipeline Calculated Annulus Sum std (ppm)'] = [std_annulus_sum_pipeline*10**6]
        dat_stage3['JWST Pipeline Median Relative Error (Theoretical) Annulus Sum (ppm)'] = [np.median(relative_error_annulus_sum_pipeline)*10**6]
        
        dat_stage3.write(stage3_calculation_file, format='csv', overwrite=True) #write JWST pipline photometry results for csv file

        if (showPlot == True): #plot the JWST pipeline results
            fig, axs = plt.subplots(2, 2,figsize=(15,10))
            axs[0,0].errorbar(dat['MJD'],normalized_net_aperture_sum_pipeline,yerr=relative_error_net_aperture_sum_pipeline,color='navy',fmt='.',markersize=4,elinewidth=1,ecolor='silver',label='Pipeline')
            axs[0,0].set_title("Net Aperture Sum")
            axs[0,0].set_xlabel("Time (MJD)")
            axs[0,0].set_ylabel("Normalized Flux")
            axs[0,0].legend()
            axs[0,1].errorbar(dat['MJD'],normalized_aperture_bkg_pipeline,yerr=relative_error_aperture_bkg_pipeline,color='darkred',fmt='.',markersize=4,elinewidth=1,ecolor='silver',label='Pipeline')
            axs[0,1].set_title("Aperture Background")
            axs[0,1].set_xlabel("Time (MJD)")
            axs[0,1].set_ylabel("Normalized Flux")
            axs[0,1].legend()
            
            axs[1,0].errorbar(dat['MJD'],normalized_annulus_mean_pipeline,yerr=relative_error_annulus_mean_pipeline,color ='darkgreen',fmt='.',markersize=4,elinewidth=1,ecolor='silver',label='Pipeline')
            axs[1,0].set_title("Annulus Mean")
            axs[1,0].set_xlabel("Time (MJD)")
            axs[1,0].set_ylabel("Normalized Flux")
            axs[1,0].legend() 
            
            axs[1,1].errorbar(dat['MJD'],normalized_annulus_sum_pipeline,yerr=relative_error_annulus_sum_pipeline,color = 'indigo',fmt='.',markersize=4,elinewidth=1,ecolor='silver',label='Pipeline')
            axs[1,1].set_title("Annulus Sum")
            axs[1,1].set_xlabel("Time (MJD)")
            axs[1,1].set_ylabel("Normalized Flux")
            axs[1,1].legend()
                        
    elif (STAGE3 == 'tshirt'):
        print(style.BOLD + "STAGE 3 (tshirt Pipeline) IN PROGRESS" + style.END)
        if(radii_values ==[None, None, None]) and (sweep == False):
            stage3_results_dir = output_dir
        elif (sweep==True):
            stage3_results_dir = output_dir+"Sweep_Results/"
            if os.path.exists(stage3_results_dir) == False:       #If the output directory does not exist, create it
                os.makedirs(stage3_results_dir)
        else: #Using Given Radii Values
            radius_src = radii_values[0]
            radius_inner = radii_values[1]
            radius_outer = radii_values[2]
                
            stage3_results_dir = output_dir+str(radius_src)+str(radius_inner)+str(radius_outer)+"_radii/"      
            if os.path.exists(stage3_results_dir) == False:       #If the output directory does not exist, create it
                os.makedirs(stage3_results_dir)
        
        if (os.path.exists(stage3_results_dir+'{}_{}_ROEBA_phot_pipeline.yaml'.format(TargetName, short_Detector)) == True):
                    print(stage3_results_dir+'{}_{}_ROEBA_phot_pipeline.yaml'.format(TargetName, short_Detector)+ style.BOLD + " EXISTS" + style.END)
        else:
            print(stage3_results_dir+'{}_{}_ROEBA_phot_pipeline.yaml'.format(TargetName, short_Detector)+" does't exist ... creating default parameter YAML file. Feel free to alter the parameters once the file has been created and run this pipeline again.") 
            #Generating a default parameter file
            
            defaultParams = {'procFiles': output_dir+'splintegrate/*.fits',
                             'excludeList': None,
                             'srcName': str(TargetName)+'_'+str(short_Detector),
                             'nightName': "{}_{}_{}".format(short_Detector, TargetName, date_obs), 
                             'refStarPos': [[XREF_SCI, YREF_SCI]],
                             'apRadius': 3, 
                             'backStart': 4,
                             'backEnd': 5, 
                             'refPhotCentering': None, 
                             'copyCentroidFile': None, 
                             'srcGeometry': 'Circular', 
                             'bkgSub': True,
                             'bkgGeometry': 'CircularAnnulus', 
                             'bkgMethod': 'mean', 
                             'backOffset': [0.,0.],
                             'boxFindSize': 5, 
                             'jdRef': 2457551, 
                             'timingMethod': 'JWSTint', 
                             'scaleAperture': False, 
                             'apScale': 2.5 ,
                             'apRange': [2,17], 
                             'isCube': False, 
                             'cubePlane': 0, 
                             'doCentering': True , 
                             'FITSextension': 0,
                             'HEADextension': 0, 
                             'isSlope': True, 
                             'subpixelMethod': 'exact',
                             'readNoise': 16.2, 
                             'detectorGain': 2.05,
                             'dateFormat': 'Two Part', 
                             'diagnosticMode':  False, 
                             'bkgOrderX': 0, 
                             'bkgOrderY': 0, 
                             'backsub_directions': ['X'], 
                             'saturationVal': None, 
                             'satNPix': 5, 
                             'nanReplaceValue': 22e3}
            #Write the default parameter file
            with open(stage3_results_dir+'{}_{}_ROEBA_phot_pipeline.yaml'.format(TargetName, short_Detector), 'w') as file:
                paramfile = yaml.dump(defaultParams, file)
            
        default_phot = phot_pipeline.phot(paramFile=stage3_results_dir+'{}_{}_ROEBA_phot_pipeline.yaml'.format(TargetName, short_Detector))
        
        alteredParam = deepcopy(default_phot.param)
        
        if (sweep==True):
            alteredParam['srcNameShort'] = str(TargetName)+"_sweep"
            print(style.BOLD+"Sweep of Aperture Sizes in PROGRESS"+style.END)
            #Weonly want the first segment. 
            if (len(glob.glob(output_dir+'splintegrate/*seg001*.fits')) == 0):
                alteredParam['procFiles'] = output_dir+'splintegrate/*seg002*.fits'
            elif (len(glob.glob(output_dir+'splintegrate/*seg002*.fits')) == 0):
                print(style.BOLD+"Segment 1 & segment 2 missing from stage 1 results. Required for Aperture sWweep."+style.END)
            else:
                 alteredParam['procFiles'] = output_dir+'splintegrate/*seg001*.fits'
        elif (radii_values == [None, None, None]):
            if (pupil == 'WLP8'):
                radius_src = 50
                radius_inner = 60
                radius_outer = 70
                alteredParam['srcNameShort'] = TargetName+"_"+str(radius_src)+str(radius_inner)+str(radius_outer)+"_radii"
                alteredParam['apRadius'] = 50 #Default source aperture radius
                alteredParam['backStart'] = 60 #Background annulus start radius
                alteredParam['backEnd'] = 70 #Background annulus end radius
            elif (filter_element == 'WLP4'):
                radius_src = 3
                radius_inner = 4
                radius_outer = 5
                alteredParam['srcNameShort'] = TargetName+"_"+str(radius_src)+str(radius_inner)+str(radius_outer)+"_radii" 
                alteredParam['apRadius'] = 3 #Default source aperture radius
                alteredParam['backStart'] = 4 #Background annulus start radius
                alteredParam['backEnd'] = 5 #Background annulus end radius
            else: 
                print("Unsupported Filter + Pupil Combination")
                
        else:
            alteredParam['srcNameShort'] = str(TargetName)+"_"+str(radius_src)+str(radius_inner)+str(radius_outer)+"_radii"
            alteredParam['apRadius'] = radii_values[0] #Input source aperture radius
            alteredParam['backStart'] = radii_values[1] #Input Background annulus start radius
            alteredParam['backEnd'] = radii_values[2] #Input Background annulus end radius
            
        #Re-write the parameter file with these updates for the radii and shortName   
        with open(stage3_results_dir+'{}_{}_ROEBA_phot_pipeline.yaml'.format(TargetName, short_Detector), 'w') as file:
                paramfile = yaml.dump(alteredParam, file)
        
        if(sweep==True):
            phot_sweep = phot_pipeline.phot(directParam=alteredParam) #create new photometric object
            phot_sweep.get_allimg_cen(recenter=True,useMultiprocessing=useMultiprocessing) #recenter the centroids each time. 
            phot_sweep.do_phot(useMultiprocessing=useMultiprocessing) #extract the photometric data
            
            #Coarse Grid Sweep: 
            BaseDir_tshirt = phot_pipeline.get_baseDir()
            Coarse_results = BaseDir_tshirt+"/tser_data/phot_aperture_optimization/aperture_opt_{}_sweep_aperture_sizing_{}_src_5_150_step_10_back_5_150_step_10.csv".format(TargetName, phot_sweep.param['nightName'])
            
            if (os.path.exists(Coarse_results) == True):
                print(str(Coarse_results)+ " EXISTS")
            else: 
                print(str(Coarse_results)+ " does not EXIST ... calculating")
                sweep_analysis_coarse = analysis.aperture_size_sweep(phot_sweep,stepSize=10,srcRange=[5,150],backRange=[5,150],minBackground=10) #coarse sweep
                
            sweep_coarse_results = pd.read_csv(Coarse_results)
            MAD_min_idx_coarse = abs(sweep_coarse_results[['mad_arr']]).idxmin()
            best_coarse_radii = sweep_coarse_results.iloc[MAD_min_idx_coarse]

            best_src_radius_coarse = float(sweep_coarse_results.iloc[MAD_min_idx_coarse]['src'].to_string(index=False))
            best_inner_radius_coarse = float(sweep_coarse_results.iloc[MAD_min_idx_coarse]['back_st'].to_string(index=False))
            best_outer_radius_coarse = float(sweep_coarse_results.iloc[MAD_min_idx_coarse]['back_end'].to_string(index=False))
            print(style.BOLD+"Coarse Sweep Results: "+str(best_src_radius_coarse)+ ", "+ str(best_inner_radius_coarse)+ ", "+ str(best_outer_radius_coarse)+style.END)
                
            #New Fine Radius Values:
            best_src_radius_upperlimit = best_src_radius_coarse+5
            best_src_radius_lowerlimit =best_src_radius_coarse-5
            
            best_inner_radius_limit = best_inner_radius_coarse-5
            best_outer_radius_limit =best_outer_radius_coarse+5
            
            
            #Fine Sweep: 
            Fine_results = BaseDir_tshirt+"/tser_data/phot_aperture_optimization/aperture_opt_{}_sweep_aperture_sizing_{}_src_{}_{}_step_1_back_{}_{}_step_1.csv".format(TargetName,phot_sweep.param['nightName'],int(best_src_radius_lowerlimit),int(best_src_radius_upperlimit),int(best_inner_radius_limit), int(best_outer_radius_limit))
            
            if (os.path.exists(Fine_results) == True):
                print(str(Fine_results)+ " EXISTS")
            else: 
                print(str(Fine_results)+ " does not EXIST ... calculating")
                sweep_analysis_fine = analysis.aperture_size_sweep(phot_sweep,stepSize=1,srcRange=[int(best_src_radius_lowerlimit),int(best_src_radius_upperlimit)],backRange=[int(best_inner_radius_limit),int(best_outer_radius_limit)],minBackground=2) #fine sweep
                
            analysis.plot_apsizes(Fine_results)
            
            sweep_fine_results = pd.read_csv(Fine_results)
            MAD_min_idx = abs(sweep_fine_results[['mad_arr']]).idxmin()
            best_radii = sweep_fine_results.iloc[MAD_min_idx]

            best_src_radius = float(sweep_fine_results.iloc[MAD_min_idx]['src'].to_string(index=False))
            best_inner_radius = float(sweep_fine_results.iloc[MAD_min_idx]['back_st'].to_string(index=False))
            best_outer_radius = float(sweep_fine_results.iloc[MAD_min_idx]['back_end'].to_string(index=False))
            print(style.BOLD+"Fine Sweep Results: "+str(best_src_radius)+ ", "+ str(best_inner_radius)+ ", "+ str(best_outer_radius)+style.END)
            
            #Altering to the optimal found radii:
            alteredParam['procFiles'] = output_dir+'splintegrate/*.fits'
            alteredParam['apRadius'] = best_src_radius #Changing the source radius
            alteredParam['backStart'] = best_inner_radius #Changing the inner radius
            alteredParam['backEnd'] = best_outer_radius #Changing the outer radius
            alteredParam['srcNameShort'] = '{}_fine_best'.format(TargetName) #provide a new name for centroid realignment
            
            radius_src = best_src_radius
            radius_inner = best_inner_radius
            radius_outer = best_outer_radius
            
            #Assignimg a object for fine sweep reults. 
            print(style.BOLD+"Extracting Photometry"+style.END)
            phot2 = phot_pipeline.phot(directParam=alteredParam) #create new photometric object
            phot2.get_allimg_cen(recenter=True,useMultiprocessing=useMultiprocessing) #recenter the centroids each time.
            phot2.do_phot(useMultiprocessing=useMultiprocessing) #extract the photometric data
            
            if (showPlot==True):
                print(style.BOLD+"Plotting the Apeture Choices & Extracting Photometry for Radii Values: "+str(best_src_radius)+str(best_inner_radius)+str(best_outer_radius)+" ..."+style.END)
                phot2.showStarChoices(showAps=True,showPlot=True,apColor='red',backColor='pink', figSize=(30,20)) #Plot the source and background subtraction area
            else: 
                None
            print(style.BOLD+"Sweep of Aperture Sizes COMPLETE"+style.END)
    
        #If Sweep = False    
        else:
            print(style.BOLD+"Extracting Photometry"+style.END)
            phot2 = phot_pipeline.phot(directParam=alteredParam) #create a photometric object
            phot2.get_allimg_cen(recenter=True,useMultiprocessing=useMultiprocessing) #recenter the centroids each time.
            phot2.do_phot(useMultiprocessing=useMultiprocessing) #extract the photometric data
            
            if (showPlot==True):
                print(style.BOLD+"Plotting the Apeture Choices & Extracting Photometry for Radii Values: "+str(radius_src)+str(radius_inner)+str(radius_outer)+" ..."+style.END)
                phot2.showStarChoices(showAps=True,showPlot=True,apColor='red',backColor='pink', figSize=(30,20)) #Plot the source and background subtraction area
            else: 
                None
                    
        #Standard Deviation Range: We ideally want only segment 001. 
        if (len(glob.glob(output_dir+'splintegrate/*seg001*.fits')) == 0):
            seg01_len = 20
        else:
            seg01_len = len(glob.glob(output_dir+'splintegrate/*seg001*.fits'))
        
        dat_tshirt = Table()
        stage3_calculation_file = stage3_results_dir+"{}_{}_tshirt_Pipeline_Calculations".format(TargetName, short_Detector)

        #Tshirt: net aperture
        Flux2, Flux_error2 = phot2.get_tSeries() #The flux data and flux data errors
        normalized_flux_tshirt2 = Flux2['Flux 0']/Flux2['Flux 0'][0] #normalized net aperture sum
        std_tshirt2 = np.std(normalized_flux_tshirt2[0:seg01_len]) #calculated standard deviation
        relative_error_tshirt2 = (Flux_error2['Error 0']/Flux2['Flux 0'])
        #MAD*1.48 Calculation: 
        deviation_tshirt = normalized_flux_tshirt2[0:seg01_len] - np.median(normalized_flux_tshirt2[0:seg01_len])
        mad_tshirt = np.median(np.abs(deviation_tshirt))*1.48
        
        dat_tshirt['Time (JD)'] = Flux2['Time (JD)']
        dat_tshirt['Normalized Flux'] = normalized_flux_tshirt2
        dat_tshirt['Source Radius']  = radius_src
        dat_tshirt['Inner Background Radius'] = radius_inner
        dat_tshirt['Outer Background Radius'] = radius_outer 
        dat_tshirt['Tshirt Pipeline Calculated Net Aperture Sum MAD*1.48 (ppm)'] = [mad_tshirt*10**6]
        dat_tshirt['Tshirt Pipeline Calculated Net Aperture Sum std (ppm)'] = [std_tshirt2*10**6]
        dat_tshirt['Tshirt Pipeline Calculated Median Relative Error (Theoretical) Net Aperture Sum (ppm)'] = [np.median(relative_error_tshirt2)*10**6]

        print(style.BOLD+"Tshirt Calculated Net Aperture Sum MAD*1.48 (ppm):"+style.END + " " +str(mad_tshirt*10**6))
        print(style.BOLD+"Tshirt Calculated Net Aperture Sum std (ppm):"+style.END + " " +str(std_tshirt2*10**6))
        print(style.BOLD+"Thshirt Median Relative Error (Theoretical) Net Aperture Sum (ppm):"+style.END+" "+str(np.median(relative_error_tshirt2)*10**6))
        
        dat_tshirt.write(stage3_calculation_file, format='csv', overwrite=True)
        
        if (showPlot == True):
            fig = plt.figure()
      
            plt.errorbar(Flux2['Time (JD)'],normalized_flux_tshirt2,yerr=relative_error_tshirt2,fmt='b.',markersize=4,elinewidth=1,ecolor='silver')
            plt.xlabel('Time (JD)', fontsize = 14)
            plt.ylabel('Normalized Flux', fontsize = 14)
            plt.title('Net Aperture Sum',
                fontsize = 20, fontweight ='bold')
            plt.show()
        else: 
            None
    else:
        print("Invalid Input. Input for parameter STAGE3 must be 'JWST' or 'tshirt'.")
        
    executionTime_stage3 = (time.time() - startTime_stage3)
    print(style.BOLD + "STAGE 3 COMPLETE: Execution Time "+ str(executionTime_stage3) + " seconds" + style.END)