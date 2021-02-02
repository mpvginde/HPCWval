#!/usr/bin/env python3

# In this example it is shown how one can calibrate the reference ensemble 
# for a new model or new model version.
# Here we use  400 members/runs of the ICON-O medium testcase
# where the initial temperature was perturbed with perturbation of order O(-16)
# The relevant data is already extracted and saved in a pickle file

# 

import pandas as pd
import pickle
from netCDF4 import Dataset
from HPCWval import *

# Load the reference ensemble:
with open("ICON-O_medium_refENS_400.pkl",'rb') as f:
  ensemble = pickle.load(f)

# Initialize the calibrator
cal = Calibrator(ensemble=ensemble)

# First, set the number of Principle components we will use.
# The cumulatie explain variance versus the number of components can be plotted with
cal.calibrateComp()
# The function setComp() without arguments sets the number of components to that number that
# that explains 90% of the variance
cal.setComp()
# An arbitrary number of components kan be chosen by using the argument nComponents
# cal.setComp(nComponents=20)

# Second, we calibrate the the typical parameters (nNew, nRun, nPc)  used for 
# identifying divergent installations/configurations
# arguments are:
# - nCPU = number of CPUs used (if omitted, the maximum available number of CPUs is used)
# - nRunMax = maximum of the parameter nNew that will be explored, for nNew > 5 computations 
#             become very expensive
# - nPcMax  = maximum of the parameter nPc that will be explored
# output:
#   False positve rate for every possible combination of nNew,nRun and nPc. Typically we are looking
#   for a combination where the FRP is between 0.5% and 1.5% 
print("Calibrating Fail parameters...")
caliFail = cal.calibrateFailParams(nCPU=4,nRunMax=3,nPcMax=3)
print("nNew = 2, rows indicate nRun-1, columns nPc-1")
print(caliFail[0])
print("nNew = 3, rows indicate nRun-1, columns nPc-1")
print(caliFail[1])
print("We choose nNew=3, nRun=2, nPc=2 with FPR = " + str(caliFail[1].iloc[1,1]) + "%" )
cal.setFailParams(nNew=3,nRun=2,nPc=2)

# Next we investigate what is the ideal reference ensemble-size
# You can specify a list of ensemble sizes as argument
# Ouput is a matrix containing 10000 False positive rates for each ensemblesizes 
# and a boxplot of the FPRs for each ensemble size
print("Calibrating ensemble-size...")
caliSize = cal.calibrateEnsSize([200,250])

# Ensemble size is large enough if the shape of the boxplot stabelizes 
# with a median between 0.5 and 1.5
# From these FPRs we can set the limit for rejecting new ensembles
limit = np.percentile(caliSize[200],90)
print("We select ensemble size= 200 and Failure Rate limit " + str(limit) + "%")
cal.setEnsSize(ensSize=200,limitFR=limit)

# Finally we need to set the list of variables to ignore, typically dimensional 
# or constant variables
# These variables are not in the reference ensemble we loaded, 
# but they are in the 'new' outputs.
# Thus, we pass the variables that are in the output file but not in the reference ensembe
# to the setIgnore function
member = Dataset('O16/001/ocean_omip_medium_GLOBALMEANOUT__23000201T000000Z.nc','r')
ignore = [var for var in member.variables if var not in ensemble.columns]
cal.setIgnore(ignore=ignore)

# All the parameters of the reference ensemble can then be export to a pickle file
# which can be read by the validator for validating new installations of the model.
model = cal.export(modelName="ICON-O_medium_test",dir="./")


# Start the validation of a 'new' installation 
val = Validator(model = model)
# Set the name of the outputfiles
val.setOname(oname = "ocean_omip_medium_GLOBALMEANOUT__23000201T000000Z.nc")
newMembers=range(1,31)
val.loadEnsemble(members=newMembers,dir="./O16")
val.validate()

