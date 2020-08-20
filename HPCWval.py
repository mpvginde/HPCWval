from netCDF4 import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import combinations
import random
import pickle

class Validator:
  def __init__(self,model=None,**kw):
    '''Initalize Validator
    Args:
        model: choose the model, possible options: ICON-O, ICON-A, IFS, IFSVM, NEMO, new or load. 
        For each model different testcases are available, run without testcase keyword 
        to get info on the available testcases for a given model.
        testcase: (optional) select the testcase for a given model (not needed for model = 'new' or 'load')
    Returns:
        Validator object.
    ''' 
    if model == None:
      raise ValueError('choose the model, possible options: ICON-O, ICON-A, IFS, IFSVM or NEMO.')
    self.__Msigma = 2
    self.__ignore = []
    testcase = kw.get('testcase', None)
    if model == 'ICON-O':
      if testcase == 'small':
        print('to be implemented...')
      elif testcase == 'medium':
        print('Validator for ICON-O medium testcase')
        filename = model + '_' + testcase + '.pkl'
        with open(filename,'rb') as f:
          pkl = pickle.load(f)
        self.__ignore = pkl[0]
        self.__sc = pkl[1]
        self.__pca = pkl[2]
        self.nNew=3
        self.nRun=2
        self.nPc=2
        self.__limit = 3.448275
        self.__n_components = self.__pca.n_components_
        self.__iname = "ocean_omip_medium_OUT__23000201T000000Z.nc"
        self.__oname = "ocean_omip_medium_GLOBALMEANOUT__23000201T000000Z.nc"
      elif testcase == 'large':
        print('to be implemented...')
      else: 
        raise Exception("please select a testcase for ICON-O \n available options are: small, medium, large.")
    elif model == 'ICON-A':
      print('to be implemented...')
    elif model == 'IFS':
      print('to be implemented...')
    elif model == 'IFSFVM':
      print('to be implemented...')
    elif model == 'NEMO':
      print('to be implemented...')
    elif model == 'new':
      print('Setting up Validator for a new model and/or testcase \n use calibrate(ensemble) to calibrate the validator')

  def __isFail(self,input):
    '''Checks if nPC number of identical Principle Components deviate more than Msigma standard deviations
       from the reference Principle components for nRun new ensemble members.
       nPc and nNew are class-variables
    Args:
        input: boolean matrix of shape (nEnsemblemembers,n_components)
    Returns:
        boolean: True if the ensemble fails, False if ensemble does not fail
    '''
    candidates =  input[np.sum(input,axis=1) >= self.nPc ,:]
    output = [ np.sum(np.prod(candidates[index,:],axis=0)) >= self.nPc for index in combinations(range(candidates.shape[0]),self.nRun) ] 
    return (any(output))

  def __EET(self,ensNEW,**kwargs):
    ''' Returns Failure Rate (FR) for all combinations of nNew out of the ensNEW-members,
        uses the ensREF for pca and sc calculations for model='new' else the loaded class-wide pca and sc
        are used.
    Args:
        ensNEW: new ensemble to be validated shape (nMembers,nVariables)
        ensREF: (optional) reference ensemble used for scaling and PCA
    returns:
        Failure Rate (in percentage)
    '''
    ensREF = kwargs.get('ensREF',None)
    if (ensREF is not None):
      sc = StandardScaler()
      pca = PCA(n_components = self.__n_components)
      refENS_sc = sc.fit_transform(refENS)
      pca.fit(refENS_sc)
    else:
      sc = self.__sc
      pca = self.__pca
    ensNEW_sc = sc.transform(ensNEW)
    pcaNEW = pca.transform(ensNEW_sc)
    x = np.abs(pcaNEW/np.sqrt(pca.explained_variance_)[None,:]) > self.__Msigma
    fails = [ self.__isFail(x[index,:]) for index in combinations(range(x.shape[0]),self.nNew) ]
    return((np.sum(fails)/len(fails))*100)


  def loadEnsemble(self,members, dir=None, timestep=-1, saveCDO = False,calcGlobalMean = False):
    if calcGlobalMean:
      try:
        from cdo import Cdo
      except:
        print('Error: cdo module not found!')
      cdo = Cdo()
    else:
      print("Loading global means...")
    first_member = True
    result = np.array([])
    for member in members:
      ifile = '/'.join(filter(None,(dir,str(member).zfill(3),self.__iname)))
      ofile = '/'.join(filter(None,(dir,str(member).zfill(3),self.__oname)))
      if calcGlobalMean:
        print("calculating values for member: " + str(member).zfill(3) + " ...")
        if saveCDO:
          final = cdo.fldmean(input='-vertavg %s'%(ifile), options='-P 16',output=ofile)
        else: 
          final = cdo.fldmean(input='-vertavg %s'%(ifile), options='-P 16') 
        nc_fid = Dataset(final,"r")
      else:
        nc_fid = Dataset(ofile,'r')
      while first_member:
        if len(self.__ignore) == 0:
          self.__ignore = [ dim for dim in nc_fid.dimensions ]
          self.__ignore.append('time_bnds') if 'time_bnds' in nc_fid.variables else self.__ignore 
          self.__ignore.append('wet_c') if 'wet_c' in nc_fid.variables else self.__ignore
          dims.append('wet_e') if 'wet_e' in nc_fid.variables else self.__ignore
        vrbls = [ var for var in nc_fid.variables if var not in self.__ignore ]
        first_member = False
      vals = np.stack([ nc_fid.variables[var][:].flatten() for var in vrbls ],axis=0)
      result = np.dstack([result,vals]) if result.size else vals
      nc_fid.close()
    if timestep < 0:
      df = pd.DataFrame(np.mean(result,axis=1),index=vrbls).transpose()
    else:  
      df = pd.DataFrame(result[:,timestep,:],index=vrbls).transpose()
    self.ensNEW = df
    return(df)
 
  def validate(self, *args):
    if args:
      FR = self.__EET(args[0])
    else:
      FR = self.__EET(self.ensNEW) 
    print('The failure rate for the new ensemble is: {0:5.2f}%'.format(FR))
    if FR <= self.__limit:
      print(' - SUCCES!')
      print(' -- The new ensemble can be considered statistically equivalent to the reference ensemble.')
    else:
      print(' - FAILURE!')
      print(' -- The new ensemble is not statistically equivalent to the reference ensemble.')
      print(' --- The failure rate should not exceed {0:5.2f}%'.format(self.__limit))
     

def main():
  print('This is the main function')

if __name__ == "__main__":
  main()

