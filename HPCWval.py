from netCDF4 import Dataset
import numpy as np
import pandas as pd
from itertools import combinations
import random
import pickle


def loadEnsemble(members, dir=None, ignore=[], timestep=-1, saveCDO = False,calcGlobalMean = False, iname = None, oname = None):
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
      ifile = '/'.join(filter(None,(dir,str(member).zfill(3),iname)))
      ofile = '/'.join(filter(None,(dir,str(member).zfill(3),oname)))
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
        if len(ignore) == 0:
          ignore = [ dim for dim in nc_fid.dimensions ]
        vrbls = [ var for var in nc_fid.variables if var not in ignore ]
        first_member = False
      vals = np.stack([ nc_fid.variables[var][:].flatten() for var in vrbls ],axis=0)
      result = np.dstack([result,vals]) if result.size else vals
      nc_fid.close()
    if timestep < 0:
      df = pd.DataFrame(np.mean(result,axis=1),index=vrbls).transpose()
    else:  
      df = pd.DataFrame(result[:,timestep,:],index=vrbls).transpose()
    return(df)


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
        self.__Vmu = pkl[1]
        self.__Vsigma = pkl[2]
        self.__Vmu2 = pkl[3]
        self.__P = pkl[4]
        self.__Ssigma = pkl[5]
        self.nNew=3
        self.nRun=2
        self.nPc=2
        self.__limit = 3.448275
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
      from sklearn.preprocessing import StandardScaler
      from sklearn.decomposition import PCA
      sc = StandardScaler()
      pca = PCA(n_components = self.__n_components)
      refENS_sc = sc.fit_transform(refENS)
      pca.fit(refENS_sc)
    else:
      Vmu    = self.__Vmu
      Vsigma = self.__Vsigma
      Vmu2   = self.__Vmu2
      P      = self.__P
      Ssigma = self.__Ssigma
    pcaNEW = np.dot((((ensNEW-Vmu)/Vsigma).fillna(0.0)-Vmu2),P)
    x = np.abs(pcaNEW/Ssigma) > self.__Msigma
    fails = [ self.__isFail(x[index,:]) for index in combinations(range(x.shape[0]),self.nNew) ]
    return((np.sum(fails)/len(fails))*100)

  def loadEnsemble(self,members,dir=None, timestep=-1, saveCDO = False, calcGlobalMean = False):
    self.ensNEW = loadEnsemble(members = members, dir = dir, ignore= self.__ignore, timestep=timestep, saveCDO = saveCDO ,calcGlobalMean = calcGlobalMean, iname = self.__iname, oname = self.__oname) 

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

class Calibrator:
  def __init__(self,**kwargs):
    '''Initalize Calibrator 
     Args:
         ensemble (optional):  the ensemble used for calibrating in pandas.dataframe format [#members, #variables]. 
     Returns:
        Calibrator object.
    '''
    self.ensemble = kwargs.get('ensemble',None)
    self.ensSize = None
    self.nNew    = None
    self.nRun    = None
    self.nPc     = None
    self.nComponents = None

  def _isFail(self,input,nRun,nPc):
    candidates =  input[np.sum(input,axis=1) >= nPc ,:]
    output = [ np.sum(np.prod(candidates[index,:],axis=0)) >= nPc for index in combinations(range(candidates.shape[0]),nRun) ]
    return (any(output))

  def _EET(self,ensNEW,transform,nNew,nRun,nPc,Msigma = 2):
    pcaNEW = np.dot((((ensNEW-transform[0])/transform[1]).fillna(0.0)-transform[2]),transform[3])
    x = np.abs(pcaNEW/transform[4]) > Msigma
    fails = [ self._isFail(input =x[index,:],nRun = nRun, nPc = nPc) for index in combinations(range(x.shape[0]),nNew) ]
    return(np.sum(fails)/len(fails)*100)

  def _simpleEET(self,param,ensemble,n_components,Msigma = 2):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    print('starting ' + str(param) + '..')
    nNew, nRun, nPc = param
    size = int(5 * np.ceil(len(ensemble.columns)*1.5/5))
    temp = []
    for _ in range(100):
      ensREF    = ensemble.sample(size)
      ensNEW    = ensemble.drop(ensREF.index).sample(30)
      scaler    = StandardScaler()
      ensREF_sc = scaler.fit_transform(ensREF)
      pca       = PCA(n_components = n_components).fit(ensREF_sc)
      ensNEW_sc = scaler.transform(ensNEW)
      pcaNEW    = pca.transform(ensNEW_sc)
      x         = np.abs(pcaNEW/np.sqrt(pca.explained_variance_)[None,:]) > Msigma
      fails = [ self._isFail(input =x[index,:],nRun = nRun, nPc = nPc) for index in combinations(range(x.shape[0]),nNew) ]
      temp.append(np.sum(fails)/len(fails)*100)
    print(str(param) + ' finished')
    return(np.mean(temp))

  def _fullEET(self,iREF,ensemble, n_components, nNew, nRun, nPc, Msigma = 2):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    ensREF    = ensemble.values[iREF,:]
    scaler    = StandardScaler()
    ensREF_sc = scaler.fit_transform(ensREF)
    pca       = PCA(n_components = n_components).fit(ensREF_sc)
    ensREST   = np.delete(arr=ensemble.values, obj=iREF,axis=0)
    iNEWs     = [ random.sample(range(ensREST.shape[0]),30) for _ in range(100) ]
    output    = []
    for iNEW in iNEWs:
      ensNEW_sc = scaler.transform(ensREST[iNEW,:])
      pcaNEW    = pca.transform(ensNEW_sc)
      x         = np.abs(pcaNEW/np.sqrt(pca.explained_variance_)[None,:]) > Msigma
      fails     = [ self._isFail(input = x[index,:], nRun = nRun, nPc = nPc) for index in combinations(range(x.shape[0]),nNew) ]
      output.append(np.sum(fails)/len(fails)*100)
    return(output)

  def calibrateComp(self):
    '''Calibrate/Set the number of PC components to use when defining the reference ensemble.
    Args:
        None
    Returns:
        Plot of the cumulative explained variance for all principle components
    '''
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    if self.ensemble is None:
      raise Exception('No ensemble present, load/set an ensemble first.')
    sc  = StandardScaler()
    pca = PCA()
    ensSC = sc.fit_transform(self.ensemble)
    pca.fit(ensSC)
    plt.plot(range(1,pca.n_components_+1),np.cumsum(pca.explained_variance_ratio_))
    plt.show(block=False)

  def setComp(self,nComponents=None):
    '''Set the number of PC components to use when defining the reference ensemble.
    Args:
        nComponents : this is the number of components used for PCA (default = Number of PCs
                       needed to explain 90% of the variance.
    Returns:
        None
    '''
    if nComponents is not None:
      self.nComponents = nComponents  
    else:
      from sklearn.preprocessing import StandardScaler
      from sklearn.decomposition import PCA
      import matplotlib.pyplot as plt
      sc  = StandardScaler()
      pca = PCA(n_components=0.9)
      ensSC = sc.fit_transform(self.ensemble)
      pca.fit(ensSC)
      self.nComponents = pca.n_components_
      print('Number of PC-components to use is set to: ' + str(self.nComponents) + ' (explains 90% of the variance)')

  def calibrateFailParams(self,nCPU=None,nRunMax=5,nPcMax=5):
    ''' Calibrate the nNew, nRun and nPc parameters. 
    Args:
        nCPU    : Number of CPUs used for calculations (default = max number of CPUs available)
        nRunMax : Maximum value of number of new runs (nNEW) to investigate. Caution: becomes 
                 calculational very heavy for nRunMax > 5 (default = 5)
        nPcMax  : Maximum value of number of failed PCs to investigate (default = 5)
    Returns:
        A list of dataframes, one element for each value going from 2 to nRunMax, every dataframe
        lists the False positive rate in in function of nNew (rows) and nPc (columns)
    '''
    import multiprocessing as mp
    import functools
    if nCPU is None:
      nCPU = mp.cpu_count()
      print('using maximum CPUs available: ' +str(nCPU) )
    elif nCPU > mp.cpu_count():
      nCPU = mp.cpu_count()
      print('nCPU exceeds maximum avaible number of CPUs, nCPU is set to:' + str(nCPU))
    else:
      print('using ' + str(nCPU) + ' CPUs')
    params = [ (nNew, nRun, nPC) for nNew in range (2,nRunMax+1) for nRun in range(1,nNew+1) for nPC in range(1,nPcMax+1) ]
    f_simpleEET = functools.partial(self._simpleEET,ensemble=self.ensemble,n_components=self.nComponents)
    with mp.Pool(nCPU) as pool:
      result = pool.map(f_simpleEET,params)
    hmaps = []
    k = 0
    for nNew in range(2,nRunMax+1):
      temp = np.zeros(shape=(nNew,nPcMax))
      for i,nRun in enumerate(range(1,nNew+1)):
        for j,nPc in enumerate(range(1,nPcMax+1)):
          temp[i,j] = result[k]
          k += 1
      hmaps.append(pd.DataFrame(temp))
#      plt.pcolor(df)
#      plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
#      plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
#      plt.show()
    return(hmaps)

  def setFailParams(self,nNew,nRun,nPc):
    ''' Set the nNew, nRun adn nPc variables
    Args:
       nNew
       nRun
       nPc
    Returns:
       none
    '''
    self.nNew = nNew
    self.nRun = nRun
    self.nPc  = nPc

  def calibrateEnsSize(self,ensSizes,nCPU=None):
    ''' Calibrate the ensemble size the ensemble size
    Args:
       ensSizes : List of ensemble sizes (e.g. [30, 40, 50])
       nCPU     : Number of CPUs used for calculations (default = max number of CPUs available) 
    Returns:
       Boxplot of the false positive rate for each ensemble size
       Dataframe containing the 10000 false positive rates for each ensemble size
    '''
    import multiprocessing as mp
    import functools
    from matplotlib.pyplot import show
    if self.nNew is None or self.nRun is None or self.nPc is None:
      raise Exception('Failure parameters nNew, nRun, nPc are not set, calibrate and set them first')
    if nCPU is None:
      nCPU = mp.cpu_count()
      print('using maximum CPUs available: ' +str(nCPU) )
    elif nCPU > mp.cpu_count():
      nCPU = mp.cpu_count()
      print('nCPU exceeds maximum avaible number of CPUs, nCPU is set to:' + str(nCPU))
    else:
      print('using ' + str(nCPU) + ' CPUs')
    if max(ensSizes) > self.ensemble.shape[0]-32:
      raise Exception('Given the provided ensemble, the maximum ensemble size cannot exceed ' + str(self.ensemble.shape[0]-32))
    result = []
    for ensSize in ensSizes:
      iREFs = [ random.sample(range(self.ensemble.values.shape[0]),ensSize) for _ in range(100) ]
      print('Performing fullEET with ensemble size: ' + str(ensSize))
      f_fullEET = functools.partial(self._fullEET, ensemble = self.ensemble, n_components = self.nComponents, nNew = self.nNew, nRun = self.nRun, nPc = self.nPc, Msigma = 2)
      print(' - Starting pool..')
      with mp.Pool(nCPU) as pool:
        res = pool.map(f_fullEET, iREFs)
      print(' - Pool finished.')
      flat = [ item for sublist in res for item in sublist ]
      result.append(flat)
    result = pd.DataFrame(result).transpose()
    result.columns = ensSizes
    ax = result.boxplot(showfliers=False)
    ax.axhline(0.5)
    show()
    return(result)
  
  def setEnsSize(self,ensSize):
    ''' Set the ensemble size
     Args:
         ensSize
     Returns:
         none
    '''
    self.ensSize = ensSize
   

def main():
  print('This is the main function')

if __name__ == "__main__":
  main()

