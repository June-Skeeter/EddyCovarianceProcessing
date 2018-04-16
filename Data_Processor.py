import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import datetime as dt
import pytz
from sklearn.utils import resample
from sklearn import metrics
from scipy.optimize import curve_fit


class Compile:
    def __init__(self,Flux_Path,Met,Soil,frequency = '30T'):
        self.Fluxes = ['H','LE','co2_flux','ch4_flux']
        Flux = self.Format(pd.read_csv(Flux_Path,delimiter = ',',skiprows = 0,parse_dates={'datetime':[1,2]},header = 1,na_values = -9999),v=1,drop = [0,1])
        Met = self.Format(pd.read_csv(Met,delimiter = ',',skiprows = 1,parse_dates={'datetime':[0]},header = 0),v=2,drop = [0])
        Soil = self.Format(pd.read_csv(Soil,delimiter = ',',skiprows = 0,parse_dates={'datetime':[0]},header = 0),v=0,drop = [0])

        Soil = Soil.resample(frequency).mean()
        Met = Met.resample(frequency).mean()

        self.RawData = pd.concat([Flux,Met,Soil],axis = 1, join = 'outer')
        for var in self.Fluxes:
            self.RawData[var+'_drop'] = 0
        self.RawData['Minute'] = self.RawData.index.hour*60+self.RawData.index.minute
        self.RawData['Day'] = np.floor(self.RawData['DOY'])
        Mt = pytz.timezone('US/Mountain')
        self.RawData['UTC'] = self.RawData.index.tz_localize(pytz.utc).tz_convert(Mt)
        self.uThresh = .1
        self.Data=self.RawData.copy()

    def Format(self,df,v,drop):
        df = df.ix[v:]
        df = df.set_index(pd.DatetimeIndex(df.datetime))
        df = df.drop(df.columns[drop],axis=1)
        df = df.astype(float)
        return(df)
    
    def Date_Drop(self,Date,Vars):
        if Vars == 'All':
            self.Data = self.Data.drop(self.Data.loc[(self.Data.index>Date[0])&(self.Data.index<Date[1])].index)
        else:
            self.Data.loc[(self.Data.index>Date[0])&(self.Data.index<Date[1]),[Vars]]=np.nan

    def Date_Key(self,Date,key):
        self.Data.loc[(self.Data.index>Date[0])&(self.Data.index<Date[1]),'Date_Key'] = key
        self.Data['Month'] = self.Data.index.month
        
    def Wind_Bins(self,Bins):
        self.bins = np.arange(0,360.1,Bins)
        self.Data['Dir'] = pd.cut(self.Data['wind_dir'],bins=self.bins,labels = (self.bins[0:-1]+self.bins[1:])/2)
        
    def ustar_Bins(self,Bins,LightFilter = {'Var':'PPFD_Avg','Thresh':10},
               uFilter={'Var':'co2_flux','Plot':False},BootStraps={'Repetitions':100,'n_samples':10000}):
        def Rcalc(Grp,thrsh=0.95):
            Ratios=[]
            for G in pd.to_numeric(Grp.index).values:
                m1 = Grp[uFilter['Var']][pd.to_numeric(Grp.index)==G].values[0]
                m2 = Grp[uFilter['Var']][pd.to_numeric(Grp.index)>G].mean()
                Ratios.append(m1/m2)
            Ratios = np.asanyarray(Ratios)
            Ratios[np.where(np.isnan(Ratios)==True)[0]]=1
            try:
                idx = pd.to_numeric(Grp.index).values
                uThresh = idx[np.where(Ratios>=.95)[0]][0]
            except:
                print('Could not find u* thersh, defaulting to 0.1')
                uThresh = 0.1
            return(uThresh)
        self.uFilterData = self.Data[self.Data[LightFilter['Var']]<=LightFilter['Thresh']].copy()
        self.bins = self.uFilterData['u*'].quantile(np.arange(0,Bins,1)/Bins).values
        self.uFilterData['u*bin'] = pd.cut(self.uFilterData['u*'],bins=self.bins,labels = (self.bins[0:-1]+self.bins[1:])/2)

        Grp = self.uFilterData.groupby(['u*bin']).mean()
        GrpC = self.uFilterData.groupby(['u*bin']).size()
        GrpSE = self.uFilterData.groupby(['u*bin'])['fco2'].std()/(GrpC)**.5
        self.uThresh_SampSize = GrpC.sum()
        
        self.uThresh = Rcalc(Grp)
        self.BootStraps = {}
        for i in range(BootStraps['Repetitions']):
            Samp = resample(self.Data,replace=True,n_samples=BootStraps['n_samples'])
            Samp = Samp[Samp[LightFilter['Var']]<=LightFilter['Thresh']]
            bins = Samp['u*'].quantile(np.arange(0,Bins,1)/Bins).values
            Samp['u*bin'] = pd.cut(Samp['u*'],bins=bins,labels = (bins[0:-1]+bins[1:])/2)
            self.BootStraps[str(i)] = Samp
        Ge = []
        for i in self.BootStraps:
            G = self.BootStraps[i].groupby(['u*bin']).mean()
            Ge.append(Rcalc(G))
        Ge = np.asanyarray(Ge)
        self.Pct = {'5%':np.percentile(Ge,[5]),'50%':np.percentile(Ge,[50]),'95%':np.percentile(Ge,[95])}
        self.uThresh = Ge.mean()
        if uFilter['Plot'] == True:
            plt.figure(figsize=(6,5))
            plt.hist(Ge,bins=30,density=True)
            ymin, ymax = plt.ylim()
            def Vlines(var,c,l):
                plt.plot([var,var],[ymin,ymax],
                         color = c,label=l,linewidth=5)
            Vlines(self.uThresh,c='red',l='Mean')
            Vlines(self.Pct['5%'],c='green',l='5%')
            Vlines(self.Pct['50%'],c='yellow',l='50%')
            Vlines(self.Pct['95%'],c='blue',l='95%')
            plt.legend()
            plt.title('u* Thershold & Bootstrapped 95% CI')
            plt.grid()
        
    def PPFD_Bins(self,Bins):
        self.bins = np.arange(0,self.Data['PPFD_Avg'].max()+1,Bins)
        self.Data['Photon_Flux'] = pd.cut(self.Data['PPFD_Avg'],bins=self.bins,labels = (self.bins[0:-1]+self.bins[1:])/2)

    def Rain_Check(self,thresh):
        # self.Data['Rain_diff'] = self.Data['Rain_mm_Tot'].diff()
        for var in self.Fluxes:
            if var!='ch4_flux':
                self.Data.loc[self.Data['Rain_mm_Tot']>thresh[0],[var,var+'_drop']]=[np.nan,1]
            else:
                self.Data.loc[self.Data['Rain_mm_Tot']>thresh[1],[var,var+'_drop']]=[np.nan,1]
        
    def Spike_Removal(self,z_thresh,AltData=None):
        def Remove(series):
            di1 = series.diff()
            di1[:-1] = di1[1:]
            di = di1.diff()
            MD = di.median()
            MAD = np.abs(di-MD).median()
            F1 = di<MD-(z_thresh*MAD/0.6745)
            F2 = di>MD+(z_thresh*MAD/0.6745)
            series.loc[F1==True]=np.nan
            series.loc[F2==True]=np.nan
            Droppers = series.index[np.isnan(series)==True]
            VAR = self.Data[var].copy()
            VAR.ix[Droppers] = np.nan
            dina = VAR.diff()
            dina[:-1] = dina[1:]
            dina2 = VAR.diff()
            NaMid = VAR.index[((np.isnan(dina)==True)&(np.isnan(dina2)==True))]
            VAR.ix[NaMid] = np.nan
            return(VAR)       
        
        if AltData == None:
            for var in self.Fluxes:
                self.Data[var]=Remove(self.Data[var].dropna())
        else:
            AltData[var]=Remove(self.AltData[var].dropna())
            return(AltData[0])
        
    def Wind_Filter(self,width):
        for var in self.Fluxes:
            self.Data.loc[((self.Data['wind_dir']>215-width)&(self.Data['wind_dir']<215+width)),[var,var+'_drop']]=[np.nan,1]
        
    def StorageCorrection(self,Raw=True):
        if Raw == False:
            self.Data['fco2'] = self.Data['co2_flux']+self.Data['co2_strg']
            self.Data['fch4'] = self.Data['ch4_flux']+self.Data['ch4_strg']
        else:
            self.Data['fco2'] = self.Data['co2_flux']+self.Data['co2_strg']
            self.Data['fch4'] = self.Data['ch4_flux']+self.Data['ch4_strg']
        
    def Signal_Check(self,thresh):
        self.Data['ch4_noSSFilter'] = self.Data['ch4_flux']
        self.Data.loc[self.Data['rssi_77_mean']<thresh,['ch4_flux','ch4_flux_drop']] = [np.nan,1]
    
    def QC_Check(self,thresh):
        for var in self.Fluxes:
            self.Data.loc[self.Data['qc_'+var]>=thresh,[var,var+'_drop']]=[np.nan,1]
            self.Data.loc[np.isnan(self.Data[var]) == True,[var+'_drop']]=1
            
    def Ustar_Drop(self,Override=None):
        if Override != None:
            self.uThresh = Override
        for var in self.Fluxes:
            self.Data.loc[self.Data['u*']<self.uThresh,[var,var+'_drop']]=[np.nan,1]
        self.StorageCorrection(Raw=False)
        
    def CustomVars(self):
        self.Data['24H Rain']= self.Data['Rain_mm_Tot'].rolling('24H').sum()
        self.Data['Wtr Tbl Trnd']= self.Data['Table_1'].diff()
        self.Data['Time']= self.Data.index.hour
#         
    def LTR(self,X,alpha,beta,theta,r10,q10):#r1,r2,r3):
            PPFD,temp = X
            return(-1/2*theta*(alpha*PPFD+beta-((alpha*PPFD+beta)**2-4*alpha*beta*theta*PPFD)**.5)+\
                   r10*q10**((temp-10)/10))#(1/(r1*r2**temp+r3)))

    def GPP_ER(self,X,alpha,beta,theta,r10,q10):#r1,r2,r3):
            PPFD,temp = X
            ER = r10*q10**((temp-10)/10)
            GPP = -1/2*theta*(alpha*PPFD+beta-((alpha*PPFD+beta)**2-4*alpha*beta*theta*PPFD)**.5)
            return(ER,GPP)

    def Hyperbola(self,PPFD,alpha,beta):
        return((alpha*beta*PPFD)/(alpha*PPFD+beta))

    def ER(self,Temp,r1,r2,r3):
        return(1/(r2*r2**Temp+r3))
        

    def Fco2_Fill(self,PPFD,Temp,p0 =(0.00699139,  3.08946606,  0.83363605,  0.57199121,  2.01299858)):#,p0 =(0.07456007,30.82786468,0.32274278,0.63617274,1.68500993)):# (0.00716274,1.52597427,1,0.5,0.01)):
        self.Data['NEE'] = np.nan
        self.Data['ER'] = np.nan
        self.Data['GPP'] = np.nan

        Dataset = self.Data[['fco2',PPFD,Temp,'Date_Key','Month']].dropna()
        Filler = self.Data[[PPFD,Temp,'Date_Key','Month']].dropna()
        self.popts = {}
        self.popts2 = {}
        Key = 'Month'
        popt=p0

        for i in Dataset[Key].unique():
            print(i)
            Data = Dataset[Dataset[Key]==i].copy()
            FillData = Filler[Filler[Key]==i].copy()
            popt, pcov = curve_fit(self.LTR, (Data[PPFD].values,Data[Temp].values,),
                           Data['fco2'].values,p0=popt)
            self.popts[str(i)]=popt
            FillData['NEE'] = self.LTR((FillData[PPFD],FillData[Temp]),popt[0],popt[1],popt[2],popt[3],popt[4])
            FillData['ER'],FillData['GPP'] = self.GPP_ER((FillData[PPFD],FillData[Temp]),popt[0],popt[1],popt[2],popt[3],popt[4])
            self.Data.loc[self.Data[Key]==i,'NEE'] = FillData['NEE']
            self.Data.loc[self.Data[Key]==i,'ER'] = FillData['ER']
            self.Data.loc[self.Data[Key]==i,'GPP'] = FillData['GPP']

        self.Data['Fco2'] = self.Data['fco2'].fillna(self.Data['NEE'])

    def Soil_Data_Avg(self,ratios=[.8,.2]):
        self.Data['Ts 2.5cm'] = self.Data['Temp_2_5_1']*ratios[0]+self.Data['Temp_2_5_2']*ratios[1]
        self.Data['Ts 5cm'] = self.Data['Temp_5_1']*ratios[0]+self.Data['Temp_5_2']*ratios[1]
        self.Data['Ts 15cm'] = self.Data['Temp_15_1']*ratios[0]+self.Data['Temp_15_2']*ratios[1]

    def Write(self,Root,Vars,Aliases):
        self.Data[Aliases]=self.Data[Vars]
        self.Data[Aliases].to_csv(Root+'FilteredData' +str(dt.datetime.now()).split(' ')[0]+'.csv')
