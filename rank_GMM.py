#Code for ranking GMMs based on the methods described in Lanzano et al. (2019)
#by Elisa Zuccolo, EUCENTRE, Pavia (Italy)
#August 2020

#%% Import libraries
from os import path
import os
import sys
import numpy as np
import pandas as pd
from openquake.hazardlib import gsim, imt, const
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import csv

def evaluate_quantile(periods,GMM,T_database,events,labels,GMMtype,folder):
    Quantile=np.zeros([len(periods),len(GMM)])
    for j in range(len(GMM)):
        for k in range(len(periods)):
            index=np.where(T_database==periods[k])
            index=index[0][0]
            Quantile[k,j]=compute_Quantile_score(GMM[j],events,periods[k],index,GMMtype[j])

    create_figure_score(labels,periods,Quantile,'Quantile',folder)

    Quantile_average=np.zeros([len(GMM)])
    for j in range(len(GMM)):
        Quantile_average[j]=0
        for k in range(len(periods)):
            Quantile_average[j]=Quantile_average[j]+Quantile[k,j]
        Quantile_average[j]=Quantile_average[j]/len(periods)
    temp = sorted(Quantile_average)
    order = [temp.index(i)+1 for i in Quantile_average]

    return Quantile_average.T,order

def evaluate_LLH(periods,GMM,T_database,events,labels,GMMtype,folder):
    LLH=np.zeros([len(periods),len(GMM)])
    for j in range(len(GMM)):
        for k in range(len(periods)):
            index=np.where(T_database==periods[k])
            index=index[0][0]
            LLH[k,j]=compute_LLH_score(GMM[j],events,periods[k],index,GMMtype[j])

    create_figure_score(labels,periods,LLH,'LLH',folder)

    LLH_average=np.zeros([len(GMM)])
    for j in range(len(GMM)):
        LLH_average[j]=0
        for k in range(len(periods)):
            LLH_average[j]=LLH_average[j]+LLH[k,j]
        LLH_average[j]=LLH_average[j]/len(periods)
    temp = sorted(LLH_average,reverse=True)
    order = [temp.index(i)+1 for i in LLH_average]

    return LLH_average.T,order

def evaluate_gambling(periods,GMM,T_database,events,labels,GMMtype,folder):
    Gambling=np.zeros([len(periods),len(GMM)])
    for k in range(len(periods)):
        index=np.where(T_database==periods[k])
        index=index[0][0]
        gambling_all_models=compute_gambling_score(GMM,events,periods[k],index,GMMtype)
        for j in range(len(GMM)):
            Gambling[k,j]=gambling_all_models[j]

    create_figure_score(labels,periods,Gambling,'Gambling',folder)

    Gambling_average=np.zeros([len(GMM)])
    for j in range(len(GMM)):
        Gambling_average[j]=0
        for k in range(len(periods)):
            Gambling_average[j]=Gambling_average[j]+Gambling[k,j]
        Gambling_average[j]=Gambling_average[j]/len(periods)
    temp = sorted(Gambling_average)
    order = [temp.index(i)+1 for i in Gambling_average]

    return Gambling_average.T,order

def create_figure_distribution(all_events,folder):

    nomefigure=folder+'/events_distribution.png'
    plt.figure()
    mag=[]
    dist=[]
    for event in range(len(all_events)):
        mag.append(all_events[event].mw)
        dist.append(all_events[event].rjb)
    plt.semilogx(dist,mag,'o')
    plt.ylabel('$M_w$')
    plt.xlabel('$R_{JB} (km)$')
    plt.savefig(nomefigure)
    plt.close()
    return

def write_selected_database(all_events,folder):
    nomefile=folder+'/selected_events.csv'
    with open(nomefile,mode='w') as csv_file:
        fieldnames = ['mw','repi','rjb','rhypo','width','length','depth','dip','rake','strike','ztor','azimuth','rx','ry0','rrup','vs30','vs30measured','ec8_code']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=' ')
        writer.writeheader()
        for event in range(len(all_events)):
            writer.writerow({'mw': all_events[event].mw,'repi': all_events[event].repi[0],'rjb': all_events[event].rjb[0],'rhypo': all_events[event].rhypo[0],'width': all_events[event].width,'length': all_events[event].length,'depth': all_events[event].depth,'dip': all_events[event].dip,'rake': all_events[event].rake, 'strike': all_events[event].strike, 'ztor': all_events[event].ztor, 'azimuth': all_events[event].azimuth, 'rx': all_events[event].rx[0], 'ry0': all_events[event].ry0[0], 'rrup': all_events[event].rrup[0], 'vs30': all_events[event].vs30, 'vs30measured': all_events[event].vs30measured, 'ec8_code': all_events[event].ec8_code})
    return

def write_results(score_LLH,order_LLH,score_gambling,order_gambling,score_quantile,order_quantile,order_total,model_abbreviations,folder):
    nomefile=folder+'/results.csv'
    with open(nomefile, mode='w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(['method_result']+model_abbreviations)
        writer.writerow(['LLH_score']+list(score_LLH))
        writer.writerow(['LLH_ranking']+list(order_LLH))
        writer.writerow(['gambling_score']+list(score_gambling))
        writer.writerow(['gambling_ranking']+list(order_gambling))
        writer.writerow(['quantile_score']+list(score_quantile))
        writer.writerow(['quantile_ranking']+list(order_quantile))
        writer.writerow(['final_ranking']+list(order_total))
    return

def create_figure_score(models,periods_interest,score,score_name,folder_name):

    nomefigure=folder_name+'/'+score_name+'.png'
    fig=plt.figure()
    ax = fig.add_subplot(111)
    for j in range(len(models)):
        score_single_model=[]
        for k in range(len(periods_interest)):
            if(periods_interest[k]==0):
                periods_interest[k]=0.05
            score_single_model.append(score[k,j])
        ax.semilogx(periods_interest,score_single_model)
    lgd=ax.legend(models,loc='center left', bbox_to_anchor=(1, 0.5),fontsize=8)
    ax.set_xlabel('Period (s)')
    text=score_name+' score'
    ax.set_ylabel(text)
    fig.savefig(nomefigure,bbox_extra_artists=(lgd,), bbox_inches='tight')
    return

class Event:
    def __init__(self,mw,repi,rjb,rhypo,width,length,depth,dip,rake,strike,ztor,azimuth,rx,ry0,rrup,vs30,vs30measured,ec8_code,rotD50,geo):
        self.mw = mw
        self.repi = repi
        self.rjb = rjb
        self.rhypo = rhypo
        self.width = width
        self.length = length
        self.depth = depth
        self.dip = dip
        self.rake = rake
        self.strike = strike
        self.ztor = ztor
        self.azimuth = azimuth
        self.rx = rx
        self.ry0 = ry0
        self.rrup = rrup
        self.vs30 = vs30
        self.vs30measured = vs30measured
        self.ec8_code = ec8_code
        self.rotD50 = rotD50
        self.geo = geo

class Constraint:
    def __init__(self,mw,rjb,depth,fm,ec8,vs30,proximity,sensor_depth,event_nation,station_nation):
        self.mw=mw
        self.rjb=rjb
        self.depth=depth
        self.fm=fm
        self.ec8=ec8
        self.vs30=vs30
        self.proximity=proximity
        self.sensor_depth=sensor_depth
        self.event_nation=event_nation
        self.station_nation=station_nation

def read_dataset(data,T,allowed):
    all_events=[]
    number_contraints=10
    for i in range(len(data)):
        index=np.zeros(number_contraints)
        try:
            ec8class=float(data['ec8_code'][i])
        except ValueError:
            ec8class='is_a_letter'
        if(ec8class=='is_a_letter'):
            ec8_code=str(data['ec8_code'][i])[0]
        mw=float(data['Mw'][i])
        repi=float(data['epi_dist'][i])
        rjb=float(data['JB_dist'][i])
        if(mw<5.5 and np.isnan(rjb)):
            rjb=repi
        rrup=float(data['rup_dist'][i])
        #rx=float(data['Rx_dist'][i])
        #ry0=float(data['Ry0_dist'][i])
        depth=float(data['ev_depth_km'][i])
        proximity=data['proximity_code'][i]
        ev_nation=data['ev_nation_code'][i]
        st_nation=data['st_nation_code'][i]
        sensor_depth=float(data['sensor_depth_m'][i])
        rake=float(data['es_rake'][i])
        width=float(data['es_width'][i])
        length=float(data['es_length'][i])
        dip=float(data['es_dip'][i])
        ztor=float(data['es_z_top'][i])
        strike=float(data['es_strike'][i])
        azimuth=float(data['epi_az'][i])
        vs30=float(data['vs30_m_sec'][i])
        
        if(isinstance(allowed.mw,str) and allowed.mw=='*'):
            index[0]=1
        else:
            if(mw>=allowed.mw[0] and mw<=allowed.mw[1]):
                index[0]=1

        if(isinstance(allowed.rjb,str) and allowed.rjb=='*'):
            index[1]=1
        else:
            if(not np.isnan(rjb) and rjb>=allowed.rjb[0] and mw<=allowed.rjb[1]):
                index[1]=1

        if(isinstance(allowed.depth,str) and allowed.depth=='*'):
            index[2]=1
        else:
            if(depth>=allowed.depth[0] and depth<=allowed.depth[1]):
                index[2]=1

        if(allowed.fm=='*'):
            index[3]=1
        else:
            if(str(data['fm_type_code'][i]) in allowed.fm): 
                index[3]=1

        if(allowed.ec8=='*'):
            index[4]=1
        else:
            if(ec8_code[0] in allowed.ec8):
                index[4]=1

        if(isinstance(allowed.vs30,str) and allowed.vs30=='*'):
            index[5]=1
        else:
            if(vs30>=allowed.vs30[0] and vs30<=allowed.vs30[1]):
                index[5]=1

        if(allowed.proximity=='*'):
            index[6]=1
        else:
            if(proximity in allowed.proximity):
                index[6]=1

        if(isinstance(allowed.sensor_depth,str) and allowed.sensor_depth=='*'):
            index[7]=1
        else:
            if(sensor_depth>=allowed.sensor_depth[0] and sensor_depth<=allowed.sensor_depth[1]):
                index[7]=1

        if(allowed.event_nation=='*'):
            index[8]=1
        else:
            if(ev_nation in allowed.event_nation):
                index[8]=1

        if(allowed.station_nation=='*'):
            index[9]=1
        else:
            if(st_nation in allowed.station_nation):
                index[9]=1

        if(sum(index)==number_contraints):
            if(np.isnan(rake)):
                if(str(data['fm_type_code'][i]=='SS')):
                    rake=0
                if(str(data['fm_type_code'][i]=='TF')):
                    rake=90
                if(str(data['fm_type_code'][i]=='NF')):
                    rake=-90
            if(np.isnan(depth)):
                if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
                    depth=5.63+0.68*mw
                else:
                    depth=11.24-0.2*mw
            rhypo=np.sqrt(repi**2+depth**2)
            if(np.isnan(width)):
                if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
                     # strike slip
                    width= 10.0 ** (-0.76 + 0.27 *mw)
                elif rake > 0:
                    # thrust/reverse
                    width= 10.0 ** (-1.61 + 0.41 *mw)
                else:
                    # normal
                    width= 10.0 ** (-1.14 + 0.35 *mw)
            if(np.isnan(dip)):
                if (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
                    dip=90
                elif rake > 0:
                    dip=40
                else:
                    dip=50
            if(np.isnan(ztor)):
                source_vertical_width=width*np.sin(np.radians(dip))
                ztor=depth-0.6*source_vertical_width
            if(np.isnan(azimuth)):
                azimuth=50

            #if(np.isnan(rx)):
            if(rjb==0):
                rx=0.5*width*np.cos(np.radians(dip))
            else:
                if(dip==90):
                    rx=rjb*np.sin(np.radians(azimuth))
                else:
                    if (azimuth>=0 and azimuth<90) or (azimuth>90 and azimuth<=180):
                        if(rjb*np.abs(np.tan(np.radians(azimuth)))<=width*np.cos(np.radians(dip))):
                            rx=rjb*np.abs(np.tan(np.radians(azimuth)))
                        else:
                            rx=rjb*np.tan(np.radians(azimuth))*np.cos(np.radians(azimuth)-np.arcsin(width*np.cos(np.radians(dip))*np.cos(np.radians(azimuth))/rjb))
                    elif (azimuth==90): #we assume that Rjb>0 
                        rx=rjb+width*np.cos(np.radians(dip))
                    else:
                        rx=rjb*np.sin(np.radians(azimuth))

            #if(np.isnan(ry0)):
            if(azimuth==90 or azimuth==-90):
                ry0=0
            elif(azimuth==0 or azimuth==180 or azimuth==-180):
                ry0=rjb
            else:
                ry0=np.abs(rx*1./np.tan(np.radians(azimuth)))

            if(np.isnan(rrup)):
                if(dip==90):
                    rrup=np.sqrt(np.square(rjb)+np.square(ztor))
                else:
                    if(rx<ztor*np.tan(np.radians(dip))):
                        rrup1=np.sqrt(np.square(rx)+np.square(ztor))
                    if(rx>=ztor*np.tan(np.radians(dip)) and rx<=ztor*np.tan(np.radians(dip))+width*1./np.cos(np.radians(dip))):
                        rrup1=rx*np.sin(np.radians(dip))+ztor*np.cos(np.radians(dip))
                    if(rx>ztor*np.tan(np.radians(dip))+width*1./np.cos(np.radians(dip))):
                        rrup1=np.sqrt(np.square(rx-width*np.cos(np.radians(dip)))+np.square(ztor+width*np.sin(np.radians(dip))))
                    rrup=np.sqrt(np.square(rrup1)+np.square(ry0))

            if(not np.isnan(vs30)):
                vs30measured=False
            else:
                vs30measured=True
                if(ec8_code[0]=='D'):
                    vs30=100
                if(ec8_code[0]=='C'):
                    vs30=250
                if(ec8_code[0]=='B'):
                    vs30=600
                if(ec8_code[0]=='A'):
                    vs30=800

            rotD50=[]
            geo=[]
            for k in range(0,len(T)):

                if(T[k]==0.):
                    U_pga=float(data['U_pga'][i])
                    V_pga=float(data['V_pga'][i])
                    geo.append(np.sqrt(np.abs(U_pga)*np.abs(V_pga)))
                    if(not np.isnan(float(data['rotD50_pga'][i]))):
                        rotD50.append(float(data['rotD50_pga'][i]))
                    else:
                        rotD50.append(np.sqrt(np.abs(U_pga)*np.abs(V_pga)))
                else:
                    stringT='{0:.3f}'.format(T[k])
                    stringT=stringT.replace(".", "_")

                    stringU='U_T'+stringT
                    U=float(data[stringU][i])
                    stringV='V_T'+stringT
                    V=float(data[stringV][i])
                    geo.append(np.sqrt(np.abs(U)*np.abs(V)))

                    string='rotD50_T'+stringT
                    if(not np.isnan(float(data[string][i]))):
                        rotD50.append(float(data[string][i]))
                    else:
                        rotD50.append(np.sqrt(np.abs(U)*np.abs(V)))

            rjb=np.array([rjb])
            repi=np.array([repi])
            rhypo=np.array([rhypo])
            rx=np.array([rx])
            ry0=np.array([ry0])
            rrup=np.array([rrup])
            event = Event(mw,repi,rjb,rhypo,width,length,depth,dip,rake,strike,ztor,azimuth,rx,ry0,rrup,vs30,vs30measured,ec8_code,rotD50,geo)
            all_events.append(event)
    return all_events

def compute_GMM(event,GMPE_input,T_im):
    # Initialise GSIMs
    for name_gmpe, gmpes in gsim.get_available_gsims().items():
        if name_gmpe==GMPE_input:
            bgmpe=gmpes
            break

    if(GMPE_input=='ChiouYoungs2014'):
        z1pt0=np.exp(28.5-3.82/8*np.log(event.vs30**8+378.7**8))
        vs30measured = {
                True : 1,
                False : 0,
                }[event.vs30measured]
    else:
        vs30measured = {
                True : (True,),
                False : (False,),
                }[event.vs30measured]

        if(event.vs30<180):
            z1pt0=np.exp(6.745)
        elif(event.vs30>=180 and event.vs30<=500):
            z1pt0=np.exp(6.745-1.35*np.log(event.vs30/180))
        else:
            z1pt0=np.exp(5.394-4.48*np.log(event.vs30/500))

    
    z2pt5=519.0+3.595*z1pt0

    z1pt0=z1pt0+np.zeros(event.rjb.shape)
    z2pt5=z2pt5+np.zeros(event.rjb.shape)

    sctx = gsim.base.SitesContext()
    rctx = gsim.base.RuptureContext()
    dctx = gsim.base.DistancesContext()

    # Initialise contexts


    setattr(rctx, 'width', event.width)
    setattr(rctx, 'ztor', event.ztor)
    setattr(rctx, 'dip', event.dip)
    setattr(dctx, 'rx', event.rx)
    setattr(dctx, 'rrup', event.rrup)
    setattr(dctx, 'ry0', event.ry0)
    setattr(sctx, 'vs30measured',vs30measured)
    setattr(rctx, 'mag', event.mw)
    setattr(rctx, 'hypo_depth', event.depth)
    setattr(rctx, 'rake', event.rake)
    setattr(dctx, 'rjb', event.rjb)
    setattr(dctx, 'repi', event.repi)
    setattr(dctx, 'rhypo', event.rhypo)
    vs30=event.vs30+np.zeros(event.rjb.shape)
    setattr(sctx, 'vs30',vs30)
    setattr(sctx, 'z1pt0',z1pt0)
    setattr(sctx, 'z2pt5',z2pt5)


    if(T_im==0):
        P = imt.PGA()
    else:
        P = imt.SA(period=T_im)
    S=[const.StdDev.TOTAL]

    im_avg,sigma=bgmpe().get_mean_and_stddevs(sctx,rctx,dctx,P,S)

    return im_avg,sigma

def compute_LLH_score(model,all_events,T,indexT,GMtype):
    N=len(all_events) #number of observations
    sum_LLH=0
    for i in range(N):
        event=all_events[i]
        pred,sigma=compute_GMM(event,model,T)

        if(GMtype=='rotD50'):
            obs=np.log(event.rotD50[indexT]/981)
        if(GMtype=='geo'):
            obs=np.log(event.geo[indexT]/981)
        
        single_LLH=math.log2(norm.pdf(obs,pred,sigma))
        sum_LLH=sum_LLH+single_LLH
    LLH=-sum_LLH/float(N)

    return LLH

def compute_Quantile_score(model,all_events,T,indexT,GMtype):
    R=len(all_events) #number of observations
    sumG=0
    alfa=0.95 
    for i in range(R):
        event=all_events[i]
        pred,sigma=compute_GMM(event,model,T)

        if(GMtype=='rotD50'):
            obs=np.log(event.rotD50[indexT]/981)
        if(GMtype=='geo'):
            obs=np.log(event.geo[indexT]/981)
        
        z_alfa1=np.abs((obs-pred)/sigma) #normalized residuals
        alfa1=(1+alfa)/2
        r1=norm.ppf(alfa1) #quantile function
        if(z_alfa1<=r1):
            g_z=(z_alfa1-r1)*(1-(1-alfa))
        else:
            g_z=(z_alfa1-r1)*(-(1-alfa))
        sumG=sumG+g_z

    G=abs(sumG/R)

    return G

def compute_gambling_score(GMM,all_events,T,indexT,GMtype):
    R=len(all_events) #number of observations
    m=len(GMM) #m number of models
    R_ij=np.zeros([R,m])
    S_ij=np.zeros([R,m])
    G=np.zeros(m)

    p_ij=np.zeros([R,m])
    for j in range(m):
        for i in range(R):
            event=all_events[i]
            model=GMM[j]
            pred,sigma=compute_GMM(event,model,T)

            if(GMtype[j]=='rotD50'):
                obs=np.log(event.rotD50[indexT]/981)
            if(GMtype[j]=='geo'):
                obs=np.log(event.geo[indexT]/981)

            residual=np.abs((obs-pred)/sigma) #normalized residuals
            p_ij[i,j]=2*(1-norm.cdf(residual)) #probability of not being exceeded

    for i in range(R):
        sp_ij=0
        for j in range(m):
            sp_ij=sp_ij+p_ij[i,j]
        for j in range(m):
            R_ij[i,j]=p_ij[i,j]/sp_ij
            S_ij[i,j]=-1+m*R_ij[i,j]
    
    for j in range(m):
        sS_ij=0
        for i in range(R):
            sS_ij=sS_ij+S_ij[i,j]
        G[j]=sS_ij/R

    return G

##################################
#%% Initial setup
try:
    fileini = sys.argv[1]
except IndexError:
    sys.exit('usage: python3 rank_GMPE.py job.ini')

input={}
with open(fileini) as fp:
   line = fp.readline()
   while line:
       if line.strip().find('=')>=0:
            key, value =line.strip().split('=', 1)
            input[key.strip()] = value.strip()
       line = fp.readline()

#%% Extract input parameters

#Database input data
try:
    database_path=input['database_path']
    if not path.exists(database_path):
        sys.exit('Error: The database file does not exist: %s' % database_path)
except KeyError:
    sys.exit('Error: The database path must be defined')

try:
    T_database=[ x.strip() for x in input['periods_database'].strip('[]').split(',') ]
    T_database=np.array(T_database,dtype=float)
except KeyError:
    sys.exit('Error: The periods specified in the database must be defined')

#screening input data

try:
    mw_range=[ x.strip() for x in input['mw_range'].strip('[]').split(',') ]
    mw_range=np.array(mw_range,dtype=float)
except KeyError:
    mw_range='*'

try:
    rjb_range=[ x.strip() for x in input['rjb_range'].strip('[]').split(',') ]
    rjb_range=np.array(rjb_range,dtype=float)
except KeyError:
    rjb_range='*'

try:
    depth_range=[ x.strip() for x in input['depth_range'].strip('[]').split(',') ]
    depth_range=np.array(depth_range,dtype=float)
except KeyError:
    depth_range='*'

try:
    fm_allowed=[ x.strip() for x in input['focal_mechanism'].strip('[]').split(',') ]
except KeyError:
    fm_allowed='*'

try:
    ec8_allowed=[ x.strip() for x in input['ec8'].strip('[]').split(',') ]
except KeyError:
    ec8_allowed='*'

try:
    vs30_allowed=[ x.strip() for x in input['vs30_range'].strip('[]').split(',') ]
    vs30_allowed=np.array(vs30_allowed,dtype=float)
except KeyError:
    vs30_allowed='*'

try:
    proximity_allowed=[ x.strip() for x in input['proximity'].strip('[]').split(',') ]
except KeyError:
    proximity_allowed='*'

try:
    sensor_depth_range=[ x.strip() for x in input['sensor_depth_range'].strip('[]').split(',') ]
    sensor_depth_range=np.array(sensor_depth_range,dtype=float)
except KeyError:
    sensor_depth_range='*'

try:
    event_nation_allowed=[ x.strip() for x in input['event_nation'].strip('[]').split(',') ]
except KeyError:
    event_nation_allowed='*'

try:
    station_nation_allowed=[ x.strip() for x in input['station_nation'].strip('[]').split(',') ]
except KeyError:
    station_nation_allowed='*'

allowed=Constraint(mw_range,rjb_range,depth_range,fm_allowed,ec8_allowed,vs30_allowed,proximity_allowed,sensor_depth_range,event_nation_allowed,station_nation_allowed)

#GMPE input data
try:
    models=[x.strip() for x in input['models'].strip('[]').split(',')]
except KeyError:
    sys.exit('Error: The GMPEs of interest must be defined')
if(len(models)<=1):
    sys.exit('Error: more than one GMPEs must be defined')

model_abbreviations=[]
GMMtype=[]
for i in range(len(models)):
    if models[i]=='AbrahamsonEtAl2014':
        model_abbreviations.append('ASK14')
        GMMtype.append('rotD50')
    if models[i]=='AkkarBommer2010':
        model_abbreviations.append('AB10')
        GMMtype.append('geo')
    if models[i]=='AkkarEtAlRjb2014':
        model_abbreviations.append('ASB14-RJB')
        GMMtype.append('geo')
    if models[i]=='AkkarEtAlRepi2014':
        model_abbreviations.append('ASB14-Repi')
        GMMtype.append('geo')
    if models[i]=='AkkarEtAlRhyp2014':
        model_abbreviations.append('ASB14-Rhyp')
        GMMtype.append('geo')
    if models[i]=='AmeriEtAl2017Rjb':
        model_abbreviations.append('AM17-RJB')
        GMMtype.append('geo')
    if models[i]=='AmeriEtAl2017Repi':
        model_abbreviations.append('AM17-Repi')
        GMMtype.append('geo')
    if models[i]=='BindiEtAl2011':
        model_abbreviations.append('ITA10')
        GMMtype.append('geo')
    if models[i]=='BindiEtAl2014Rjb':
        model_abbreviations.append('BND14-RJB-vs30')
        GMMtype.append('geo')
    if models[i]=='BindiEtAl2014Rhyp':
        model_abbreviations.append('BND14-Rhyp-vs30')
        GMMtype.append('geo')
    if models[i]=='BindiEtAl2014RjbEC8':
        model_abbreviations.append('BND14-RJB-EC8')
        GMMtype.append('geo')
    if models[i]=='BindiEtAl2014RhypEC8':
        model_abbreviations.append('BND14-Rhyp-EC8')
        GMMtype.append('geo')
    if models[i]=='BooreAtkinson2008':
        model_abbreviations.append('BA08')
        GMMtype.append('rotD50')
    if models[i]=='BooreEtAl2014':
        model_abbreviations.append('BSSA14')
        GMMtype.append('rotD50')
    if models[i]=='CampbellBozorgnia2014':
        model_abbreviations.append('CB14')
        GMMtype.append('rotD50')
    if models[i]=='CauzziFaccioli2008':
        model_abbreviations.append('CF08')
        GMMtype.append('geo')
    if models[i]=='CauzziEtAl2014':
        model_abbreviations.append('CZ15-vs30')
        GMMtype.append('geo')
    if models[i]=='CauzziEtAl2014Eurocode8':
        model_abbreviations.append('CZ15-EC8')
        GMMtype.append('geo')
    if models[i]=='ChiouYoungs2014':
        model_abbreviations.append('CY14')
        GMMtype.append('rotD50')
    if models[i]=='DerrasEtAl2014':
        model_abbreviations.append('DBC14')
        GMMtype.append('geo')
    if models[i]=='ChenGalasso20192020':
        model_abbreviations.append('HG20')
        GMMtype.append('rotD50')
    if models[i]=='Idriss2014':
        model_abbreviations.append('ID14')
        GMMtype.append('rotD50')
    if models[i]=='Kuehn_Scherbaum2015':
        model_abbreviations.append('KS15')
        GMMtype.append('geo')
    if models[i]=='LanzanoEtAl2019Rjb':
        model_abbreviations.append('ITA18-RJB')
        GMMtype.append('rotD50')
    if models[i]=='LanzanoEtAl2019Rrup':
        model_abbreviations.append('ITA18-Rrup')
        GMMtype.append('rotD50')
    if models[i]=='ZhaoEtAl2006Asc':
        model_abbreviations.append('ZHA06')
        GMMtype.append('geo')
try:
    periods_interest=[ x.strip() for x in input['periods'].strip('[]').split(',') ]
    periods_interest=np.array(periods_interest,dtype=float)
except KeyError:
    sys.exit('Error: The periods of interest must be defined')

for i in range(len(periods_interest)):
    if periods_interest[i] not in T_database:
        sys.exit('Error: The period %s must be included in the periods of the database' % periods_interest[i])

# output data
try:
    output_folder=input['output_folder']
except KeyError:
    sys.exit('Error: The output folder must be defined')
os.makedirs(output_folder,exist_ok=True)


#End of reading input data

database = pd.read_csv(database_path, sep=';',header='infer',dtype='unicode')
all_events=read_dataset(database,T_database,allowed)

create_figure_distribution(all_events,output_folder)
write_selected_database(all_events,output_folder)

score_LLH,order_LLH=evaluate_LLH(periods_interest,models,T_database,all_events,model_abbreviations,GMMtype,output_folder)
score_gambling,order_gambling=evaluate_gambling(periods_interest,models,T_database,all_events,model_abbreviations,GMMtype,output_folder)
score_quantile,order_quantile=evaluate_quantile(periods_interest,models,T_database,all_events,model_abbreviations,GMMtype,output_folder)

order_total=np.zeros([len(models)],dtype=int)
for j in range(len(models)):
    order_total[j]=order_gambling[j]+order_LLH[j]+order_quantile[j]
    
write_results(score_LLH,order_LLH,score_gambling,order_gambling,score_quantile,order_quantile,order_total,model_abbreviations,output_folder)

