#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

## Set your directory
os.chdir('...')

os.getcwd()  # Prints the current working directory



# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams, colors
from matplotlib import gridspec as gspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.path as mpat
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict

color = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen', 'olive', 'gold', 
         'orange', 'peachpuff', '#fb8072', 'red', 'hotpink', '#fccde5','#bebada' ]

# Analysis imports
import numpy.ma as ma
import csv
import netCDF4
from netCDF4 import Dataset
import glob

import pandas as pd
from sklearn import linear_model

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import gls

from numpy import NaN
from patsy import dmatrices

import seaborn as sns


## Import PYREALM and new C3/C4 model
from pyrealm import pmodel
from pyrealm import C3C4model



# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    
  
plt.style.use('default')
plt.rcParams.update({'font.family':'Helvetica'})



## Simple example

get_ipython().run_line_magic('matplotlib', 'inline')

# Load an example dataset contaidong the main variables.
ds = netCDF4.Dataset('pmodel_inputs_C3C4_example.nc')
ds.set_auto_mask(False)

# Extract the six variables for all months
temp = ds['temp'][:]
vpd = ds['vpd'][:]
co2 = ds['co2'][:]    # Note - spatially constant but mapped.
elev = ds['elev'][:]  # Note - temporally constant but repeated
patm = ds['patm'][:]  # Note - temporally constant but repeated
fapar = ds['fapar'][:]
ppfd = ds['ppfd'][:]
theta = ds['theta'][:]
d13co2 = ds['d13co2'][:]  # Note - spatially constant but mapped.
D14co2 = ds['D14co2'][:]   # Note - spatially constant but mapped.
cropland = ds['cropland'][:]  # Note - temporally constant but repeated
treecover = ds['treecover'][:]  # Note - temporally constant but repeated

ds.close()


## Gridcell area 0.5degrees
ds5 = netCDF4.Dataset('gridarea_0.5.nc')
ds5.set_auto_mask(False)
gridarea = ds5['cell_area'][:]
ds5.close()


lat = np.arange(-90, 90, 0.5) # new coordinates
lon = np.arange(-179.75, 180.25, 0.5) # new coordinates


# Data filtering
temp[temp < -25] = np.nan

# Calculate the photosynthetic environment
env = pmodel.PModelEnvironment(tc=temp, co2=co2, patm=patm, vpd=vpd, theta=theta, d13CO2 = d13co2, D14CO2 = D14co2)
#env.summarize()

# Run the P model
modelc3 = pmodel.PModel(env,c4=False,kphio= 1/8)
modelc4 = pmodel.PModel(env,c4=True,kphio= 1/8)
#modelc3.summarize()
#modelc4.summarize()

### Calculation GPP for C3 and C4 plants
modelc3.estimate_productivity(fapar, ppfd)
modelc4.estimate_productivity(fapar, ppfd)


## Calculate annual average/sum
temp_ave = np.nanmean(env.tc,axis = 0)
d13CO2_ave = np.nanmean(env.d13CO2,axis = 0)

D13Cc3_ave = np.nanmean(modelc3.delta.Delta13C,axis = 0)
D13Cc4_ave = np.nanmean(modelc4.delta.Delta13C,axis = 0)

gppc3_ave = np.nansum(modelc3.gpp,axis = 0)
gppc4_ave = np.nansum(modelc4.gpp,axis = 0)

treecover = np.nanmean(treecover,axis=0)
cropland = np.nanmean(cropland,axis=0)


## Calculate fraction of C4 plants
Adv4,F4 = C3C4model.c4fraction(temp_ave,gppc3_ave,gppc4_ave,treecover,cropland)

## Calculation total GPP
gpp_tot,gpp_c3,gpp_c4 = C3C4model.gpp_tot(F4,gppc3_ave,gppc4_ave)

## Conversion from gC m-2 yr-1 to kgC m-2 yr-1
gpp_tot = gpp_tot/1000
gpp_c3 = gpp_c3/1000
gpp_c4 = gpp_c4/1000


## Calculation total D13C
D13C_tot,D13C_c3,D13C_c4 = C3C4model.D13C_tot(F4,D13Cc3_ave,D13Cc4_ave)


## Calculation total d13Cplant
d13C_tot,d13C_c3,d13C_c4 = C3C4model.d13C_tot(F4,d13CO2_ave,D13Cc3_ave,D13Cc4_ave)


## Contribution C4 photosynthesis to GPP (%)
contribuC4 = gpp_c4/gpp_tot*100


## GPP weighted by gridarea and converted from kgC yr-1 to PgC yr-1
gpp_tot_PgC = gpp_tot*gridarea*1e-12
gpp_c3_PgC = gpp_c3*gridarea*1e-12
gpp_c4_PgC = gpp_c4*gridarea*1e-12

## Total carbon budget
np.nansum(gpp_tot_PgC),np.nansum(gpp_c3_PgC),np.nansum(gpp_c4_PgC)  
## 143.1 PgC yr-1, 111.0 PgC yr-1, 32.1 PgC yr-1


# ## Compare simulations with isotopic dataset

# Find grid points when latitude are the closest

def get_data(lat_input, long_input):
    lat_index  = np.nanargmin((np.array(lat)-lat_input)**2)
    long_index = np.nanargmin((np.array(lon)-long_input)**2)
    return lat_index,long_index


## Extraction soil isotopic data from published sources derived from Dong et al. (2022) compilation
data_isotope_dong = pd.read_csv('Glob_Soil_δ13C.csv', index_col=0, na_values=['(NA)'])

data_isotope_dong['Lat'] = data_isotope_dong['Lat'].apply(pd.to_numeric, errors='coerce')
data_isotope_dong['Lon'] = data_isotope_dong['Lon'].apply(pd.to_numeric, errors='coerce')
data_isotope_dong = data_isotope_dong.dropna(subset=["Lat","Lon"])


## Extraction coordinates data for C3 and C4 plants
coord_dong_site = [[0 for i in range(2)] for i in range(len(data_isotope_dong.Lon))]  

for x in range(len(data_isotope_dong.Lat)):
    coord_dong_site[x] = get_data(data_isotope_dong.Lat[x], data_isotope_dong.Lon[x])
coord_dong_site = np.squeeze(pd.DataFrame(coord_dong_site))    


## Extraction d13C for each site from the soil isotopic network
## Predictions
d13C_pred_dong_sites = d13C_tot[coord_dong_site[0],coord_dong_site[1]]

## Observations
d13C_obs_dong_sites = data_isotope_dong.soil_d13C


## Comparison observations-predictions


from sklearn.metrics import mean_squared_error

df_model_reg = pd.DataFrame({'model': d13C_pred_dong_sites, 'soil': d13C_obs_dong_sites}) 
mdf = gls('soil ~ model', data=df_model_reg).fit()

print('model R2: ', mdf.rsquared, 'params', mdf.params, 'pval', mdf.pvalues)


# ## Figures


## Figure: spatial GPP variations for C3 and C4 plants

fig_figure = plt.figure(1, figsize=(20,25))
gs = gspec.GridSpec(3, 3, figure=fig_figure, width_ratios=[0.15, 1, 0.05], hspace=0.2)
# set rows and column
column = 0
row = 0
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)

#%%

# Figure a

column += 1
ax = fig_figure.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(0, 3, 0.05)

diff = plt.contourf(lon, lat, gpp_c3,60, cmap = 'Greens', extend='max', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(a)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure.colorbar(diff, ax, orientation='vertical').set_label(u'GPP$_{C3}$ (kgC m$^{-2}$ yr$^{-1}$)')


# Figure b

column = 1
row +=1
ax = fig_figure.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(0, 3, 0.05)
diff = plt.contourf(lon, lat, gpp_c4,60, cmap = 'Greens', extend='max', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(b)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure.colorbar(diff, ax, orientation='vertical').set_label(u'GPP$_{C4}$ (kgC m$^{-2}$ yr$^{-1}$)')


# Figure c

column = 1
row = 2
ax = fig_figure.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(0, 3, 0.05)
diff = plt.contourf(lon, lat, gpp_tot,60, cmap = 'Greens', extend='max', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(c)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure.colorbar(diff, ax, orientation='vertical').set_label(u'GPP$_{C3+C4}$ (kgC m$^{-2}$ yr$^{-1}$)')

#%%

fig_figure.savefig('Figure_GPP_example.jpg', bbox_inches='tight')

plt.close()




## Figure: fraction of C4 plants (F4) across the globe

fig_figure = plt.figure(1, figsize=(20,25))
gs = gspec.GridSpec(4, 3, figure=fig_figure, width_ratios=[0.15, 0.25, 0.015], hspace=0.3)
# set rows and column
column = 0
row = 0
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)


column += 1
ax = fig_figure.add_subplot(gs[row, column], projection=ccrs.Robinson())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(0, 1, 0.01)

diff = plt.contourf(lon, lat, F4,line, cmap2 = 'YIOrRd', extend='neither', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.15, '(a)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure.colorbar(diff, ax, orientation='vertical').set_label(u'$F_4$')


#%%

fig_figure.savefig('Figure_F4_example.jpg', bbox_inches='tight')

plt.close()



## Figure d13C predicted against d13Csoil

# Set up the subplot figure

fig_figure = plt.figure(1, figsize=(17,18))

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)


gs = gspec.GridSpec(3, 5, figure=fig_figure, hspace=0.4,  width_ratios=[1, 0.1, 1, 0.1, 1])


column = 0
row = 0


ax = fig_figure.add_subplot(gs[row, column])

ax.plot([-35,-10], [-35,-10], color='grey', label='fitted line', ls='dotted')

ax.plot(df_model_reg.model, df_model_reg.soil,'o', color="grey")
sns.regplot(x = "model",y = "soil", 
            data = df_model_reg,order=1,color="black")

ax.set_xlabel(u'$\mathregular{\u03B4^{13}C_{pmodel}}$ (‰)',fontsize=18)
ax.set_ylabel(u'$\mathregular{\u03B4^{13}C_{soil}}$ (‰)',fontsize=18)
ax.set_ylim((-33,-10))
ax.set_xlim((-33,-10))
ax.set_yticks([-30, -25, -20,-15, -10],fontsize=16)
ax.set_xticks([-30,-25, -20, -15, -10],fontsize=16)

ax.tick_params(labelsize=16)

ax.text(0.68, 0.18, u'$R^2$ = O.57',transform=ax.transAxes,va = 'top',fontsize=16)


fig_figure.savefig('Figure_d13C_pred_obs_example.jpg', bbox_inches='tight')

plt.close()





