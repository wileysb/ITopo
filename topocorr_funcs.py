#!/usr/bin/python

'''Functions for calculating topographic corrections to surface downwelling irradiance

These functions are mostly implementations of the equations from the following two refernces:
Modelling topographic variation in solar radiation in a GIS environment
L Kumar et al, 1997

An Hourly Diffuse Fraction Model with Correction for Variability and Surface Albedo
Skartveit et al, 1998

I_0 = irradiance incident on a horizontal surface = srb['sw_sfc_dn']
I_0 = I_0direct + I_0diffuse
I_topo = irradiance incident on a surface with slope,aspect as derived from DEM
I_topodirect = I_0direct * cos(i) * 0/1
i = vector(sfc) * vector(sun), dot product between vector from sfc to sun and vector normal to sfc
0/1 = shade or sun, function of solar geometry and surrounding topography

I_topodiffuse = approximated via one of the following:
a) I_0diffuse;
b) I_0diffuse*skyview%; skyview% is % of upward hemisphere not shaded by surrounding topography
c) more advanced calculations requiring BRDF of surrounding surfaces as well as precise cloud geometry


then:
I_topo = I_0 * cos(i) * %direct + I_0 * %skyview * %diffuse
%direct is 0 if cell is shaded by topography


alternative approach: get 'i' via:
import ephem # pyephem, fabulous package for getting angles from an observer on earth's surface to any celestial body at any time
sfc_az  = aspect
sfc_alt = 
i = ephem.separation((sfc_az, sfc_alt), (sun_az, sun_alt))

then cos i = cos(i)

These should yield the same slope correction, but need to derive sfc_alt from slope
http://stackoverflow.com/questions/18685275/angle-between-two-pairs-of-azimuth-and-altitude
* a little lower on the same answer page gives a method for calculating a normal vector from 
altitude and azimuth
Then, cos i = normalvector(sun) * normalvector(sfc)

The Kumar cos_i should be checked against these derivations, however the Kumar calculations are
probably superior to any of them because they implicitly derive solar position from local time,
so the function can be applied to the whole grid at once without separate calculations for
solar geometry at each time step and each pixel location
'''

from scipy.io.netcdf import netcdf_file
from scipy.ndimage.interpolation import zoom
import datetime as dt
import numpy as np
import gdal
import ogr
import osr
import os
import sys

# file paths and such
srb_3hr_fmt = '/space/wib_data/CLIMATE/NASA/srb_3hr/srb_rel3.0_shortwave_3hrly_{0}{1}.nc'#.format(yyyy,mm)
srb_mo_fmt  = '/space/wib_data/CLIMATE/NASA/srb_monthly_utc/srb_rel3.0_shortwave_monthly_utc_{0}{1}.nc'#.format(yyyy,mm)

srb_utm33_dir = '/space/wib_data/CLIMATE/NASA/srb_utm33n/'
dem_dir = '/home/sl_wib/dem_dir/'
shade_dir='/home/sl_wib/shade/'

# Define metadata for raster grids
deleteme = '''
utm33n_md   = { 'outfn':None,
                'from_dset':None, # [y,x] array or gdal dataset
                'epsg':32633,
                'x_size':1210,
                'y_size':1532,
                'dx':1000,
                'dy':-1000,
                'ulx':-84500.00,
                'uly':7961500.0}

wgs84lo =     { 'outfn':'MEM',
                'from_dset':None,
                'epsg':4326,
                'x_size':28, #srb_3hr_vars['diffuse'].shape[2],
                'y_size':15,#srb_3hr_vars['diffuse'].shape[1],
                'dx':1,
                'dy':-1,
                'ulx':4.5,
                'uly':72.5}

wgs84hi =     { 'outfn':'MEM',
                'from_dset':None,
                'epsg':4326,
                'x_size':wgs84lo['x_size']*120, #srb_3hr_vars['diffuse'].shape[2]*120,
                'y_size':wgs84lo['y_size']*120, #srb_3hr_vars['diffuse'].shape[1]*120,
                'dx':0.008333333333333333,
                'dy':-0.008333333333333333,
                'ulx':4.5,
                'uly':72.5}
'''

def Correct_a_month(mm='06', yyyy='2006'):
    '''
    srb_*_fn should be srb netcdfs
    dem_dir should have some kind of grid file (tif, ascii, npy) holding the following grids:
    -dem (elevations, meters)
    -latitude (for each cell, what's the latitude)
    -longitude (for each cell, what's the longitude)
    -slope (degrees!?)
    -aspect (degrees from north??)
    '''

    srb_3hr_fn = srb_3hr_fmt.format(yyyy,mm)
    srb_mo_fn  = srb_mo_fmt.format(yyyy,mm)

    srb_3hr_vars = unpack_srb_variables(srb_3hr_fn)

    toa_ratio = get_toa_ratio(srb_3hr_fn, srb_mo_fn)
    msr = np.max(np.abs(toa_ratio-1)) # most significant ratio
    toa_crit = 0.001 # if toa_ratio is greater than 0.1%

    if msr>toa_crit: # flag the month for the user to consider
        print yyyy,mm,'Consider incorporating toa_ratio:'
        print 'Up to',msr*100,'% significance'

    ydays     = srb_3hr_vars['ydays']
    utc_hours = srb_3hr_vars['utc_hours']

    # resample srb_3hr_sfc_dn and diffuse to ~30arcsec or 1arcmin grids

    # reproject sw_sfc_dn and diffuse to utm33, 1km grids
    # http://jgomezdans.github.io/gdal_notes/reprojection.html
    out_dir = srb_utm33_dir
    itopo_fmt = os.path.join(out_dir, 'itopo_{1}_{2}_{3}.tif')
    srb_to_utm_smoother(srb_3hr_vars, out_dir)

    # load slope, aspect, lat, lon, skyview
    slope_fn = os.path.join(dem_dir, 'gtopo30slp_utm33n.asc')
    aspect_fn = os.path.join(dem_dir, 'gtopo30asp_utm33n.asc')
    lat_fn = os.path.join(dem_dir, 'gtopo30lat_utm33n.asc')
    lon_fn = os.path.join(dem_dir, 'gtopo30lon_utm33n.asc')
    sky_fn = os.path.join(dem_dir, 'gtopo30sky_utm33n.asc')

    slope  = gdal_load(slope_fn)  # np.loadtxt(slope_fn,skiprows=6,delimiter=' ')
    aspect = gdal_load(aspect_fn) # np.loadtxt(aspect_fn,skiprows=6,delimiter=' ')
    lat    = gdal_load(lat_fn)    # np.loadtxt(lat_fn,skiprows=6,delimiter=' ')
    lon    = gdal_load(lon_fn)    # np.loadtxt(lon_fn,skiprows=6,delimiter=' ')
    skyview= gdal_load(sky_fn)    #np.loadtxt(sky_fn,skiprows=6,delimiter=' ')

    for i in range(len(ydays)):
        yday = ydays[i]
        utc_hour = utc_hours[i]

        shade = collect_shade(lat, lon, yday, utc_hour)
        sw_sfc_dn_utm = ld_srb_utm33('sw_sfc_dn',yyyy,yday,utc_hour)
        diffuse_utm = ld_srb_utm33('diffuse',yyyy,yday,utc_hour)
        diffuse_utm[diffuse_utm>1]=1

        # srb_mo_sfc_dn = mean(srb_3hr_sfc_dn) * toa_ratio # would then have to resample and reproject toa_ratio with sw_sfc_dn and diffuse
        utm33n_md['outfn'] = itopo_fmt.format(yyyy, yday, utc_hour)

        topo_params = {'sw_sfc_dn':sw_sfc_dn_utm, 'p_diffuse':diffuse_utm, 'shade':shade,
                       'slope':slope, 'aspect':aspect, 'skyview':skyview, 'lat':lat, 'lon':lon,
                       'yday':yday, 'utc_hour':utc_hour}
        utm33n_md['from_dset'] = Apply_topo_corr(sw_sfc_dn_utm, diffuse_utm, shade, slope, aspect, skyview, lat, lon, yday, utc_hour)
        gdal_save_grid(**utm33n_md)
        # save to file?


def get_toa_ratio(srb_3hr_fn, srb_mo_fn):

    scaling_factor_3hrly = 0.1

    srb_3 = netcdf_file(srb_3hr_fn,'r',mmap=False)
    srb_mo = netcdf_file(srb_mo_fn,'r',mmap=False)

    # Norway boundaries
    lon_s = 4
    lon_e = 32
    lat_s = 90+57
    lat_e = 90+72

    sw3_toa_dn = srb_3.variables['sw_toa_dn'][:,lat_s:lat_e,lon_s:lon_e]
    swMO_toa_dn = srb_mo.variables['sw_toa_dn'][0,lat_s:lat_e,lon_s:lon_e]

    moRAW = scaling_factor_3hrly * np.mean(sw3_toa_dn,axis=0)

    ratio = swMO_toa_dn / moRAW

    return ratio


def Apply_topo_corr(sw_sfc_dn, p_diffuse, shade, slope, aspect, skyview, lat, lon, yday, utc_hr):
    '''derive cos_i and apply along with topographic shading/skyview and diffuse/direct ratios

    cos(i) is the dot product of the normal vectors representing solar and terrain angles

    :param sw_sfc_dn: array with dimensions [time, y, x]
    :param p_diffuse: array with dimensions [time, y, x]
    :param shade:     array with dimensions [time, y, x]
    :param slope:     array with dimensions [y,x]
    :param aspect:    array with dimensions [y,x]
    :param skyview:   array with dimensions [y,x]
    :param lat:       array with dimensions [y,x]
    :param lon:       array with dimensions [y,x]
    :param yday:      array with dimensions [time]
    :param utc_hr:    array with dimensions [time]
    :return: sw_sfc_dn_topo, array with dimensions [time, y, x]
    '''

    subzero = sw_sfc_dn < 0
    sw_sfc_dn[subzero] = 0

    subzero = p_diffuse < 0
    p_diffuse[subzero] = 0
    
    irrat   = p_diffuse > 1
    p_diffuse[irrat] = 1

    local_hr = Get_utc_offset(lon) + utc_hr # map/grid
    h_s = Get_solar_hour_angle(local_hr) # value
    delta_s = Get_solar_declination(yday) # value

    cos_i_params     = {'delta_s':delta_s, 'h_s':h_s, 'lat':lat, 'slope':slope, 'aspect':aspect}
    cos_i_corr = Get_corr(**cos_i_params)

    sza_fl = (Get_solar_elevation(lat, yday, local_hr, return_sza=True))
    sza_int = np.round(sza_fl,0)

    sunset = sza_int < 89

    shade[sunset] = 0

    p_direct = 1-p_diffuse
    topodir = sw_sfc_dn * p_direct * cos_i_corr * shade
    # (topodir[(np.isnan(topodir)!=(np.isnan(slope) | np.isnan(aspect)))]==0).all()
    # topodir is nan everywhere slope or aspect are nan, unless topodir has been set to 0
    # Are all these slope and aspect nans lakes?

    topodiff = sw_sfc_dn * p_diffuse * skyview

    itopo = topodir + topodiff

    # make sure nans don't screw up averaging!
    zer0s = sw_sfc_dn==0
    itopo[zer0s] = 0
    # any other nanjustments indicated?

    return itopo


def Get_cos_i(delta_s, h_s, lat, slope, aspect):
    '''Reference: Kumar et al, 1997, 'Solar radiation modelling'

    The function is an implementation of the following equation from this reference:
    cos i = sin(delta_s) * (sin(L)cos(beta) - cos(L)sin(beta)cos(a_w)) + 
            cos(delta_s)cos(h_s) * (cos(L)cos(beta) + sin(L)sin(beta)cos(a_w)) + 
            cos(delta_s)sin(beta)sin(a_w)sin(h_s)
            
            a_w = aspect angle
            beta = slope angle
            L = latitude
            delta_s = solar declination
            h_s = sun hour angle
    '''
    # for some reason this seems to take aspect 0 as south facing?
    aspect = (aspect-180)%360

    # first prepare a bunch of short, familiar names for readability
    from numpy import cos
    from numpy import sin
    rlat = np.deg2rad(lat)
    rslope = np.deg2rad(slope)
    raspect = np.deg2rad(aspect)
    rdelta_s = np.deg2rad(delta_s)
    rh_s = np.deg2rad(h_s)
    
    # then define the function
    term1 = sin(rdelta_s) * \
                           (sin(rlat) * cos(rslope) - \
                            cos(rlat) * sin(rslope) * cos(raspect))
            
    term2 = cos(rdelta_s) * cos(rh_s) * \
                                      (cos(rlat) * cos(rslope) + \
                                       sin(rlat) * sin(rslope) * cos(raspect))
    
    term3 = cos(rdelta_s) * sin(rslope) * sin(raspect) * sin(rh_s)
            
    cos_i = term1 + term2 + term3
    
    return cos_i


def Get_corr(delta_s, h_s, lat, slope, aspect):
    cos_i_horizontal_params     = {'delta_s':delta_s, 'h_s':h_s, 'lat':lat, 'slope':0, 'aspect':0}
    cos_i_horizontal = Get_cos_i(**cos_i_horizontal_params)

     # if slope or aspect are nan, it's a body of water, so they are basically just horizontal terrain
    # for which slope is 0, aspect is meaningless
    aspect[np.isnan(aspect)] = 0
    slope[np.isnan(slope)] = 0

    cos_i_topo_params     = {'delta_s':delta_s, 'h_s':h_s, 'lat':lat, 'slope':slope, 'aspect':aspect}
    cos_i_topo       = Get_cos_i(**cos_i_topo_params)
    # Alternate formulation, from Grenfell et al 1994:
    # cos_i_topo = np.cos(np.deg2rad(sza-slope*np.cos(np.deg2rad(aspect-solar_azimuth))))

    cos_i_corr = cos_i_topo / cos_i_horizontal # should be >1 when south facing slope, <1 when northfacing. Negative means 0 irrad

    # Set corrected direct irradiance to zero when:
    # * cos_i_horizontal is too close to 0 (result grows towards infinity),
    # * either cos_i is 0 or negative
    # ...(but cos_i_horizontal shouldn't have nans)
    small_thresh = 0.01
    # mask = np.isnan(cos_i_topo)
    # mask = mask | np.isnan(cos_i_horizontal)
    mask = (cos_i_horizontal <= (0 + small_thresh)) # | mask
    mask = (cos_i_topo <= 0) | mask

    cos_i_corr[mask] = 0 # the direct component in any of these situations should be 0

    return cos_i_corr


# Most of these equations are short and simple.
def Get_solar_declination(yday):
    '''delta_s = 23.45 sin (360d (284+N) / 365)'''
    delta_s = 23.45 * np.sin(np.deg2rad(360 * (284+yday) / 365))
    return delta_s


def Get_solar_hour_angle(hr):
    '''h_s = Get_solar_hour(hr)
    hr = time of day, decimal hours (1pm = 13.00)
    
    h_s = 15*t (degrees)
    t = hours after local noon 
    '''
    t = hr-12
    h_s = 15*t # 15 == 360 degrees / 24 hours == sun's rate of rotation around azimuth
    return h_s


def Get_solar_azi(h_s,lat,sza,decl):
    pm = (h_s > 0)
    am = (h_s <= 0)

    azi = np.zeros(h_s.shape, dtype='float')

    decl = np.deg2rad(decl)
    lat  = np.deg2rad(lat)
    s_el = np.deg2rad(90-sza)
    h_s  = np.deg2rad(h_s)

    term1 = np.sin(decl)*np.cos(lat)
    term2 = np.cos(h_s)*np.cos(decl)*np.sin(lat)
    term3 = np.cos(s_el)

    azi_term = np.rad2deg(np.arccos((term1-term2)/term3))

    # Fix am/pm orientation
    azi[pm] = 360 - azi_term[pm]
    azi[am] = azi_term[am]

    # fix rare nan errors at h_s==0 (noon) or 180 (midnight)
    if np.isnan(azi).any():
        noonish = np.isnan(azi) & (np.abs(h_s)<0.02)
        midnightish = np.isfinite(azi) & (abs(np.round(abs(h_s),8)-np.pi)<0.02)

        azi[noonish] = 180. # due south
        azi[midnightish] = 0 # due north

    return azi


def Get_utc_offset(lon):
    '''UTC_offset = lon * 24 / 360'''
    utc_offset = lon*24/360.
    return utc_offset
    

def Get_solar_elevation(lat, yday, hour, return_sza=False):
    '''from cell AE2, http://www.esrl.noaa.gov/gmd/grad/solcalc/NOAA_Solar_Calculations_day.ods
    cos(sza) = sin(lat)*sin(decl) + cos(lat)*cos(decl)* cos(h_s)
    '''
    b3 = np.deg2rad(lat)
    t2 = np.deg2rad(Get_solar_declination(yday))
    ac2 = np.deg2rad(Get_solar_hour_angle(hour))
    solar_zenith = np.rad2deg(np.arccos(np.sin(b3)*np.sin(t2) + np.cos(b3)*np.cos(t2)*np.cos(ac2)))
    if return_sza:
        return solar_zenith
    else:
        solar_elevation = 90-solar_zenith
        return solar_elevation


# functions for diffuse fraction
# from Skartveit, Olseth and Tuft, 1998
def var_k(sw_sfc_dn, sw_toa_dn):
    k = sw_sfc_dn.astype('float') / sw_toa_dn
    return k


def var_k_1(h):
    k_1 = 0.83 - 0.56*np.e**(-0.06*h)
    return k_1


def var_k_max(k_b_max, d_2, k_2):
    k_max = (k_b_max + d_2*k_2/(1-k_2)) / (1 + d_2*k_2/(1-k_2))
    return k_max


def var_d_1(h):
    d_1 = np.copy(h) # type?
    d_1[:] = 1
    d_1[h>1.4] = 0.07 + 0.046 * (90-h[h>1.4])/(h[h>1.4]+3)
    #if h>1.4:
    #    d_1 = 0.07 + 0.046 * (90-h)/(h+3)
    #else:
    #    d_1 = 1
    return d_1


def var_d_max(d_2, k_2, k_max):
    d_max = d_2 * k_2 * (1-k_max) / (k_max * (1-k_2))
    return d_max


def var_K(k, k_1):
    K = 0.5*(1+np.sin(np.deg2rad(np.pi*(k - 0.22) / (k_1 - 0.22) - np.pi/2)))
    return K


def eq5(k,h):
    k_1 = var_k_1(h) # 6b
    d_1 = var_d_1(h) # 6d
    K = var_K(k, k_1) # 6a
    d = 1 - (1-d_1) * (0.11*np.sqrt(K) + 0.15*K + 0.74 * K**2) # eq 5
    return d


def eq7(d_2, k_2, k):
    d = d_2 * k_2 * (1-k) / (k*(1-k_2))
    return d


def eq10(k_max, d_max, k):
    d = 1-k_max*(1-d_max)/k
    return d


# The diffuse fraction model and the helper function for deriving diffuse
# fraction from a given srb dataset
def Get_diffuse_fraction(sw_sfc_dn, sw_toa_dn, utc_hour, yday,lat, lon):
    '''Given the downwelling shortwave irradiance at the top of the atmosphere
    and at the surface, as well as the times and positions corresponding to the array cells,
    return the fraction of irradiance made up by diffuse light.

    Direct fraction = 1 - diffuse fraction

    Equations sourced from:
    'An Hourly Diffuse Fraction Model with Correction for Variability and Surface Albedo'
    Arvid Skartveit, Jan Asle Olseth, Marit Elisabet Tuft
    Solar Energy vol 63, no 3, pp 173-183
    1998

    Note: Due to the broad geographical (1 degree) and temporal (3 hour) averaging of the
    NASA srb dataset, neither of the corrections detailed in the above paper are implemented.

    Surface Albedo could be useful, but the equations are defined incompletely in the paper referenced.

    Variability (correcting for heterogeneous cloud conditions) was implemented, but introduced
    extremely faulty values for diffuse fraction, so was removed from the model.

    Each of the following should be 1, 2 or 3 dimensional arrays with the same dimensions:
    :param sw_sfc_dn:
    :param sw_toa_dn:
    :param utc_hour:
    :param yday:
    :param lat:
    :param lon:
    :return: diffuse fraction (0-1)
    '''

    sw_sfc_dn = np.where(sw_sfc_dn==0,np.nan,sw_sfc_dn)
    sw_toa_dn = np.where(sw_toa_dn==0,np.nan,sw_toa_dn)


    hour = Get_utc_offset(lon) + utc_hour # 'local time in hours' #

    h = Get_solar_elevation(lat, yday, hour) # solar elevation, degrees
    # Establish out-of-bounds values
    # h = np.where(h<0,np.nan,h)

    k = sw_sfc_dn.astype('float') / sw_toa_dn

    k_1 = var_k_1(h)
    k_2 = 0.95*k_1


    # k_b_max = 0.81**((1/np.sin(np.deg2rad(h)))**0.6)
    # k_b_max==incomputable as float64 where h<0:
    k_b_max = 0.81**((1/np.sin(np.deg2rad(np.where(h<0,np.nan,h))))**0.6)
    d_2 = eq5(k_2, h)
    k_max = var_k_max(k_b_max, d_2, k_2)
    d_max = var_d_max(d_2, k_2, k_max)


    # start collecting diffuse fraction over various conditions
    d = np.copy(k) # type?
    # (1)
    print 'start diffuse clearness conditionals ==>'

    con1 =  (k <= 0.22) # 'invalid value encountered in less equal' because of nans in k
    d[con1] = 1.00

    # (2)
    con2 = (0.22 <= k) & (k <= k_2)
    d[con2] = eq5(k,h)[con2]

    # (3)
    con3 =(k_2 <= k) & (k <= k_max)
    d[con3] = eq7(d_2, k_2, k)[con3]

    # (4)
    con4 = k >= k_max
    d[con4] = eq10(k_max, d_max, k)[con4]


    d[d>1] = 1
    d[h<0] = 1 # no direct component when sun is below horizon

    print '==> end diffuse clearness conditionals'
    return d


def unpack_srb_variables(srb_fn):

    srb = netcdf_file(srb_fn,'r',mmap=False)

    # Norway boundaries
    lon_s = 4 # todo these have to be dynamic, not fixed to whole norway
    lon_e = 32
    lat_s = 90+57
    lat_e = 90+72

    sw_sfc_dn = srb.variables['sw_sfc_dn'][:][:,lat_s:lat_e,lon_s:lon_e] * 0.1
    sw_toa_dn = srb.variables['sw_toa_dn'][:][:,lat_s:lat_e,lon_s:lon_e] * 0.1
    lat = srb.variables['lat'][:][lat_s:lat_e]
    lon = srb.variables['lon'][:][lon_s:lon_e]
    sza = srb.variables['sza'][:][:,lat_s:lat_e,lon_s:lon_e] # I don't think I use this anywhere
    itime = srb.variables['time'][:]

    srb.close()

    t_0 = dt.datetime(1960,1,1)

    itimes = [t_0 + dt.timedelta(hours=itime[i]) for i in range(len(itime))]
    utc_hours = np.array([srbtime.hour for srbtime in itimes])
    ydays     = [srbtime.timetuple().tm_yday for srbtime in itimes]

    year = itimes[0].year

    # make output grid, float values for same shape as sw_sfc_dn
    # make lat and lon into grids
    lonv, latv = np.meshgrid(lon, lat)

    hr3  = np.ones(sw_sfc_dn.shape,dtype=np.float64)*10
    lon3 = np.ones(sw_sfc_dn.shape,dtype=np.float64)*10
    lat3 = np.ones(sw_sfc_dn.shape,dtype=np.float64)*10
    yday3 = np.ones(sw_sfc_dn.shape,dtype=np.float64)*10

    for i in range(len(utc_hours)):
        lon3[i,:,:] = lonv
        lat3[i,:,:] = latv
        hr3[i,:,:]  = utc_hours[i]
        yday3[i,:,:]  = ydays[i]


    #sw_sfc_dn = np.where(sw_sfc_dn==0,np.nan,sw_sfc_dn)
    #sw_toa_dn = np.where(sw_toa_dn==0,np.nan,sw_toa_dn)
    diffuse_params = {'sw_sfc_dn':sw_sfc_dn, 'sw_toa_dn':sw_toa_dn, 'utc_hour':hr3,
                      'yday':yday3,'lat':lat3,'lon':lon3}

    d =  Get_diffuse_fraction(**diffuse_params)

    srb_vars = {'sw_sfc_dn':sw_sfc_dn, 'sw_toa_dn':sw_toa_dn,
                'diffuse':d,
                'lat':lat, 'lon':lon, 'sza':sza, 'year':year,
                'utc_hours':utc_hours, 'ydays':ydays}

    return srb_vars


def collect_shade(lat, lon, yday, utc_hour):
    shade_map = np.ones(lat.shape,dtype='int')

    shade_fmt = os.path.join(shade_dir,'solaraz{0}solarzen{1}.asc')

    hour = Get_utc_offset(lon) + utc_hour # 'local time in hours' #
    sza_fl = (Get_solar_elevation(lat, yday, hour, return_sza=True))
    decl = Get_solar_declination(yday)
    h_s = Get_solar_hour_angle(hour)
    s_az_fl = Get_solar_azi(h_s, lat, sza_fl, decl)

    # make 'em integers
    sza = np.round(sza_fl,0)
    s_az = np.round(s_az_fl,0)

    az = np.unique(s_az)
    zen= np.unique(sza)

    for solar_zen in zen:
        if solar_zen<89:
            for solar_az in az:
                mask = ((sza==solar_zen)&(s_az==solar_az))
                if mask.any():
                    shade = gdal_load(shade_fmt.format(int(solar_az),int(solar_zen))).astype('int') #,skiprows=6,dtype='bool',delimiter=' ')
                    shade_map[mask]=shade[mask]

    return shade_map


# Some general purpose or notemaking geospatial functions:
def Prj_mkdir(dir_path):
    '''Create a directory, with a bit of extra syntax.

    Check if directory exists, print error and exit if directory cannot be created.
    This error probably indicates a bad path, pointing to a directory structure
    which doesn't exist yet.

    :param dir_path: path to directory to create
    :return: Exit(1) on failure, None on success
    '''
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except:
            print '[ERROR] Problem creating ',dir_path
            sys.exit(1)


def Get_gt_dict(ras_fn):
    ds = gdal.Open(ras_fn,gdal.GA_ReadOnly)
    srs_wkt = ds.GetProjectionRef()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(srs_wkt)
    srs.AutoIdentifyEPSG() # this can, unlikely, fail.
    srs_epsg = srs.GetAuthorityCode(None)
    gt = ds.GetGeoTransform()

    gt_dict = {'epsg':srs_epsg,
                'x_size': ds.RasterXSize,
                'y_size': ds.RasterYSize,
                'dx':gt[1],
                'dy':gt[5],
                'ulx':gt[0],  # 4.5,
                'uly': gt[3]}#  72.5}
    return gt_dict


def srb_to_utm(srb_3hr_vars, out_dir):

    yyyy = srb_3hr_vars['year']

    out_fmt = os.path.join(out_dir, '{0}_{1}_{2}_{3}.tif')#.format(dset, year, yday, utc_hour)
    wgs84lo = { 'outfn':'MEM',
                'from_dset':None,
                'epsg':4326,
                'x_size':srb_3hr_vars['diffuse'].shape[2],
                'y_size':srb_3hr_vars['diffuse'].shape[1],
                'dx':1,
                'dy':-1,
                'ulx':4.5,
                'uly':72.5}

    wgs84hi = { 'outfn':'MEM',
                'from_dset':None,
                'epsg':4326,
                'x_size':srb_3hr_vars['diffuse'].shape[2]*120,
                'y_size':srb_3hr_vars['diffuse'].shape[1]*120,
                'dx':0.008333333333333333,
                'dy':-0.008333333333333333,
                'ulx':4.5,
                'uly':72.5}

    for i in range(len(srb_3hr_vars['utc_hours'])):
        yday = srb_3hr_vars['ydays'][i]
        utc_hour = srb_3hr_vars['utc_hours'][i]

        # save sw_sfc_dn to mem
        wgs84lo['from_dset'] = np.flipud(srb_3hr_vars['sw_sfc_dn'][i,:,:])
        wgs84lo['nanhandle'] = -999
        wgs84hi['from_dset'] = gdal_save_grid(**wgs84lo)

        # resample sw_sfc_dn to high res geographic
        utm33n_md['from_dset'] = gdal_resample(**wgs84hi)

        # reproject sw_sfc_dn to utm33n, 1km
        dset = 'sw_sfc_dn'
        utm33n_md['outfn'] = out_fmt.format(dset, yyyy, yday, utc_hour)
        utm_dset = gdal_resample(**utm33n_md)

        # save diffuse to mem
        # wgs84lo['from_dset'] = np.where(np.isfinite(srb_3hr_vars['diffuse'][i,:,:]), srb_3hr_vars['diffuse'][i,:,:], -999)
        wgs84lo['from_dset'] = np.flipud(np.where(np.isfinite(srb_3hr_vars['diffuse'][i,:,:]), srb_3hr_vars['diffuse'][i,:,:], -999))
        wgs84hi['from_dset'] = gdal_save_grid(**wgs84lo)

        # resample sw_sfc_dn to high res geographic
        utm33n_md['from_dset'] = gdal_resample(**wgs84hi)

        # reproject sw_sfc_dn to utm33n, 1km
        dset = 'diffuse'
        utm33n_md['outfn'] = out_fmt.format(dset, yyyy, yday, utc_hour)
        utm_dset = gdal_resample(**utm33n_md)


def srb_to_utm_smoother(srb_3hr_vars, out_dir):
    yyyy = srb_3hr_vars['year']

    out_fmt = os.path.join(out_dir, '{0}_{1}_{2}_{3}.tif')#.format(dset, year, yday, utc_hour)

    for i in range(len(srb_3hr_vars['utc_hours'])):
        yday = srb_3hr_vars['ydays'][i]
        utc_hour = srb_3hr_vars['utc_hours'][i]


        dset = 'sw_sfc_dn'
        wgs84lo['from_dset'] = np.flipud(srb_3hr_vars[dset][i,:,:])

        # interpolate from degree resolution to 30arc seconds (0.5arc minutes = 120 per degree)
        wgs84hi['from_dset'] = zoom(wgs84lo['from_dset'], zoom=120)

        # save to gdal memory
        wgs84hi['nanhandle'] = -999
        utm33n_md['from_dset'] = gdal_save_grid(**wgs84hi)

        # reproject sw_sfc_dn to utm33n, 1km
        utm33n_md['outfn'] = out_fmt.format(dset, yyyy, yday, utc_hour)
        utm_dset = gdal_resample(**utm33n_md)

        dset = 'diffuse'
        wgs84lo['from_dset'] = np.flipud(srb_3hr_vars[dset][i,:,:])

        # interpolate from degree resolution to 30arc seconds (0.5arc minutes = 120 per degree)
        wgs84hi['from_dset'] = zoom(wgs84lo['from_dset'], zoom=120)

        # save to gdal memory
        utm33n_md['from_dset'] = gdal_save_grid(**wgs84hi)

        # reproject sw_sfc_dn to utm33n, 1km
        utm33n_md['outfn'] = out_fmt.format(dset, yyyy, yday, utc_hour)
        utm_dset = gdal_resample(**utm33n_md)


def ld_srb_utm33(dset,yyyy,yday,utc_hour, tif_dir=srb_utm33_dir):
    tif_fmt = os.path.join(tif_dir, '{0}_{1}_{2}_{3}.tif')
    tif_fn = tif_fmt.format(dset, yyyy, yday, utc_hour)
    arr = gdal_load(tif_fn)
    return arr


def gdal_load(dset_fn):
    dset = gdal.Open(dset_fn)
    band = dset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr= band.ReadAsArray()

    arr = np.where(arr==nodata, np.nan, arr)
    return arr


def gdal_resample(outfn, from_dset, epsg, x_size, y_size, ulx, uly, dx, dy, nanhandle=False, rot0=0, rot1=0):

    # Get from_projection
    src_prj_string = from_dset.GetProjection()

    # Bring projection info from EPSG to WKT string
    dst_prj = osr.SpatialReference()
    dst_prj.ImportFromEPSG(epsg)
    dst_prj_string = dst_prj.ExportToWkt()

    # Create memory dataset
    if outfn=='MEM':
        mem_drv = gdal.GetDriverByName('MEM')
        dst = mem_drv.Create('',x_size, y_size, 1, gdal.GDT_Float64)

    else:
        gtiff_drv = gdal.GetDriverByName('GTiff')
        dst = gtiff_drv.Create(outfn,x_size, y_size, 1, gdal.GDT_Float64)

    # Assemble geotransform
    geo_t = ( ulx, dx, rot0, \
              uly, rot1, dy )

    # Write geotransform, projection, and array to memory dataset
    dst.SetGeoTransform( geo_t )
    dst.SetProjection( dst_prj_string )

    # can result in negatives? nans?
    res = gdal.ReprojectImage(from_dset, dst, \
          src_prj_string, dst_prj_string, \
          gdal.GRA_Cubic) #_Bilinear )

    return dst


def numpy_resample(src_arr, zm=120):

    out_array = zoom(src_arr,zoom=zm)

    '''
    # get x and y axes, input grid
    xmin = src_md['ulx']
    dx   = src_md['dx']
    xmax = xmin+dx*src_md['x_size']
    lon = np.linspace(xmin,xmax,num=src_md['x_size'],endpoint=False)

    ymax = src_md['uly']
    dy   = src_md['dy']
    ymin = ymax+dy*src_md['y_size']
    lat = np.linspace(ymin,ymax,num=src_md['y_size'],endpoint=False)

    # Create cubic spline
    f = RectBivariateSpline(lat,lon,arr)

    # make xout and yout grids
    oxmin = dst_md['ulx']
    odx   = dst_md['dx']
    oxmax = oxmin+odx*dst_md['x_size']
    ox = np.linspace(oxmin,oxmax,num=dst_md['x_size'],endpoint=False)

    oymax = dst_md['uly']
    ody   = dst_md['dy']
    oymin = oymax+ody*dst_md['y_size']
    oy = np.linspace(oymin,oymax,num=dst_md['y_size'],endpoint=False)

    xout, yout = np.meshgrid(ox,oy)
    # resample array
    out_array = f(xout,yout)

    http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.RectBivariateSpline.html#scipy.interpolate.RectBivariateSpline
    http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.interpolate.interp2d.html

    This is what I want: https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    bisplrep, bisplrev. Should I just use these old wrappers??

    http://www.slideshare.net/enthought/interpolation-with-scipy-and-numpy
    x and y deals: https://github.com/scipy/scipy/issues/3164'''

    return out_array


def gdal_save_grid(from_dset, outfn, epsg, x_size, y_size, ulx, uly, dx, dy, nanhandle=False, rot0=0, rot1=0):

    # Bring projection info from EPSG to WKT string
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(epsg)
    prj_string = prj.ExportToWkt()

     # Create memory dataset
    if outfn=='MEM':
        mem_drv = gdal.GetDriverByName('MEM')
        dst = mem_drv.Create('',x_size, y_size, 1, gdal.GDT_Float64)

    else:
        gtiff_drv = gdal.GetDriverByName('GTiff')
        dst = gtiff_drv.Create(outfn,x_size, y_size, 1, gdal.GDT_Float64)

    # Assemble geotransform
    geo_t = ( ulx, dx, rot0, \
              uly, rot1, dy )

    # Write geotransform, projection, and array to memory dataset

    dst.SetGeoTransform( geo_t )
    dst.SetProjection( prj_string )
    bnd = dst.GetRasterBand(1)
    if nanhandle!=False:
        from_dset[np.isinf(from_dset)] = nanhandle
        from_dset[np.isnan(from_dset)] = nanhandle
        bnd.SetNoDataValue(nanhandle)
    bnd.WriteArray(from_dset)

    return dst


def mk_gtopo_latlon():
    import numpy as np
    ncols = 3240
    nrows=1680
    xll = 4.5
    yll = 57.5
    cellsize = 0.008333333333
    
    latitudes  = (yll + cellsize/2.) + cellsize*(np.arange(nrows))
    longitudes = (xll + cellsize/2.) + cellsize*(np.arange(ncols))
    
    lonv, latv = np.meshgrid(longitudes, latitudes[::-1])
    
    np.savetxt('/home/wiley/wrk/ryan/gtopo30_lon.asc',lonv)
    np.savetxt('/home/wiley/wrk/ryan/gtopo30_lat.asc',latv)


def Define_grid(ras_fn):
    ds = gdal.Open(ras_fn,gdal.GA_ReadOnly)
    ulx,dx,rot0,uly,rot1,dy = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize

    xs = ulx + np.arange(xsize)*dx
    ys = uly + np.arange(ysize)*dy

    return xs,ys


def Snap_extents_to_grid(grid_x,grid_y,fxmin,fymin,fxmax,fymax):
    xmin = grid_x[np.where(fxmin>=grid_x)[0][-1]]
    xmax = grid_x[np.where(fxmax<=grid_x)[0][0]]
    ymin = grid_y[np.where(fymin>=grid_y)[0][0]]
    ymax = grid_y[np.where(fymax<=grid_y)[0][-1]]

    if (xmin<=fxmin) & (ymin<=fymin) & (xmax>=fxmax) & (ymax>=fymax):
        return xmin,ymin,xmax,ymax
    else:
        return 'SNAP_TO_GRID ERROR'


def transform_epsg2epsg(src_epsg,dst_epsg):
    '''t = transform_epsg2epsg(src_epsg,dst_epsg)
    Both src_epsg and dst_epsg should be a valid integer Authority Code identifying a unique coordinate reference system'''
    src = osr.SpatialReference()
    src.ImportFromEPSG(src_epsg)

    dst = osr.SpatialReference()
    dst.ImportFromEPSG(dst_epsg)

    t = osr.CoordinateTransformation(src,dst)
    return t


def transform_utm33n2geog():
    wgs84 = osr.SpatialReference()
    wgs84.ImportFromEPSG(4326)

    utm33n = osr.SpatialReference()
    utm33n.ImportFromEPSG(32633)

    src_srs = utm33n
    dst_srs = wgs84

    t = osr.CoordinateTransformation(src_srs,dst_srs)

    return t


def t_xy(t,x,y):
    '''transform coordinates (x,y) according to transform t
    returns x,y after given transformation'''
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x,y)
    point.Transform(t)
    
    return point.GetX(), point.GetY()


def mk_utm33_latlon():
    '''Check these by converting to tifs and loading into scene with projection set to geog'''
    ncols    = 1210
    nrows    = 1532
    xll      = -85000.000000000000
    yll      = 6430000.000000000000
    cellsize = 1000.000000000000

    nortings  = (yll + cellsize/2.) + cellsize*(np.arange(nrows))
    eastings  = (xll + cellsize/2.) + cellsize*(np.arange(ncols))

    lonv, latv = np.meshgrid(eastings, nortings[::-1])

    t = transform_utm2geog()
        
    for i in range(len(eastings)):
        for j in range(len(nortings)):
            lonX, latY = t_xy(t,eastings[i],nortings[j])
            lonv[j,i]=lonX
            latv[j,i]=latY
                
    np.savetxt('/home/wiley/wrk/ryan/dem_dir/lon_utm33n.asc',np.flipud(lonv))
    np.savetxt('/home/wiley/wrk/ryan/dem_dir/lat_utm33n.asc',np.flipud(latv))


