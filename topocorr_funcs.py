#!/usr/bin/python

"""
Functions for calculating topographic corrections to surface downwelling irradiance

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
# pyephem, fabulous package for getting angles from an observer on earth's surface to any celestial body at any time
import ephem
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
"""

from scipy.io.netcdf import netcdf_file
from scipy.ndimage.interpolation import zoom
import numpy as np
import gdal
import ogr
import osr
import os
import sys
import datetime as dt


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


# Workflow (scriptlike) functions
def Cast_shade(project_parameters, lat, lon, yday, utc_hour):
    shade_map = np.ones(lat.shape,dtype='int')
    zen_over_horizon = 90-project_parameters['steepest_slope']

    #shade_fmt = os.path.join(shade_dir,'solaraz{0}solarzen{1}.asc')

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
                    # bog = np.loadtxt(project_parameters['BOG'].format(az, zen), skiprows=6, delimiter=' ')
                    if solar_zen > zen_over_horizon:
                        shade = gdal_load(project_parameters['BOG'].format(int(solar_az),int(solar_zen))).astype('int') #,skiprows=6,dtype='bool',delimiter=' ')
                    else:
                        shade = np.ones_like(lat) # not zeros?
                    shade_map[mask]=shade[mask]

    return shade_map


def unpack_srb_variables(srb_fn, project_parameters):
    """

    :param project_parameters:
    :param srb_fn:
    :return:
    """
    srb = netcdf_file(srb_fn,'r',mmap=False)

    srb_gt = project_parameters['srb_gt']

    # AOI boundaries
    lon_s = srb_gt['ulx']  # 4 # 4.5??!
    lon_e = srb_gt['ulx'] + srb_gt['x_size']  # 32 = 4+28
    lat_s = 90+srb_gt['uly'] - srb_gt['y_size']  # 57
    lat_e = 90+srb_gt['uly']  # 72



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

    d = Get_diffuse_fraction(**diffuse_params)

    srb_vars = {'sw_sfc_dn':sw_sfc_dn, 'sw_toa_dn':sw_toa_dn,
                'diffuse':d,
                'lat':lat, 'lon':lon, 'sza':sza, 'year':year,
                'utc_hours':utc_hours, 'ydays':ydays}

    return srb_vars


def srb_to_projectEpsg(srb_3hr_vars, project_parameters):

    wgs84lo = project_parameters['srb_gt']
    wgs84hi = project_parameters['srb_hi_gt']
    project_gt = project_parameters['dem_gt']

    yyyy = srb_3hr_vars['year']

    out_fmt = os.path.join(project_parameters['tmp'], '{0}_{1}_{2}_{3}.tif')#.format(dset, year, yday, utc_hour)

    for i in range(len(srb_3hr_vars['utc_hours'])):
        yday = srb_3hr_vars['ydays'][i]
        utc_hour = srb_3hr_vars['utc_hours'][i]


        dset = 'sw_sfc_dn'
        wgs84lo['from_dset'] = np.flipud(srb_3hr_vars[dset][i,:,:])

        # interpolate from degree resolution to 30arc seconds (0.5arc minutes = 120 per degree)
        wgs84hi['from_dset'] = zoom(wgs84lo['from_dset'], zoom=project_parameters['srb_zoom_factor'])

        # save to gdal memory
        wgs84hi['nanhandle'] = -999
        project_gt['from_dset'] = gdal_save_grid(**wgs84hi)

        # reproject sw_sfc_dn to utm33n, 1km
        project_gt['outfn'] = out_fmt.format(dset, yyyy, yday, utc_hour)
        utm_dset = gdal_resample(**project_gt)

        dset = 'diffuse'
        wgs84lo['from_dset'] = np.flipud(srb_3hr_vars[dset][i,:,:])

        # interpolate from degree resolution to 30arc seconds (0.5arc minutes = 120 per degree)
        wgs84hi['from_dset'] = zoom(wgs84lo['from_dset'], zoom=project_parameters['srb_zoom_factor'])

        # save to gdal memory
        project_gt['from_dset'] = gdal_save_grid(**wgs84hi)

        # reproject sw_sfc_dn to utm33n, 1km
        project_gt['outfn'] = out_fmt.format(dset, yyyy, yday, utc_hour)
        utm_dset = gdal_resample(**project_gt)


# Functions relating solar angle to time, latitude, and longitude
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


# Some general purpose or notemaking geospatial functions:
def Safe_mkdir(dir_path):
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

    gt_dict = {'epsg':int(srs_epsg),
                'x_size': ds.RasterXSize,
                'y_size': ds.RasterYSize,
                'dx':gt[1],
                'dy':gt[5],
                'ulx':gt[0],  # 4.5,
                'uly': gt[3]}#  72.5}
    return gt_dict


def gdal_load(dset_fn):
    dset = gdal.Open(dset_fn)
    band = dset.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    arr= band.ReadAsArray()

    arr = np.where(arr==nodata, np.nan, arr)
    return arr


def gdal_resample(outfn, from_dset, epsg, x_size, y_size, ulx, uly, dx, dy,
                  r=gdal.GRA_Cubic, dtype=gdal.GDT_Float64, nanhandle=False, rot0=0, rot1=0):

    # Get from_projection
    src_projection_string = from_dset.GetProjection()

    # Bring projection info from EPSG to WKT string
    dst_projection = osr.SpatialReference()
    dst_projection.ImportFromEPSG(epsg)
    dst_projection_string = dst_projection.ExportToWkt()

    # Create memory dataset
    if outfn=='MEM':
        mem_drv = gdal.GetDriverByName('MEM')
        dst = mem_drv.Create('',x_size, y_size, 1, dtype)

    else:
        gtiff_drv = gdal.GetDriverByName('GTiff')
        dst = gtiff_drv.Create(outfn,x_size, y_size, 1, dtype)

    # Assemble geotransform
    geo_t = ( ulx, dx, rot0, \
              uly, rot1, dy )

    # Write geotransform, projection, and array to memory dataset
    dst.SetGeoTransform( geo_t )
    dst.SetProjection( dst_projection_string )

    # can result in negatives? nans?
    res = gdal.ReprojectImage(from_dset, dst, \
          src_projection_string, dst_projection_string, \
          r )# gdal.GRA_Average) #gdal.GRA_Cubic) #gdal.GRA_Bilinear )

    return dst


def gdal_save_grid(from_dset, outfn, epsg, x_size, y_size, ulx, uly, dx, dy, nanhandle=False, rot0=0, rot1=0):

    # Bring projection info from EPSG to WKT string
    projection = osr.SpatialReference()
    projection.ImportFromEPSG(epsg)
    projection_string = projection.ExportToWkt()

     # Create memory dataset
    if outfn=='MEM':
        mem_drv = gdal.GetDriverByName('MEM')
        dst = mem_drv.Create('',x_size, y_size, 1, gdal.GDT_Float64)

    else:
        gtiff_drv = gdal.GetDriverByName('GTiff')
        dst = gtiff_drv.Create(outfn,x_size, y_size, 1, gdal.GDT_Float64)

    # Assemble geotransform
    geo_t = ( ulx, dx, rot0,
              uly, rot1, dy )

    # Write geotransform, projection, and array to memory dataset

    dst.SetGeoTransform( geo_t )
    dst.SetProjection( projection_string )
    bnd = dst.GetRasterBand(1)
    if nanhandle!=False:
        from_dset[np.isinf(from_dset)] = nanhandle
        from_dset[np.isnan(from_dset)] = nanhandle
        bnd.SetNoDataValue(nanhandle)
    bnd.WriteArray(from_dset)

    return dst


def transform_epsg2epsg(src_epsg,dst_epsg):
    """t = transform_epsg2epsg(src_epsg,dst_epsg)
    Both src_epsg and dst_epsg should be a valid integer
    Authority Code identifying a unique coordinate reference system"""
    src = osr.SpatialReference()
    src.ImportFromEPSG(src_epsg)

    dst = osr.SpatialReference()
    dst.ImportFromEPSG(dst_epsg)

    t = osr.CoordinateTransformation(src,dst)
    return t


def t_xy(t,x,y):
    '''transform coordinates (x,y) according to transform t
    returns x,y after given transformation'''
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x,y)
    point.Transform(t)
    
    return point.GetX(), point.GetY()


def mk_latlon_grids(ncols,nrows,ulx,uly,cellsize, epsg, out_ds):

    nortings  = (uly - cellsize/2.) - cellsize*(np.arange(nrows))
    eastings  = (ulx + cellsize/2.) + cellsize*(np.arange(ncols))

    lonv, latv = np.meshgrid(eastings, nortings[::-1])

    t = transform_epsg2epsg(epsg,4326)

    for i in range(len(eastings)):
        for j in range(len(nortings)):
            lonX, latY = t_xy(t,eastings[i],nortings[j])
            lonv[j,i]=lonX
            latv[j,i]=latY
        gdal.TermProgress_nocb(i/float(len(eastings)))

    gdal_args = {'from_dset':lonv, 'outfn':out_ds+'_lon.tif', 'epsg':epsg,
                 'x_size':ncols, 'y_size':nrows, 'ulx':ulx, 'uly':uly, 'dx':cellsize, 'dy':-1*cellsize}
    gdal_save_grid(**gdal_args)
    gdal_args['from_dset'] , gdal_args['outfn'] = latv,out_ds+'_lat.tif'
    gdal_save_grid(**gdal_args)
    gdal.TermProgress_nocb(100)
