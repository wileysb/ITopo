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

import numpy as np
import gdal
import ogr
import osr
import os
import sys

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


def Cast_shade(prj, lat, lon, yday, utc_hour):
    shade_map = np.ones(lat.shape,dtype='int')
    zen_over_horizon = 90-prj['steepest_slope']

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
                    # bog = np.loadtxt(prj['BOG'].format(az, zen), skiprows=6, delimiter=' ')
                    if solar_zen > zen_over_horizon:
                        shade = gdal_load(prj['BOG'].format(int(solar_az),int(solar_zen))).astype('int') #,skiprows=6,dtype='bool',delimiter=' ')
                    else:
                        shade = np.ones_like(lat) # not zeros?
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
    geo_t = ( ulx, dx, rot0,
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

    gdal_args = {'from_dset':lonv, 'outfn':out_ds+'_lon.tif', 'epsg':epsg,
                 'x_size':ncols, 'y_size':nrows, 'ulx':ulx, 'uly':uly, 'dx':cellsize, 'dy':-1*cellsize}
    gdal_save_grid(**gdal_args)
    gdal_args['from_dset'] , gdal_args['outfn'] = latv,out_ds+'_lat.tif'
    gdal_save_grid(**gdal_args)
