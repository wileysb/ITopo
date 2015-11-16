require(raster)
require(insol)

# collect commandline arguments
args<-commandArgs(TRUE)

# parse args
demfile<-args[1]
sol_zen<-strtoi(args[2])
sol_azi<-strtoi(args[3])
shdfile<-args[4]

# load dem
dem=raster(demfile)

# define unit vector from horizontal surface toward sun
sol_vector=normalvector(sol_zen,sol_azi) # solarVector

# make shade map
shd=doshade(dem,sol_vector)

# write shade map to shdfile
writeRaster(shd, filename=shdfile, format='ascii')