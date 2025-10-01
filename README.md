# Contact Info: 
- Name: Aysan Hemmatiortakand 
- Email address: Personal : Aysanhemmatiortakand@gmail.com , U of A: hemmatio@ualberta.ca

# Brief Summery of What Each Code does: 
## find_star.py:

### Inputs: 
- folder_path: Location of your images.
- save_directory: Where to save the .npy file containing the coordinates.
- save_filename: What to name the aformentioned .npy file

### Outputs: 
- An .npy file containing the location to the star in question for the list of .fits files in the input folder. 

### How to operate: 
After typing in the needed parameters mentioned in **inputs**, images from the chosen folder pop up one by one. click on the **center** of the star that you are interested in. Keep doing this until there are no images left. the list of all the places you have clicked will be saved to an .npy file, so you wouldn't have to do it again.  

## B) PSF.py

### inputs:
- box_size: the size of the array cutout around the given star coordinate. Change according to seeing conditions and FOV. 
- pixel_scale: in arcseconds, currently set for the INO.
- color: the colormap used for plots.
- filter: filter : u, g, r, i for INO
- mode: "High", "Low" or "Merge" for INO
- folder_path: Location of your images
- star_coordinates_loc: Location of the coordinates
- specific_plot_idx: Index of the specific plot to display separately (0-based index)

### outputs: 
- image of all cutouts
- radial profiles of all stars + HWHM values in pixels + COM line 
- radial profile of one specific image (specific_plot_idx)
- Histogram of the PSF distribution +  Median PSF in arcseconds
- list of FWHM for each image in pixels

## C) star_flux.py

### inputs: 
- box_size: the size of the array cutout around the given star coordinate. Change according to seeing conditions and FOV. 
- seeing: PSF in arcseconds (from subroutine B)
- aperture ratio: aperture ratio $\times$ PSF is the radius of the circular aperture around the star.
- pixel_scale: in arcseconds, currently set for the INO.
- color: the colormap used for plots.
- folder_path: Location of your images
- star_coordinates_loc: Location of the coordinates (output of sub-routine A)

### Outputs: 
- image of each star + apertures ()
- Mean Star Flux
- Mean Sky Flux
- Star Flux Minus Sky Flux

# Ancillary codes:

## object_visibilty.py
(a function version of this is in ancillary_functions.py)
### Inputs: 

- Observer info: 
    1) latitude
    2) longitude
    3) elevation 
    4) timezone 
    5) name of observatory
- date: "year-month-day" format 
- mid_day: "12:00:00"
- n_steps: number of points in the plot
- star_coords: "name" : ("HH:MM:SS", "DD:MM:SS")

### Outputs: 

- An object visibilty plot for each star
- An object visibilty plot of all stars together. 


## airmass.py
(a function version of this is in ancillary_functions.py)
### inputs: 

location : INO_LOCATION = EarthLocation(lat=33.674 * u.deg,
                             lon=51.3188 * u.deg,
                             height=3600 * u.m)
date : "year-month-day"
hour : e.g. 21 (utc or local, depends on input_timezone)
minute: e.g. 0
RA: e.g. "23:13:38.8"
DEC: e.g. "+39:25:02.6"
input_timezone: "UTC" or "Aisa/Teharn" or ...
plot_night_curve: False or True, if True it plots a altitude vs. time plot colored by airmass. 