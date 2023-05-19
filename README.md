# NTE_ExposureTimeCalculator
An exposure time calculator for NTE.
# How to use:
The wished values of parameters must be inserted in the config.ini file in the data folder. All the parameters are as following:

They are in multiple groups:

[OBSERVATION]
* exposure_time = exposure time in seconds
* nr_of_exposures
* template = {P|B|T}{Index|T in K|FWHM in A}. P for powerlaw, B for plancklaw and T for template.
* template_file = Name of the file containing the spectrum template. Wavelengths in å. Only used if 'template = T'.
* ab_mag = AB magnitude of object
* wavelength_of_ab_mag = Wavelength (in A) where AB mag is given
* fwhm_seeing = The FWHM of the Gaussian PSF in arcsec
* airmass
* moon_stage = {full, half, new, none, custom} Custom will let you input a custom spectrum. The spectra for the moon are taken while looking at the zenith, with the moon positioned at a 45-degree angle from the zenith.
* custom_sky_file = the sky spectrum to use if moon_stage = custom. The moon should be included in this spectrum. Wavelengths in nm and flux in ph/s/m2/micron/arcsec2.
* custom_moon_file = the moon only spectrum to use if moon_stage = custom. Wavelengths in nm and flux in ph/s/m2/micron/arcsec2.
* csv_output = {1|0} If 1, the output will be saved as a csv file.

[SPEC]
* wl_lower_limit = Lower limit wavelength to plot in A
* wl_upper_limit = Upper limit wavelength to plot in A
* slit_width = The width of the slit in arcsec
* slit_length = The length of the slit in arcsec
* detector_binning_dispersion_direction
* detector_binning_spatial_direction
* post_detector_binning
* lambda_split_uvb_vis = Wavelength in microns that separates the UV and VIS flux calculations.  Below this split, UV files and calculations are used, while above it, VIS files and calculations are used.
* lambda_split_vis_ir =  Wavelength in microns that separates the VIS and IR flux calculations. Below this split, VIS files and calculations are used, while above it, IR files and calculations are used.
* ff_error = Flat field error as a fraction.
* tel_area = Telescope area in m2.

Then we have 3 detectors:
[UV_DET]/[VIS_DET]/[IR_DET]
* dark = dark current in electrons/pixel/s 
* ron = readout noise in electrons/pixel 
* pix_length = pixel length in arcsec 
* pix_width = pixel width in arcsec
* arm_min = minimum wavelength for arm in micron
* arm_max = maximum wavelength for arm in micron

There are 3 outputs from the calculator:

* A median S/N for all bands used with be given in the terminal.
* A plot over the S/N as a function of wavelength.
* A plot over the simulated spectrum.

To run the code, run the 'nte_etc_w_uvb_new.py' file from within the folder it is located.

The plots will be saved in the same folder as 'nte\_etc.py'.

## Note on using the template
If you want to use the custom template option, then you indicate it by the letter 'T' in front of the 'template' parameter in the config file, followed by the FWHM of the photometric bandpass used to normalize the spectrum. 

The template file has to contain two columns. One is wavelength in Ångstrum and the other is flux per wavelength in arbitrary units. Then the spectrum you give is normalised to the magnitude you give in the config file.

