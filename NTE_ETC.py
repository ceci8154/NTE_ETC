"""
Original code was written by Bjarne Thomsen.

Translated and additional code written by Cecilie Valet Henneberg and Mads Nymann-Lynggaard

Comments also written by Bjarne Thomsen.
"""



# This is a translation of code originally by Bjarne Thomsen

import configparser
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.special import erf


def gauss (x):
    """
    Caclulate the normalized Gaussian function for the array x.

    :param x: array of values. In this code x is used as (arr - arr0)/sigma)
    :return: The values of the Gaussian.
    """
    return np.exp(-0.5 * x**2)/np.sqrt(2.0 * np.pi)


def hgauss (x,s):
    """
    Generates a semi-realistic object profile along the slit.
    hgauss(x, s) = exp(-(sqrt(1 + (x*s)^2) - 1)/x^2)
                 = exp(-2.0*(sinh(0.5*asinh(x*s))/x)^2)
    This is a simple way to obtain a profile resembling a disk galaxy
    convolved by a Gaussian seeing.

    :param x: Exponential scale divided by Gaussian sigma.
    :param s: Distance from center of slit in sigma units.
    :return: Signal profile along the slit.
    """
    return np.exp(-2.0*(np.sinh(0.5 * asinh(x*s))/x)**2)


def asinh (x):
    """
    Calculates the inverse of the hyperbolic sine function.
    asinh(x) = sign(x)*asinh(abs(x))
    asinh(x) = alog(sqrt(1 + x^2) + x)

    :param x: Real scalar or array.
    :return: Inverse of the hyperbolic sine function.
    """
    # use that asinh(x) = sign(x)*asinh(abs(x))
    z = abs(x)
    # calculate asinh(x) for positive arguments, only
    i = x >= 0
    x[i] = np.log(np.sqrt(z[i]**2 + 1) + z[i])
    j = x < 0
    x[j] = -np.log(np.sqrt(z[j]**2 + 1) + z[j])
    return x


def planck(x, T):
    """
    Calculates the specific photon flux from
    a black body of themperature T kelvin.
    It is given in ph/s/m^2/um/arcsec^2

    :param x: The given wavelengths in um
    :param T: Temperature in kelvin
    :return: Specific photon flux at x
    """
    return 1.40929e16/(np.exp(1.43879e4/(T*x)) - 1)/x**4


def objem_at_all (func, abmag, maglam, param, sigma, lambd):
    """
    Calculates the specific photon flux from a selected
    spectral law or template at a given wavelength with
    given spectral resolution, as specified by a 
    Gaussian sigma.

    :param func: String. The spectral law or template to use.
    :param abmag: AB magnitude at specified wavelength
    :param maglam: Wavelength (in um) where abmag is given.
    :param param: Spectral index, temperature or bandwidth.
    :param sigma: The sky spectrum, as transmitted through
                  the atmosphere, is convolved with a 
                  Gaussian having the dispersion sigma[i]
                  at lambda[i]. The sigmas are given in um.
    :param lambd: The wavelengths given in um.
    :return: Specific photon flux at lambd given in
             units photons/s/m^2/um.
    """
    # convolve the object spectrum with a Gaussian
    y = lambd.copy()
    # number of points on each side of x0
    N = 30
    tmp_arange = np.arange(2*N + 1) - N
    for i in range(len(lambd)):
        # wavelength of sample point No. i
        x0 = lambd[i]
        # Gaussian sigma of sample point No. i
        s0 = sigma[i]
        # step size
        step = 0.4 * s0
        x = (tmp_arange)*step + x0
        # photon fluxes of the transmitted objec spectrum at the discrete wavelengths
        flux = step * atmos_trans(x)**am * call_func(func, abmag, maglam, param, x)
        # calculate the Gaussian weighted sum of the transmitted object spectrum
        y[i] = sum(flux * gauss((x - x0)/s0))/s0
    return y


def atmos_trans (lambd):
    """
    Calculates the atmospheric transmission at any
    wavelength in the full range.

    :param lambd: Wavelength in um.
    :return: Atmospheric transmission.
    """
    # Split between a UV-Visual part and a NIR part of the spectrum
    y = lambd.copy()
    k = lambd < lambda_split_uvb_vis
    y[k] = qes_and_intps('luvb', lambd[k], 'trans_at_luvb')
    k =  (lambd < lambda_split_vis_ir) & (lambd >= lambda_split_uvb_vis)
    y[k] = qes_and_intps('uvis', lambd[k], 'trans_at_uvis')
    k = lambd >= lambda_split_vis_ir
    y[k] = qes_and_intps('nir', lambd[k], 'trans_at_nir')
    return y**am


def powerlaw (abmag, maglam, alpha, lambd):
    """
    Calculates a power-law continuum with a spectral
    index, alpha and an AB magnitude, abmag, at maglam.
    The photon flux density is given in photons/s/m^2/um.

    :param abmag: AB magnitude at specified wavelength
    :param maglam: Wavelength (in um) where abmag is given.
    :param alpha: Spectral index (F_nu ~ 1/nu^alpha).
    :param lambd: The wavelengths given in um.
    :return: Specific photon flux at lambd given in
             units photons/s/m^2/um.
    """
    y = (lambd/maglam)**alpha * 10**(0.4*(mag_zero_point - abmag))/lambd
    return y


def plancklaw (abmag, maglam, T, lambd):
    """
    Calculates a Planck-law continuum with temperature T,
    and an AB magnitude, abmag, at maglam.
    The photon flux density is given in ph/s/m^2/um.

    :param abmag: AB magnitude at specified wavelength
    :param maglam: Wavelength (in um) where abmag is given.
    :param T: Temperature in kelvin
    :param lambd: The wavelengths given in um.
    :return: Specific photon flux at lambd given in
             units photons/s/m^2/um.
    """
    c2 = 14387.9
    a = 0.5*c2/T
    y = np.exp(a/maglam - a/lambd)*np.sinh(a/maglam)/np.sinh(a/lambd)* \
        (maglam/lambd)**4 * 10**(0.4*(mag_zero_point - abmag))/maglam
    return y


def template (lambd):
    """
    Calculates the photon flux at lamda from the 
    interpolation done in read_template().
    All wavelength are given in um, and the photon flux
    density is given in photons/s/m^2/um.

    :param lambd: The wavelengths given in um.
    :return: Specific photon flux at lambd given in
             units photons/s/m^2/um.
    """
    x_new = lambd
    f = y_spl_template(x_new)
    return f


def call_func (func, abmag, maglam, param, x):
    """
    Decides which function to use based on func string.

    :param func: String with the name of the function to use.
    :param abmag: AB magnitude at specified wavelength
    :param maglam: Wavelength (in um) where abmag is given.
    :param param: Spectral index, temperature or bandwidth.
    :param x: The wavelengths given in um.
    :return: Specific photon flux at x given in
             units photons/s/m^2/um.
             Based on specified function.
    """
    if func == 'powerlaw':
        f = powerlaw(abmag, maglam, param, x)
    elif func == 'plancklaw':
        f = plancklaw(abmag, maglam, param, x)
    elif func == 'template':
        f = template(x)
    return f


def read_template (abmag, maglam, bandwidth, name):
    """
    Reads data from the template spectrum file.
    The file must contain wavelengths in å.
    Calculates a template spectrum with an AB magnitude
    abmag at central wavelength maglam through a photometric
    system with a FWHM bandpass of bandwidth.
    Makes a spline interpolation of the template spectrum,
    with wavelengths in um, and the photon flux
    density in photons/s/m^2/um.

    :param abmag: AB magnitude in specified photometric system.
    :param maglam: Central wavelength (in um) of the bandpass.
    :param bandwidth: FWHM (in um) of the photometrix system.
    :param name: Name of the template spectrum file.
    No output.
    It puts the data in the global variables:
    - template_x - wavelengths in um
    - template_y - flux per wavelength in arbitrary units
    - y_spl_template - spline interpolation of template_y
    """
    global template_x, template_y, y_spl_template
    template_data = np.loadtxt('data/'+str(name))
    # Wavelength in um
    template_x = template_data[:,0] / 10000
    # Spectral flux in arbitrary units per wavelengthinterval
    template_y = template_data[:,1]
    # Flux density must be non negative
    template_y = template_y.clip(min=1.0E-30)
    #The sampling must be sorted according to increasing wavelength
    k = np.argsort(template_x)
    template_x = template_x[k]
    template_y = template_y[k]

    n = len(template_x)
    x = template_x
    y = template_y
    # the bandpass must not be wider than half the spectral range
    fwhm = min(bandwidth, (0.5*(x[-1] - x[0])))
    # the central wavelength must not be too close to the endpoints
    xc = min(max(maglam, (x[0] + fwhm)), (x[-1] - fwhm))
    # select all spectral samples within [xc - fwhm, xc + fwhm]
    k = (x >= (xc - fwhm)) & (x <= (xc + fwhm))
    # what should we do if there are no samples in this interval?
    if sum(k) == 0:
        fwhm = 0.5*(x[-1] - x[0])
        xc = 0.5*(x[-1] + x[0])
        k = np.arange(n)
    # calculate the bandpass profile
    P = np.exp(-np.log(2.0) * (2.0 * (x[k] - xc)/fwhm)**4)
    # calculate the scaling factor
    S = 10.0**(0.4 * (mag_zero_point - abmag)) * sum(P/x[k])/sum(P * x[k] * y[k])
    # calculate the photon flux (photons/s/m^2/um) at the sample points
    y = S * x * y
    # we should not extrapolate outside endpoints, instead we use the values at the endpoints.
    y_spl_template = UnivariateSpline(x, y, s=0, k=2)


def read_vis_sky ():
    """
    This function imports the needed data of the night-sky
    in the visual part of the spectrum.

    No input and no output.
    It puts the data in the global variables:
    - vissky_x - wavelengths in um
    - vissky_y - specific photon flux in photons/s/m^2/um
    """
    global vissky_x, vissky_y
    sky_data = np.loadtxt(sky_filename)
    # extract wavelengths into um
    vissky_x = sky_data[:,0] /1000.0
    # and photon fluxes in photons/m^2/s/um/arcsec^2
    vissky_y = sky_data[:,1]
    # the flux must be non negative
    vissky_y = vissky_y.clip(min=0.0)


def read_nir_abs ():
    """
    Reads data on atmospheric NIR absorption.

    No input and no output.
    It puts the data in the global variables:
    - nir_x - wavelengths in um
    - nir_y - transmission
    """
    global nir_x, nir_y
    abs_data = np.loadtxt('data/skycalc_abs.dat')
    # extract wavelengths (in um) and transmissions.
    nir_x = abs_data[:,0] /1000.0 # in um
    nir_y = abs_data[:,1]
    # the transmission must be in the range [0,1].
    nir_y = nir_y.clip(min=1.0e-6, max=1.0)


def read_uvis_ext ():
    """
    Reads data on atmospheric UV/Vis extinction.

    No input and no output.
    It puts the data in the global variables:
    - uvis_x - wavelengths in um
    - uvis_y - transmissions
    """
    global uvis_x, uvis_y
    ext_data = np.loadtxt('data/UVIS-ext.dat')
    # extract wavelength (in um) and transmission of the atmosphere.
    uvis_x = ext_data[:,0] /1000.0
    uvis_y =  10.0**(-0.4 * ext_data[:,1])


def read_vis_qe ():
    """
    Reads data on efficiency of NTE in VIS.

    No input and no output.
    It puts the data in the global variables:
    - vis_x - wavelengths in um
    - vis_y - efficiency
    """
    global vis_x, vis_y
    qe_data = np.loadtxt('data/efficiency/skipper_eff.csv', delimiter=',')
    # extract wavelength (in um) and QE of VIS arm of NTE.
    vis_x = qe_data[:,0]
    vis_y = qe_data[:,-1]
    # the QE must be in the range [0,1].
    vis_y = vis_y.clip(min=1.0e-6, max=1.0)


def read_oh ():
    """
    Reads data on OH lines.

    No input and no output.
    It puts the data in the global variables:
    - oh_x - wavelengths in um
    - oh_y - line strengths in ph/m^2/s/arcsec^2
    """
    global oh_x, oh_y
    oh_data = np.loadtxt('data/OH-lines.dat')
    # convert wavelengths to um.
    oh_x = oh_data[:,0]/10000.0
    # convert wavelengths from vacuum to air
    oh_x = oh_x / ((255.4/(41 - (1/oh_x)**2) + 29498.1/(146 - (1/oh_x)**2) + 64.328)*1e-6 + 1)
    # save calibrated photon fluxes (in ph/m^2/s/arcsec^2). the calibration is done by using ESO's ETC.
    oh_y = oh_data[:,1]/38.4
    # the fluxes must be non-negative.
    oh_y = oh_y.clip(min=0.0)


def read_ir_qe ():
    """
    Reads efficiency of NTE in IR.

    No input and no output.
    It puts the data in the global variables:
    - ir_x - wavelengths in um
    - ir_y - efficiency
    """
    global ir_x, ir_y, ir_y2
    qe_data = np.loadtxt('data/efficiency/h2rg_eff.csv', delimiter=',')
    # extract wavelength (in um) and QE of IR arm of NTE.
    ir_x = qe_data[:,0]
    ir_y = qe_data[:,-1]
    # the QE must be in the range [0,1].
    ir_y = ir_y.clip(min=1.0e-6, max=1.0)


def read_uvb_sky ():
    """
    Reads data on uvb night-sky emissions.

    No input and no output.
    It puts the data in the global variables:
    - uvbsky_x - wavelengths in um
    - uvbsky_y - photon fluxes in ph/m^2/s/um/arcsec^2
    """
    global uvbsky_x, uvbsky_y
    sky_data = np.loadtxt(sky_filename)
    # extract wavelength in um
    uvbsky_x = sky_data[:,0] /1000.0
    # and photon fluxes in photons/m^2/s/um/arcsec^2
    uvbsky_y = sky_data[:,1]
    # the flux must be non negative
    uvbsky_y = uvbsky_y.clip(min=0.0)


def read_uvb_qe ():
    """
    Reads NTE efficiency in uvb range.

    No input and no output.
    It puts the data in the global variables:
    - uvb_x - wavelengths in um
    - uvb_y - efficiency
    """
    global uvb_x, uvb_y
    qe_data = np.loadtxt('data/efficiency/em_eff.csv', delimiter=',')
    # extract wavelength (in um) and QE of UVB arm of NTE
    uvb_x = qe_data[:,0]
    uvb_y = qe_data[:,-1] 
    # the QE must be in the range [0,1].
    uvb_y = uvb_y.clip(min=1.0e-6, max=1.0)


def read_luvb_ext ():
    """
    Reads uvb extinction data.

    No input and no output.
    It puts the data in the global variables:
    - luvb_x - wavelengths in um
    - luvb_y - extinction
    """
    global luvb_x, luvb_y
    luvb_data = np.loadtxt('data/UVIS-ext.dat')
    # extract wavelength (in um) and transmission of the atmosphere.
    luvb_x = luvb_data[:,0]/1000
    luvb_y = 10.0**(-0.4 * luvb_data[:,1])


def init_snc_at_vis_ir_and_uvb (type, slit_width, disk_scale, psf_fwhm, bin_w=0, bin_l=0):
    """
    Decides which data to read give the type

    :param type: Which part of the spectrum is used
    :param slit_width: The width of the slit in arcsec
    :param disk_scale: The exponential scale of the disk in arcsec
    :param psf_fwhm: The FWHM of the Gaussian PSF in arcsec

    No output. It just reads data into the globals.
    Also depending on :param type: it will read different
    constants into the globals.
    """
    if type == 'vis':
        # read night sky emission data
        read_vis_sky()
        # read QE data for the VIS arm of NTE into
        read_vis_qe()
    elif type == 'ir':
        # read OH data
        read_oh()
        # read QE data for the IR arm of NTE
        read_ir_qe()
    elif type == 'uvb':
        # read night sky emission data
        read_uvb_sky()
        # read QE data for the UV arm of NTE into
        read_uvb_qe()

    # read atmospheric absorption data
    read_nir_abs()
    # read atmospheric extinction data
    read_uvis_ext()
    # read atmospheric extinction data
    read_luvb_ext()
    
    # copy parameters into the common block
    global s_width, s_eff, psf_sig, h_s, pix_length, pix_width, \
        pix_ron, pix_dark
    s_width = slit_width    # slit width in arcsec
    s_eff = erf(np.sqrt(np.log(2.0)) * slit_width / psf_fwhm)   # flux fraction passing slit
    psf_sig = psf_fwhm/np.sqrt(8.0*np.log(2.0)) # Sigma of Gaussian PSF in arcsec
    h_s = disk_scale/psf_sig    # disk scale of galaxy to PSF sigma
    if type == 'vis':
        pix_length = pix_length_vis   # pixel length in arcsec 
        pix_width = pix_width_vis    # pixel width in arcsec 
        pix_ron = ron_vis   # readout noise in electrons/pixel 
        pix_dark = dark_vis    # dark current in electrons/pixel/s 
    elif type == 'uvb':
        pix_length = pix_length_uvb   # pixel length in arcsec 
        pix_width = pix_width_uvb    # pixel width in arcsec 
        pix_ron = ron_uvb   # readout noise in electrons/pixel 
        pix_dark = dark_uvb    # dark current in electrons/pixel/s 
    elif type == 'ir':
        pix_length = pix_length_ir   # pixel length in arcsec 
        pix_width  = pix_width_ir   # pixel width in arcsec 
        pix_ron = ron_ir   # readout noise in electrons/pixel 
        pix_dark = dark_ir     # dark current in electrons/pixel/s 


def disp_at_vis_ir_and_uvb (type, x):
    """
    Calculates the spectral dispersion in arcsec/um
    at any given wavelength for any arm.

    :param type: Which part of the spectrum is used
    :param x: The wavelength in um
    :return: The dispersion in arcsec/um
    """
    if type == 'vis':
        r = 4000.0
    elif type == 'ir':
        r = 4000.0
    elif type == 'uvb':
        r = 4000.0
    return r / x


def lamgen_at_vis_ir_and_uvb (type, lambda_min, lambda_max):
    """
    Generates an array of wavelengths.

    :param type: Which part of the spectrum is used
    :param lambda_min: The minimum wavelength in um
    :param lambda_max: The maximum wavelength in um
    :return: An array of wavelengths in um.
             Wavelength array is on a logarithmix scale.
    """
    # the middle wavelength on a logarithmic scale
    lambda0 = np.sqrt(lambda_min * lambda_max)
    # size of a pixel in wavelength units at lambda0
    dlambda0 = pix_width / disp_at_vis_ir_and_uvb(type, lambda0)
    x_min = min(lambda_min, lambda_max)
    x_max = max(lambda_min, lambda_max)
    ratio = x_max/x_min
    # number of wavelength points in output array
    N = np.round(lambda0 * np.log(ratio)/dlambda0) + 2
    # float array in interval [0.0, 1.0]
    x = np.arange(N, dtype=float)/(N - 1)
    x = x_min * ratio**x
    return x


def y_spls (type):
    """
    This functions makes interpolations of the data needed for
    the given range.

    No input or output:
    It takes makes the interpolations as globals.
    """
    global y_spl_vis, y_spl_nir, y_spl_uvis, y_spl_vissky, y_spl_ir, y_spl_uvb, y_spl_uvbsky, y_spl_luvb
    if moon_stage != 'none':
        make_moon_interpolation_func()
    if type == 'vis':
        y_spl_vis = UnivariateSpline(vis_x, vis_y, s=0, k=3)
        y_spl_nir = UnivariateSpline(nir_x, nir_y, s=0, k=3)
        y_spl_uvis = UnivariateSpline(uvis_x, uvis_y, s=0, k=3)
        y_spl_luvb = UnivariateSpline(luvb_x, luvb_y, s=0, k=3)
        y_spl_vissky = interp1d(vissky_x, vissky_y, kind='slinear')
    elif type == 'ir':
        y_spl_ir = UnivariateSpline(ir_x, ir_y, s=0, k=3)
        y_spl_nir = UnivariateSpline(nir_x, nir_y, s=0, k=3)
        y_spl_uvis = UnivariateSpline(uvis_x, uvis_y, s=0, k=3)
        y_spl_luvb = UnivariateSpline(luvb_x, luvb_y, s=0, k=3)
    elif type == 'uvb':
        y_spl_uvb = UnivariateSpline(uvb_x, uvb_y, s=0, k=3)
        y_spl_nir = UnivariateSpline(nir_x, nir_y, s=0, k=3)
        y_spl_uvis = UnivariateSpline(uvis_x, uvis_y, s=0, k=3)
        y_spl_luvb = UnivariateSpline(luvb_x, luvb_y, s=0, k=3)
        y_spl_uvbsky = interp1d(uvbsky_x, uvbsky_y, kind='slinear')


def qes_and_intps (name, lambd, t):
    """
    Collective function for evaluation the interpolation.

    :param name: What part of the spectrum is used.
    :param lambd: The given wavelength range in um.
    :param t: The type of interpolation. (Needed because 
              additional processing might be needed for
              specific interpolations)
    :return: The interpolated values at lambd.
    """
    x_name = globals()[name + '_x']
    y_name = globals()['y_spl_' + name]
    x = np.clip(lambd, x_name[0], x_name[-1])
    y = y_name(x)
    if t == 'qe_at_ir' or t == 'qe_at_vis' or t == 'qe_at_uvb':
        y = pessimism_factor * y
    elif t == 'trans_at_nir':
        y = y.clip(min=0.0)
    elif t == 'trans_at_uvis' or t == 'intp_vis_sky' or t == 'intp_uvb_sky' or t == 'trans_at_luvb':
        y = y
    else:
        print('MISTAKE: wrong type inserted in qes_and_intps')
        exit()
    return y


def make_moon_interpolation_func():
    """
    Makes the function for interpolation of the moon spectrum.

    No input and no output.
    Makes the function for the moon interpolation to a global
    """
    component_data = np.loadtxt(sky_component_filename)
    wl = component_data[:, 0] / 1000
    moon = component_data[:, 1]
    global moon_interpolation
    moon_interpolation = interp1d(wl, moon, kind='slinear')


def skycont_at_ir (lambd):
    """
    Calculates the near infrafred continuum between the
    OH lines, as defined by values in J at 1.25 um
    and in H at 1.67 um. A linear interpolation in 
    ln(photon_flux) - ln(lambda) is done.
    The photon flux is given in ph/s/m^2/um/arcsec^2.

    :param lambd: The given wavelengths in um
    :return: Specific photon flux at lambda.
    """
    lamb_J = 1.25
    flux_J = 310.0
    lamb_H = 1.665
    flux_H = 590.0
    x = lambd
    HoJ = np.log(lamb_H/lamb_J)
    exp_J = np.log(lamb_H/x)/HoJ
    exp_H = np.log(x/lamb_J)/HoJ
    y = flux_J**exp_J * flux_H**exp_H
    if moon_stage != 'none':
        y += moon_interpolation(x)
    return y


def thermalem_at_ir (lambd):
    """
    Calculates the near infrared thermal continuum.
    A tempertaure of T = 288K and an emissivity of
    emis = 0.25 are assumed. The photon flux is
    given in ph/s/m^2/um/arcsec^2.

    :param lambd: The given wavelenghts in um.
    :return: Specific photon flux at lambda.
    """
    Temp = 288.0
    emis = 0.25
    x = lambd
    y = (1 - (1 - emis) * qes_and_intps('nir', x, 'trans_at_nir')) * planck(Temp, x)
    return y


def skyem_at_vis_ir_and_uvb (type, sigma, lambd):
    """
    Calculates the specific photon flux from the night
    sky at a given wavelength in the VIS/IR/UVB region
    with given spectral resolution, as specified by a
    Gaussian sigma.

    :param type: String that determines if VIS/IR/UVB is used
    :param sigma: The sky spectrum, as transmitted
                  through the dispersion sigma[i] 
                  at lambda[i].
                  The sigmas are given in um.
    :param lambd: The wavelengths given in um.
    :return: Specific photon flux at lambd
             given in units of photons/m^2/s/um/arcsec^2.
    """
    if type == 'vis':
        # step size
        step = 0.0000030
    elif type == 'ir':
        # photon fluxes of the transmitted OH lines
        oh_flux = qes_and_intps('nir', oh_x, 'trans_at_nir')**am * oh_y
    elif type == 'uvb':
        step = 0.0000025
    y = lambd.copy()
    # convolve the sky spectrum with a Gaussian
    for i in range(len(lambd)):
        # wavelength of sample point No. i
        x0 = lambd[i]
        # Gaussian sigma of sample point No. i
        s0 = sigma[i]
        if type == 'vis' or type == 'uvb':
            # number of points on each side of x0
            N = np.round(12.0*s0/step)
        elif type == 'ir':
            # number of points on each side of x0
            N = 30
            # step size
            step = 0.4 * s0
        # range [-12*s0, +12*s0] of wavelengths around the point x0
        x = (np.arange(2*N + 1) - N)*step + x0
        if type == 'vis':
            # photon fluxes of the transmitted spectrum at the discrete wavelengths
            flux = step * qes_and_intps('uvis', x, 'trans_at_uvis')**am * qes_and_intps('vissky', x, 'intp_vis_sky')
        elif type == 'ir':
            # photon fluxes of the transmitted continuum at the discrete wavelengths
            flux = step * qes_and_intps('nir', x, 'trans_at_nir')**am * skycont_at_ir(x)
        elif type == 'uvb':
            # photon fluxes of the transmitted spectrum at the discrete wavelengths
            flux = step * qes_and_intps('luvb', x, 'trans_at_luvb')**am * qes_and_intps('uvbsky', x, 'intp_uvb_sky')

        # calculate the Gaussian weighted sum of the transmitted sky spectrum.
        y[i] = sum(flux * gauss((x - x0)/s0))/s0
        if type == 'ir':
            # distances of the OH lines from the sample point x0, in units of the Gaussian sigma s0.
            z = (oh_x - x0)/s0
            # select the lines that deviates less than 12*sigma, and add the Gaussian weighted sum of the transmitted fluxes.
            k = abs(z) < 12.0
            y[i] = y[i] + sum(oh_flux[k]*gauss(z[k]))/s0

    if type == 'ir':
        y = y + thermalem_at_ir(lambd)

    return y


def s2n (ron, fe, dark, h_s, sig, flux, sky, x):
    """
    Calculates the S/N per pixel of a profile fit along the slit.

    :param ron: Readout noise in electrons per pixel.
    :param fe: Fractional flat-field error.
    :param dark: Dark current in electrons per pixel.
    :param h_s: <exp scale length> / <Gaussian sigma>
    :param sig: Sigma of the Gaussian seeing core in pixels.
    :param flux: Flux in electrons per exposure time.
    :param sky: Sky flux in electrons per pixel per exp time.
    :return: S/N per pixel along dispersion direction.
    """
    # calculate the object profile
    P = hgauss(h_s, x/sig)
    # normalize the profile sum to 1.0
    P = P/sum(P)
    # calculate the background variance per pixel
    bg = sky + dark + ron**2 + (fe * sky)**2
    # calculate the weight per pixel along the slit
    W = 1/(flux * P + bg)
    # shift the profile such that total(W*Q) = 0
    Q = P - sum(W*P)/sum(W)
    # calculate the signal-to-noise per pixel
    y = flux * np.sqrt(sum(W*Q**2))
    return y


def snc_at_vis_ir_and_uvb(type, nexp, etime, func, abmag, maglam, param, lambd):
    """
    Calculates the signal-to-noise (per pixel) for a range
    of given wavelengths for an arm.

    :param type: Which arm is wanted to use.
    :param nexp: Number of single exposures.
    :param etime: Exposure time in s.
    :param func: Which spectral law is wanted.
    :param abmag: AB magnitude at specified wavelength.
    :param maglam: Wavelength (in um) where abmag is given.
    :param param: Spectral index (alpha) or temperature.
    :param lambd: The given wavelengths in um.
    :return: The Signal-to-Noise per pixel.
    """
    # number of wavelength points
    Npts = len(lambd)
    # Gaussian sigmas in wavelength space corresponding to the given slit width
    sigma = s_width * gauss(0.0) / disp_at_vis_ir_and_uvb(type, lambd)
    # number of pixels along the slit
    Npix = np.round(s_length / pix_length)
    # wavelength interval corresponding to the size of a pixel
    dlambda = pix_width / disp_at_vis_ir_and_uvb(type, lambd)
    # QE at each of the wavelength points
    if type == 'vis':
        qe = qes_and_intps(type, lambd, 'qe_at_vis')
    elif type == 'ir':
        qe = qes_and_intps(type, lambd, 'qe_at_ir')
    elif type == 'uvb':
        qe = qes_and_intps(type, lambd, 'qe_at_uvb')
    # sky emission in photoelectrons/m^2/s/arcsec^2
    skyem = qe * dlambda * skyem_at_vis_ir_and_uvb(type, sigma, lambd)
    # sky emission in photoelectrons/pixel/exposure
    skyem = (tel_area * etime * pix_length * s_width) * skyem
    # object flux in photons/m^2/s/um
    objem = objem_at_all(func, abmag, maglam, param, sigma, lambd)
    # object flux in photoelectrons/m^2/s
    objem = qe * dlambda * objem
    # object flux in photoelectrons/exposure
    objem = (tel_area * etime * s_eff) * objem
    # dark current in electrons/pixel/exposure
    dark = pix_dark * etime
    # sigma of the Gaussian seeing core in pixels along the slit.
    sig_pix = psf_sig / pix_length
    # calculate the S/N for each wavelength
    y = np.zeros((Npts, 2))
    # there are Npix pixels along the slit
    x_s2n = np.arange(Npix, dtype=float)
    # place the object profile at the center of the slit
    x_s2n = x_s2n - (Npix - 1)/2
    for i in range(Npts):
        # let us first obtain the S/N for a single exposure
        y[i,0] = s2n(pix_ron, ff_error, dark, h_s, sig_pix, objem[i], skyem[i], x_s2n)
        # and then the S/N for the sum of nexp exposures
        y[i,0] = np.sqrt(float(nexp)) * y[i,0]
        # finally we simulate the sum of nexp exposures
        y[i,1] = (1.0 + np.random.normal() / y[i,0]) * float(nexp) * objem[i]
    return y


if __name__ == '__main__':

    # Configparser input
    config = configparser.ConfigParser()
    config.read('data/config.ini')

    Exp_Time = config['OBSERVATION']['Exposure_time']
    Exp_Time = float(Exp_Time)
    ObjectType = config['OBSERVATION']['Template'][0]
    Parameter = config['OBSERVATION']['Template'][1:]
    Parameter = float(Parameter)
    AB_mag = config['OBSERVATION']['AB_mag']
    AB_mag = float(AB_mag)
    MagLam = config['OBSERVATION']['Wavelength_of_AB_mag']
    MagLam = float(MagLam)
    LambdaMin = config['SPEC']['WL_lower_limit']
    LambdaMin = float(LambdaMin)
    LambdaMax = config['SPEC']['WL_upper_limit']
    LambdaMax = float(LambdaMax)
    SlitWidth = config['SPEC']['Slit_width']
    SlitWidth = float(SlitWidth)
    PSF_FWHM = config['OBSERVATION']['FWHM_seeing']
    PSF_FWHM = float(PSF_FWHM)
    Binning_W = config['SPEC']['Detector_binning_dispersion_direction']
    Binning_W = int(Binning_W)
    Binning_L = config['SPEC']['Detector_binning_spatial_direction']
    Binning_L = int(Binning_L)
    SN_Binning = config['SPEC']['Post_detector_binning']
    SN_Binning = int(SN_Binning)
    N_Exp = config['OBSERVATION']['Nr_of_exposures']
    N_Exp = int(N_Exp)
    # mag_zero_point is ther zero point of the AB magnitude system, when in terms of wavelength and photons.
    mag_zero_point = 26.847
    pessimism_factor = config['SPEC']['pessimism_factor']
    pessimism_factor = float(pessimism_factor)
    global ff_error
    ff_error = config['SPEC']['ff_error']
    ff_error = float(ff_error)
    global tel_area
    tel_area = config['SPEC']['tel_area']
    tel_area = float(tel_area)
    dark_vis = config['VIS_DET']['dark']
    dark_vis = float(dark_vis)
    dark_ir = config['IR_DET']['dark']
    dark_ir = float(dark_ir)
    dark_uvb = config['UV_DET']['dark']
    dark_uvb = float(dark_uvb)
    ron_vis = config['VIS_DET']['ron']
    ron_vis = float(ron_vis)
    ron_uvb = config['UV_DET']['ron']
    ron_uvb = float(ron_uvb)
    ron_ir = config['IR_DET']['ron']
    ron_ir = float(ron_ir)
    pix_length_vis = config['VIS_DET']['pix_length']
    pix_length_vis = float(pix_length_vis)
    pix_length_ir = config['IR_DET']['pix_length']
    pix_length_ir = float(pix_length_ir)
    pix_length_uvb = config['UV_DET']['pix_length']
    pix_length_uvb = float(pix_length_uvb)
    pix_width_vis = config['VIS_DET']['pix_width']
    pix_width_vis = float(pix_width_vis)
    pix_width_ir = config['IR_DET']['pix_width']
    pix_width_ir = float(pix_width_ir)
    pix_width_uvb = config['UV_DET']['pix_width']
    pix_width_uvb = float(pix_width_uvb)

    uvb_arm_min = config['UV_DET']['arm_min']
    uvb_arm_min = float(uvb_arm_min)
    uvb_arm_max = config['UV_DET']['arm_max']
    uvb_arm_max = float(uvb_arm_max)
    vis_arm_min = config['VIS_DET']['arm_min']
    vis_arm_min = float(vis_arm_min)
    vis_arm_max = config['VIS_DET']['arm_max']
    vis_arm_max = float(vis_arm_max)
    ir_arm_min = config['IR_DET']['arm_min']
    ir_arm_min = float(ir_arm_min)
    ir_arm_max = config['IR_DET']['arm_max']
    ir_arm_max = float(ir_arm_max)

    # New parameters
    global s_length
    s_length = config['SPEC']['Slit_length']
    s_length = float(s_length)

    global lambda_split_uvb_vis, lambda_split_vis_ir
    lambda_split_uvb_vis = config['SPEC']['lambda_split_uvb_vis']
    lambda_split_uvb_vis = float(lambda_split_uvb_vis)
    lambda_split_vis_ir = config['SPEC']['lambda_split_vis_ir']
    lambda_split_vis_ir = float(lambda_split_vis_ir)

    global am
    am = config['OBSERVATION']['airmass']
    am = float(am)

    global sky_filename
    global sky_component_filename
    global moon_stage
    moon_stage = config['OBSERVATION']['moon_stage']
    if moon_stage == 'full':
        sky_filename = 'data/skycalc_radiance_full_moon.dat'
        sky_component_filename = 'data/skycalc_radiance_components_full_moon.dat'
    elif moon_stage == 'new':
        sky_filename = 'data/skycalc_radiance_new_moon.dat'
        sky_component_filename = 'data/skycalc_radiance_components_new_moon.dat'
    elif moon_stage == 'half':
        sky_filename = 'data/skycalc_radiance_half_moon.dat'
        sky_component_filename = 'data/skycalc_radiance_components_half_moon.dat'
    elif moon_stage == 'none':
        sky_filename = 'data/skycalc_radiance_no_moon.dat'
    elif moon_stage == 'custom':
        sky_filename = config['OBSERVATION']['custom_sky_file']
        sky_component_filename = config['OBSERVATION']['custom_moon_file']
    else:
        raise ValueError('Invalid moon stage')

    # The binning factors must be greater than or equal to 1
    Binning_W = max(Binning_W, 1)
    Binning_L = max(Binning_L, 1)
    SN_Binning = max(SN_Binning, 1)
    N_Exp = max(N_Exp, 1)

    # The kernel for binning the Signal-to-Noise must have an odd number of elements: (2*N2 + 1)
    N2 = int(SN_Binning/2)
    N_kernel = int(2*N2 + 1)

    # The kernel elements are set to 1.0
    Kernel = np.ones(N_kernel, dtype=float)

    # I assume that wavelengths are given in Angstrom
    if ObjectType == 'T':
        Parameter /= 10000.0
    MagLam /= 10000.0
    LambdaMin /= 10000.0
    LambdaMax /= 10000.0

    # For now, we patch object type
    if ObjectType == 'P':
        ObjectType = 'powerlaw'
    if ObjectType == 'B':
        ObjectType = 'plancklaw'
    if ObjectType == 'T':
        ObjectType = 'template'
        TemplateName = config['OBSERVATION']['Template_file']

    # Read the template spectrum file if needed
    if ObjectType == 'template':
        read_template(AB_mag, MagLam, Parameter, TemplateName)
    else:
        template_x = np.array([1,2,3])
        template_y = np.array([1,2,3])

    # For now we set DiskScale to a fixed low value of 0.05
    DiskScale  = 0.05

    # Now we get the wavelength ranges for the individual arms
    Range_UVB = np.array([uvb_arm_min, uvb_arm_max])
    Range_VIS = np.array([vis_arm_min, vis_arm_max])
    Range_IR = np.array([ir_arm_min, ir_arm_max])

    Cov_UVB	= [max(LambdaMin, Range_UVB[0]), min(LambdaMax, Range_UVB[1])]
    Cov_VIS = [max(LambdaMin, Range_VIS[0]), min(LambdaMax, Range_VIS[1])]
    Cov_IR = [max(LambdaMin, Range_IR[0]), min(LambdaMax, Range_IR[1])]
    
    # Now calculate the S/N for each of the arms

    # UVB arm
    s_n_UVB = 0.0
    sim_UVB = 0.0

    if Cov_UVB[0] < Cov_UVB[1]:
        init_snc_at_vis_ir_and_uvb('uvb', SlitWidth, DiskScale, PSF_FWHM , Binning_W , Binning_L)
        y_spls ('uvb')
        lam_UVB = lamgen_at_vis_ir_and_uvb('uvb', Cov_UVB[0] , Cov_UVB[1])
        snc = snc_at_vis_ir_and_uvb('uvb', N_Exp, Exp_Time, ObjectType, AB_mag, MagLam, Parameter, lam_UVB)
        s_n_UVB = snc[:,0]
        sim_UVB = snc[:,1]
        s_n_sqr = s_n_UVB**2
        s_n_UVB = np.sqrt(np.convolve(s_n_sqr, Kernel, mode='same'))
        # The first and last N2 elements of the convolved array are set to 0.0 by the function convol(), so we must truncate s_n_UVB and lam_uvb by N2 elements at both ends.
        lam_UVB = lam_UVB[N2:len(lam_UVB)-N2]
        s_n_UVB = s_n_UVB[N2:len(s_n_UVB)-N2]
        sim_UVB = sim_UVB[N2:len(sim_UVB)-N2]
        print('Median UVB S/N = ' + str(np.median(s_n_UVB)))
        print('Mean UVB S/N = ' + str(np.mean(s_n_UVB)))
        print('Min UVB S/N = ' + str(np.min(s_n_UVB)))
        print('Max UVB S/N = ' + str(np.max(s_n_UVB)))

    # VIS arm
    s_n_VIS = 0.0
    sim_VIS = 0.0

    if Cov_VIS[0] < Cov_VIS[1]:
        type = 'vis'
    if Cov_VIS[0] < Cov_VIS[1]:
        init_snc_at_vis_ir_and_uvb('vis', SlitWidth, DiskScale, PSF_FWHM , Binning_W , Binning_L)
        y_spls ('vis')
        lam_VIS = lamgen_at_vis_ir_and_uvb('vis', Cov_VIS[0], Cov_VIS[1])
        snc = snc_at_vis_ir_and_uvb('vis', N_Exp,Exp_Time,ObjectType,AB_mag,MagLam,Parameter,lam_VIS)
        s_n_VIS = snc[:,0]
        sim_VIS = snc[:,1]
        s_n_sqr = s_n_VIS**2
        s_n_VIS = np.sqrt(np.convolve(s_n_sqr, Kernel, mode='same'))
        # The first and last N2 elements of the convolved array are set to 0.0 by the function convol(), so we must truncate s_n_VIS and lam_vis by N2 elements at both ends.
        lam_VIS = lam_VIS[N2:len(lam_VIS)-N2]
        s_n_VIS = s_n_VIS[N2:len(s_n_VIS)-N2]
        sim_VIS = sim_VIS[N2:len(sim_VIS)-N2]
        print('Median VIS S/N = ' + str(np.median(s_n_VIS)))
        print('Mean VIS S/N = ' + str(np.mean(s_n_VIS)))
        print('Min VIS S/N = ' + str(np.min(s_n_VIS)))
        print('Max VIS S/N = ' + str(np.max(s_n_VIS)))
    
    # IR arm
    s_n_IR = 0.0
    sim_IR = 0.0

    if Cov_IR[0] < Cov_IR[1]:
        type = 'ir'
    if Cov_IR[0] < Cov_IR[1]:
        init_snc_at_vis_ir_and_uvb('ir', SlitWidth, DiskScale, PSF_FWHM)
        y_spls ('ir')
        lam_IR = lamgen_at_vis_ir_and_uvb('ir', Cov_IR[0], Cov_IR[1])
        snc = snc_at_vis_ir_and_uvb('ir', N_Exp,Exp_Time,ObjectType,AB_mag,MagLam,Parameter,lam_IR)
        s_n_IR = snc[:,0]
        sim_IR = snc[:,1]
        s_n_sqr = s_n_IR**2
        s_n_IR = np.sqrt(np.convolve(s_n_sqr, Kernel, mode='same'))
        # The first and last N2 elements of the convolved array are set to 0.0 by the function convol(), so we must truncate s_n_IR and lam_ir by N2 elements at both ends.
        lam_IR = lam_IR[N2:len(lam_IR)-N2]
        s_n_IR = s_n_IR[N2:len(s_n_IR)-N2]
        sim_IR = sim_IR[N2:len(sim_IR)-N2]
        print('Median IR S/N = ' + str(np.median(s_n_IR)))
        print('Mean IR S/N = ' + str(np.mean(s_n_IR)))
        print('Min IR S/N = ' + str(np.min(s_n_IR)))
        print('Max IR S/N = ' + str(np.max(s_n_IR)))

    # Now to the plotting

    text	= np.zeros(12, dtype='object')
    param	= np.zeros(12, dtype='object')
    text[0] = 'Exposure Time (single exposure):'
    text[1] = 'Number of exposures:'
    text[2] = 'Slit Width:'
    text[3] = 'FWHM of PSF:'
    text[4] = 'Object Type:'
    if ObjectType == 'powerlaw':
        text[5] = 'Spectral Index:'
    elif ObjectType == 'plancklaw':
        text[5] = 'Temperature:'
    elif ObjectType == 'template':
        text[5] = 'FWHM of Photometric Band:'
    text[6]  = 'AB Magnitude:'
    text[7]  = 'at wavelength:'
    text[8]  = 'Detector binning along dispersion:'
    text[9]  = 'Detector binning along slit:'
    text[10]  = 'Signal-to-Noise binning factor:'
    text[11] = 'Moon stage:'

    param[0]  = str('{:.1f}'.format(Exp_Time)) + ' s'
    param[1] = str(np.round(N_Exp)) + ' exposures'
    param[2]  = str('{:.2f}'.format(SlitWidth)) + ' arcsec'
    param[3]  = str('{:.2f}'.format(PSF_FWHM)) + ' arcsec'
    param[4]  = ObjectType
    if ObjectType == 'powerlaw':
        param[5]  = str('{:.2f}'.format(Parameter))
    elif ObjectType == 'plancklaw':
        param[5]  = str('{:.1f}'.format(Parameter)) + ' K'
    elif ObjectType == 'template':
        param[5]  = str('{:.1f}'.format(10000.0 * Parameter)) + ' A'
    param[6]  = str('{:.2f}'.format(AB_mag))
    param[7]  = str('{:.1f}'.format(10000.0 * MagLam)) + ' A'
    param[8]  = str(np.round(Binning_W)) + ' pixels'
    param[9]  = str(np.round(Binning_L)) + ' pixels'
    param[10]  = str(np.round(N_kernel)) + ' binned pixels/channel'
    param[11] = moon_stage

    # plot the Signal-to-Noise

    plt.figure(figsize=(15,10))
    plt.tight_layout()
    plt.title('Signal-to-Noise', size=22)
    plt.xlabel(r'Wavelength  [ $\mu m$ ]', size=18)
    plt.ylabel('S/N  [ per channel ]', size=18)
    plt.xlim(LambdaMin, LambdaMax)
    y_max	= np.max( np.append(s_n_VIS, s_n_IR) )
    plt.ylim(-1.4*y_max , 1.4*y_max)

    if ( Cov_VIS[0] < Cov_VIS[1] ):
        plt.plot(lam_VIS, s_n_VIS, lw=1, color='k') #, psym=10
    if ( Cov_IR[0] < Cov_IR[1] ):
        plt.plot(lam_IR,  s_n_IR, lw=1, color='k') #, psym=10
    if ( Cov_UVB[0] < Cov_UVB[1] ):
        plt.plot(lam_UVB,  s_n_UVB, lw=1, color='k')

    dx	= ( LambdaMax - LambdaMin ) / 15.0
    dy	= y_max / 9.0
    x1	= LambdaMin + dx
    x2	= LambdaMin + 9*dx
    y = -1.4*dy + 0.1
    for i in range(len(text)):
        plt.text(x1, y, text[i], size=14)
        plt.text(x2, y, param[i], size=14)
        y = y - dy

    plt.savefig('Signal-to-Noise.pdf')

    # plot the simulated spectrum

    y_max = np.max( np.append(sim_VIS, sim_IR) )
    y_min = np.min( np.append(sim_VIS, sim_IR) )
    y_ave = 0.5*(y_min + y_max)
    y_hra = 0.5*(y_max - y_min)

    plt.figure(figsize=(15,10))
    plt.tight_layout()
    plt.title('Simulated Spectrum', size=22)
    plt.xlabel(r'Wavelength  [ $\mu m$ ]', size=18)
    plt.ylabel('Counts per Channel', size=18)
    plt.xlim(LambdaMin, LambdaMax)
    plt.ylim(y_ave + -1.2*y_hra , y_ave + 1.2*y_hra)

    if ( Cov_VIS[0] < Cov_VIS[1] ):
        plt.plot(lam_VIS, sim_VIS, lw=1, color='k')
    if ( Cov_IR[0] < Cov_IR[1] ):
        plt.plot(lam_IR,  sim_IR, lw=1, color='k')
    if ( Cov_UVB[0] < Cov_UVB[1] ):
        plt.plot(lam_UVB,  sim_UVB, lw=1, color='k')

    plt.savefig('Simulated-Spectrum.pdf')

    if int(config['OBSERVATION']['csv_output']):
        # Write spectrum and S/N to csv file for all arms if available
        if ( Cov_VIS[0] < Cov_VIS[1] ):
            np.savetxt('sim_VIS.csv', np.transpose([lam_VIS, sim_VIS, s_n_VIS]), delimiter=',', header='Wavelength [um], Flux [counts], S/N')
        if ( Cov_IR[0] < Cov_IR[1] ):
            np.savetxt('sim_IR.csv', np.transpose([lam_IR, sim_IR, s_n_IR]), delimiter=',', header='Wavelength [um], Flux [counts], S/N')
        if ( Cov_UVB[0] < Cov_UVB[1] ):
            np.savetxt('sim_UVB.csv', np.transpose([lam_UVB, sim_UVB, s_n_UVB]), delimiter=',', header='Wavelength [um], Flux [counts], S/N')