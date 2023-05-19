'''
File for creating the final effeciency files for each detector.

At the moment the grating efficiency is set to 0.7. This should change when data for efficiency
 is available.

When it is available, simply replace the grating_eff.csv file with the new data.
Can also change the air_to_glass_eff.csv file if needed.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import scipy.interpolate as interp
import pandas as pd

# import the QE files. These are in nm
em_qe = np.loadtxt('em_qe.dat')
skipper_qe = np.loadtxt('skipper_qe.dat')
h2rg_qe = np.loadtxt('h2rg_qe.dat')
# Change to um
em_qe[:,0] /= 1000
skipper_qe[:,0] /= 1000
h2rg_qe[:,0] /= 1000

# Import coating efficiency. These are in um
# Firstly G01 for aluminum
alum_file = pd.read_excel('Aluminum_Coating_Comparison_Data.xlsx', index_col=0)
alum_file = alum_file.to_numpy()
alum_file = alum_file[2:-2,3:5]
alum_file[:,1] /= 100
alum_eff = alum_file

# Then PO1 for silver
silver_file = pd.read_excel('Silver_Coating_Comparsion_Data.xlsx', index_col=0)
silver_file = silver_file.to_numpy()
silver_file = silver_file[2:,1:3]
# There are many nan values in the silver file. These should be removed
# First instance of nan in the first column
first_nan = np.where(pd.isna(silver_file[:,0]))[0][0]
silver_file = silver_file[:first_nan,:]
silver_file[:,1] /= 100
silver_eff = silver_file

# grating and air_to_glass efficiency will be assumed since we do not have final values for these
# TODO: This is the file you have to change when you get the grating efficiency
grating_file = np.loadtxt('grating_eff.csv', delimiter=',')
# TODO: This is the file you have to change when you get the air to glass efficiency
air_to_glass_file = np.loadtxt('air_to_glass_eff.csv', delimiter=',')

# Now make interpolations for all efficiencies
em_eff_intp = interp.interp1d(em_qe[:,0], em_qe[:,1])
skipper_eff_intp = interp.interp1d(skipper_qe[:,0], skipper_qe[:,1])
h2rg_eff_intp = interp.interp1d(h2rg_qe[:,0], h2rg_qe[:,1])
alum_eff_intp = interp.interp1d(alum_eff[:,0], alum_eff[:,1])
silver_eff_intp = interp.interp1d(silver_eff[:,0], silver_eff[:,1])
grating_eff_intp = interp.interp1d(grating_file[:,0], grating_file[:,1])
air_to_glass_intp = interp.interp1d(air_to_glass_file[:,0], air_to_glass_file[:,1])

# Now get the min and max wavelengths for each detector
em_min_wl = min(em_qe[:,0])
em_max_wl = max(em_qe[:,0])
skipper_min_wl = min(skipper_qe[:,0])
skipper_max_wl = max(skipper_qe[:,0])
h2rg_min_wl = min(h2rg_qe[:,0])
h2rg_max_wl = max(h2rg_qe[:,0])

# Now make the linspaces for each detector
em_wl = np.linspace(em_min_wl, em_max_wl, 10000)
skipper_wl = np.linspace(skipper_min_wl, skipper_max_wl, 10000)
h2rg_wl = np.linspace(h2rg_min_wl, h2rg_max_wl, 10000)

# Now evaluate the interpolations to get the efficiencies
# There are 2 aluminum mirrors (telescope), 3 silver mirrors (spectrograph), 1 grating, and 10 air to glass/glass to air.
em_eff = alum_eff_intp(em_wl)**2 * silver_eff_intp(em_wl)**3 * em_eff_intp(em_wl) * grating_eff_intp(em_wl) * air_to_glass_intp(em_wl)**10
skipper_eff = alum_eff_intp(skipper_wl)**2 * silver_eff_intp(skipper_wl)**3 * skipper_eff_intp(skipper_wl) * grating_eff_intp(em_wl) * air_to_glass_intp(skipper_wl)**10
h2rg_eff = alum_eff_intp(h2rg_wl)**2 * silver_eff_intp(h2rg_wl)**3 * h2rg_eff_intp(h2rg_wl) * grating_eff_intp(em_wl) * air_to_glass_intp(h2rg_wl)**10

# Save the files
np.savetxt('em_eff.csv', np.transpose([em_wl, em_eff]), delimiter=',')
np.savetxt('skipper_eff.csv', np.transpose([skipper_wl, skipper_eff]), delimiter=',')
np.savetxt('h2rg_eff.csv', np.transpose([h2rg_wl, h2rg_eff]), delimiter=',')

# # Now plot the efficiencies
# plt.figure()
# plt.plot(em_wl, em_eff, label='EM')
# plt.plot(skipper_wl, skipper_eff, label='Skipper')
# plt.plot(h2rg_wl, h2rg_eff, label='H2RG')
# plt.legend()
# plt.xlabel('Wavelength (um)')
# plt.ylabel('Efficiency')
# plt.show()
# #plt.savefig('efficiencies.png')