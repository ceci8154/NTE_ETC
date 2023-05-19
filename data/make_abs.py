'''
This file is for making the absorption from the transmission components from skycalc.
'''

import numpy as np

data = np.loadtxt('skycalc_all_transmission_components.dat')

wl = data[:,0]
mol_abs = data[:,1]

new_data = np.asarray([wl, mol_abs]).T

np.savetxt('skycalc_abs.dat', new_data, delimiter=' ')