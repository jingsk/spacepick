from ase.io import read, write
from ase.visualize.plot import plot_atoms
import numpy as np
import matplotlib.pyplot as plt
from spacepick import Selector

atoms = read('MoS2_ortho.vasp') * [2,4,1]
atoms_S_mapping = np.array([atom.index for atom in atoms if atom.scaled_position[2]>0.5])
atoms_S = atoms[atoms_S_mapping]

vec_dis_mat = atoms_S.get_all_distances(mic=True, vector=True)
m = 9
selector = Selector(vec_dis_mat)

fig, axs = plt.subplots(3,3, figsize = [3*3,2.25*3], layout='constrained')

for i in range(m):
    atoms_copy = atoms.copy()
    to_del = selector.get_indices(n_subset=i+1, mode='dispersed')
    del atoms_copy[atoms_S_mapping[to_del]]
    plot_atoms(atoms_copy, ax=axs.ravel()[i])
fig.savefig('dispersed.png', dpi=300)


fig, axs = plt.subplots(3,3, figsize = [3*3,2.25*3], layout='constrained')
for i in range(m):
    atoms_copy = atoms.copy()
    to_del = selector.get_indices(n_subset=i+1, mode='clustered')
    del atoms_copy[atoms_S_mapping[to_del]]
    plot_atoms(atoms_copy, ax=axs.ravel()[i])
fig.savefig('clustered.png', dpi=300)

fig, axs = plt.subplots(3,3, figsize = [3*3,2.25*3], layout='constrained')
for i in range(m):
    atoms_copy = atoms.copy()
    to_del = selector.get_indices(n_subset=i+1, mode='linear', direction_pref='yx', tol=[1e-3, 10])
    del atoms_copy[atoms_S_mapping[to_del]]
    plot_atoms(atoms_copy, ax=axs.ravel()[i])
fig.savefig('linear.png', dpi=300)

