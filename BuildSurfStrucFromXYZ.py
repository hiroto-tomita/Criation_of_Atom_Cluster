import numpy as np
import AtomEditor


VESTA = r"C:\Users\Isama\VESTA-win64\VESTA.exe"
ae = AtomEditor.Ae()

path = ae.get_file(idir=r"\\fsw-q02.naist.jp\ssip-lab\50_member\tomita\Atom_Data")
# Write lattice constance
l_axis = np.array([[3.216290, 0.0, 0.0],
                   [-1.608145, 2.785389, 0.0],
                   [0.0, 0.0, 5.239962]])

l_atom0 = ae.get_xyz(path=path)
l_atom1 = ae.select_emitter_atom(l_atom0)
l_atom2 = ae.create_2D_cluster(l_atom1, l_axis, radius=7)
l_atom3 = ae.plane_symmetric_cluster(l_atom2, yz_plane=None, zx_plane=None)
filename = ae.save_cluster_as_xyz(l_atom2)
ae.view_in_vesta(VESTA, filename)
