'''
module about Atom cluster
クラスター作成に関する全ての関数を網羅
2022/4/12 Hiroto Tomita
'''


import numpy as np
import subprocess
import copy
import math
import tkinter
from tkinter import filedialog


class Ae:
    def __init__(self):
        pass

    # Get the directory of file in Explorer
    def get_file(self, idir=r"C:"):
        path = tkinter.filedialog.askopenfilename(initialdir=idir, title="Open")
        return str(path)

    # Get XYZ file as a list
    def get_xyz(self, path):
        if path[-4:] != ".xyz":
            print(r"This file is not 'XYZ file'")
            exit()
        atom = {}
        atom['e'] = ''
        atom['pos'] = np.zeros(3)
        l_atom = []
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            line = f.readline()
            while line:
                line = f.readline()
                tmp = [x.strip() for x in line.split()]
                if not tmp:
                    break
                atom['e'] = tmp[0]
                atom['pos'][0] = tmp[1]
                atom['pos'][1] = tmp[2]
                atom['pos'][2] = tmp[3]
                l_atom.append(copy.deepcopy(atom))
        return l_atom

    # Get XTL file as a list
    def get_xtl(self, path):
        if path[-4:] != ".xtl":
            print(r"This file is not 'XTL file'")
            exit()
        atom = {}
        atom['e'] = ''
        atom['pos'] = np.zeros(3)
        l_atom = []
        lattice = np.zeros(6)
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = f.readline()
                if line.find('CELL') != -1:
                    line = f.readline()
                    tmp = [x.strip() for x in line.split()]
                    lattice = np.float_(tmp)
                if line.find('NAME') != -1:
                    while line:
                        line = f.readline()
                        if line.find("EOF") != -1:
                            break
                        tmp = [x.strip() for x in line.split()]
                        atom['e'] = tmp[0]
                        atom['pos'][0] = float(tmp[1])
                        atom['pos'][1] = float(tmp[2])
                        atom['pos'][2] = float(tmp[3])
                        l_atom.append(copy.deepcopy(atom))
        return l_atom, lattice

    # fractional(abc) -> cartesian(xyz)
    def convert_xtl_to_xyz(self, l_atom, lattice):
        a, b, c = lattice[0], lattice[1], lattice[2]
        angle_degree = [lattice[3], lattice[4], lattice[5]]
        angle = [i * math.pi / 180. for i in angle_degree]  # degree -> radian
        alpha, beta, gamma = angle[0], angle[1], angle[2]
        l_axis = np.zeros([3, 3])
        l_axis[0][0] = a
        l_axis[1][0] = b * np.cos(gamma)
        l_axis[1][1] = b * np.sin(gamma)
        l_axis[2][0] = c * np.cos(beta)
        l_axis[2][1] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        l_axis[2][2] = c * np.sqrt(
            (np.sin(beta)) ** 2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) ** 2 / np.sin(gamma) ** 2))
        v_cell = np.sqrt(
            1 - np.cos(alpha) ** 2 - np.cos(beta) ** 2 - np.cos(gamma) ** 2 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(
                gamma))
        queue = np.array([[a, b * np.cos(gamma), c * np.cos(beta)],
                          [0., b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
                          [0., 0., c * v_cell / np.sin(gamma)]])
        # atom position in unit cell.(xyz)
        atom = {}
        atom['e'] = ''
        atom['pos'] = np.zeros(3)
        tmp = l_atom
        l_atom = []
        for i in range(len(tmp)):
            atom['e'] = tmp[i]['e']
            atom['pos'] = np.dot(queue, tmp[i]['pos'])
            l_atom.append(copy.deepcopy(atom))
        return l_atom, l_axis

    # 選択した元素を原点(0,0,0)に移動させる
    def select_emitter_atom(self, l_atom):
        for i in range(len(l_atom)):
            print("%4d\t:\t%2s%d" % (i+1, l_atom[i]['e'], i))
        e_num = int(input("Input emitter atom number\t-> ")) - 1
        atom = {}
        atom['e'] = ''
        atom['pos'] = np.zeros(3)
        l_tmp = l_atom
        l_atom = []
        for i in range(len(l_tmp)):
            atom['e'] = l_tmp[i]['e']
            atom['pos'] = l_tmp[i]['pos'] - l_tmp[e_num]['pos']
            if atom['pos'][0] == 0. and atom['pos'][1] == 0. and atom['pos'][2] == 0.:
                l_atom.insert(0, copy.deepcopy(atom))
                continue
            l_atom.append(copy.deepcopy(atom))
        return l_atom

    # 直交座標系から極座標系に変換
    def convert_cartesian_to_polar(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r) * 180 / np.pi
        phi = np.sign(y) * np.arccos(x / np.sqrt(x**2 + y**2)) * 180 / np.pi
        pos = [r, theta, phi]
        print(pos)

    # 極座標系から直交座標系に変換 theta,phiはdegree
    def convert_polar_to_cartesian(self, r, deg_theta, deg_phi):
        theta = deg_theta * np.pi / 180
        phi = deg_phi * np.pi / 180
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        pos = [x, y, z]
        print(pos)

    # a,b方向に並進操作を施して2次元クラスターを作成
    def create_2D_cluster(self, l_unit, l_axis, radius=15.0):
        l_atom = []
        atom = {}
        atom['e'] = ''
        atom['pos'] = np.zeros(3)
        for i0 in range(-6, 7, 1):  # a
            for i1 in range(-6, 7, 1):  # b
                for i3 in range(len(l_unit)):
                    atom['e'] = l_unit[i3]['e']
                    atom['pos'] = np.round(l_unit[i3]['pos'] + l_axis[0] * i0 + l_axis[1] * i1, 6)
                    if np.sqrt(atom['pos'][0] ** 2 + atom['pos'][1] ** 2) > float(radius):
                        continue
                    if atom['pos'][0] == 0.0 and atom['pos'][1] == 0.0 and atom['pos'][2] == 0.0:
                        l_atom.insert(0, copy.deepcopy(atom))
                        continue
                    l_atom.append(copy.deepcopy(atom))
        return l_atom

    # 3次元の原子群を作成
    def create_3D_cluster(self, l_unit, l_axis, radius=15.0):
        print("Select the form of the atom cluster")
        form = int(input("1:Spherical  2:Column  3:Cuboid\t-> "))
        # Create spherical cluster
        if form == 1:
            l_atom = []
            atom = {}
            atom['e'] = ''
            atom['pos'] = np.zeros(3)
            for i0 in range(-7, 8, 1):  # X
                for i1 in range(-7, 8, 1):  # Y
                    for i2 in range(-7, 8, 1):  # Z
                        for i3 in range(len(l_unit)):
                            atom['e'] = l_unit[i3]['e']
                            atom['pos'] = np.round(
                                l_unit[i3]['pos'] + l_axis[0] * i0 + l_axis[1] * i1 + l_axis[2] * i2, 6)
                            if np.sqrt(atom['pos'][0]**2 + atom['pos'][1]**2 + atom['pos'][2]**2) > radius:
                                continue
                            if atom['pos'][0] == 0. and atom['pos'][1] == 0. and atom['pos'][2] == 0.:
                                l_atom.insert(0, copy.deepcopy(atom))
                                continue
                            l_atom.append(copy.deepcopy(atom))
            return l_atom
        # create column cluster
        if form == 2:
            top = input("Upper limit [A]\t-> ")
            bottom = input("Depth [A]\t-> ")
            l_atom = []
            atom = {}
            atom['e'] = ''
            atom['pos'] = np.zeros(3)
            for i0 in range(-7, 8, 1):  # X
                for i1 in range(-7, 8, 1):  # Y
                    for i2 in range(-7, 8, 1):  # Z
                        for i3 in range(len(l_unit)):
                            atom['e'] = l_unit[i3]['e']
                            atom['pos'] = np.round(l_unit[i3]['pos'] + l_axis[0] * i0 + l_axis[1] * i1 + l_axis[2] * i2, 6)
                            if np.sqrt(atom['pos'][0] ** 2 + atom['pos'][1] ** 2) > int(radius) or \
                                    atom['pos'][2] > float(top) or atom['pos'][2] < float(bottom):
                                continue
                            if atom['pos'][0] == 0. and atom['pos'][1] == 0. and atom['pos'][2] == 0.:
                                l_atom.insert(0, copy.deepcopy(atom))
                                continue
                            l_atom.append(copy.deepcopy(atom))
            return l_atom
        # Create cuboid cluster
        if form == 3:
            x_max = float(input('x maximum [A] -> '))
            x_min = float(input('x minimum [A] -> '))
            y_max = float(input('y maximum [A] -> '))
            y_min = float(input('y minimum [A] -> '))
            top = float(input('Upper limit [A] -> '))
            bottom = float(input('Depth [A] -> '))
            l_atom = []
            atom = {}
            atom['e'] = ''
            atom['pos'] = np.zeros(3)
            for i0 in range(-7, 8, 1):  # X
                for i1 in range(-7, 8, 1):  # Y
                    for i2 in range(-7, 8, 1):  # Z
                        for i3 in range(len(l_unit)):
                            atom['e'] = l_unit[i3]['e']
                            atom['pos'] = np.round(l_unit[i3]['pos'] + l_axis[0] * i0 + l_axis[1] * i1 + l_axis[2] * i2, 6)
                            if atom['pos'][0] > x_max or atom['pos'][0] < x_min or atom['pos'][1] > y_max or \
                                    atom['pos'][1] < y_min or atom['pos'][2] > top or atom['pos'][2] < bottom:
                                continue
                            if atom['pos'][0] == 0.0 and atom['pos'][1] == 0.0 and atom['pos'][2] == 0.0:
                                l_atom.insert(0, copy.deepcopy(atom))
                                continue
                            l_atom.append(copy.deepcopy(atom))
            return l_atom

    # オイラー角(ZYZ)でクラスターを回転 引数はdegree
    def rotation_by_euler_angle(self, l_atom, degree_alpha=0., degree_beta=0., degree_gamma=0.):
        alpha = degree_alpha * np.pi / 180
        beta = degree_beta * np.pi / 180
        gamma = degree_gamma * np.pi / 180
        s1, s2, s3 = np.sin(alpha), np.sin(beta), np.sin(gamma)
        c1, c2, c3 = np.cos(alpha), np.cos(beta), np.cos(gamma)
        rot_queue = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                              [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                              [-c3 * s2, s2 * s3, c2]])
        tmp = l_atom
        l_atom = []
        atom = {}
        atom['e'] = ''
        atom['pos'] = np.zeros(3)
        for i in range(len(l_atom)):
            atom['e'] = tmp[i]['e']
            atom['pos'] = np.round(np.dot(rot_queue, tmp[i]['pos']), 6)
            l_atom.append(copy.deepcopy(atom))
        return l_atom

    # 面対象クラスターを作成
    def plane_symmetric_cluster(self, l_atom0, yz_plane=None, zx_plane=None):
        atom = {}
        atom['e'] = ''
        atom['pos'] = np.zeros(3)
        # yz面で対称の場合(クラスター2つ)
        if yz_plane:
            l_atom1 = []
            for i in range(len(l_atom0)):
                atom['e'] = l_atom0[i]['e']
                atom['pos'] = l_atom0[i]['pos'] * np.array([-1., 1., 1.])
                l_atom1.append(copy.deepcopy(atom))
            return l_atom1
        # zx面で対称の場合(クラスター2つ)
        if zx_plane:
            l_atom1 = []
            for i in range(len(l_atom0)):
                atom['e'] = l_atom0[i]['e']
                atom['pos'] = l_atom0[i]['pos'] * np.array([1.0, -1.0, 1.0])
                l_atom1.append(copy.deepcopy(atom))
            return l_atom1
        else:
            return l_atom0

    # クラスターデータをXYZファイル形式でセーブする
    def save_cluster_as_xyz(self, l_atom):
        filename = input("File name to save (No extension needed)\t-> ")
        lines = []
        lines.append(str(len(l_atom)) + '\n')
        lines.append('comment\n')
        for i in range(len(l_atom)):
            lines.append("%2s\t%9.6f\t%9.6f\t%9.6f\n" % (l_atom[i]['e'], l_atom[i]['pos'][0], l_atom[i]['pos'][1], l_atom[i]['pos'][2]))
        # for i in range(len(l_atom)):
        #     lines.append(str(l_atom[i]['e']) + '\t' + str(l_atom[i]['pos'][0])
        #                  + '\t' + str(l_atom[i]['pos'][1]) + '\t' + str(l_atom[i]['pos'][2]) + '\n')
        fp = open(filename + ".xyz", 'w', encoding='utf-8')
        fp.writelines(lines)
        fp.close()
        return filename+".xyz"

    def view_in_vesta(self, vesta_path, filename):
        subprocess.Popen([vesta_path, filename])
