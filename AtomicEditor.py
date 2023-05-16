"""
Module for Atom Cluster
クラスター作成に関する全ての関数を網羅
含まれる全ての長さの単位はオングストローム[A]
2022/11/4  Hiroto Tomita
Version 1.0
"""

# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP
import copy


# atom = {'e': "", 'pos': np.zeros(3)}
# l_atom = [atom0, atom1, atom2,...]


class AE:
    pi = 3.14159265359

    def load_xyz_file(self, file_name="UnitCell.xyz"):
        """
        XYZファイルを読み込む
        :param file_name: File name containing the extension ".xyz"
        :return: Atomic arrangement list[{'e': "Si", 'pos': np.array([0, 0, 0])}]
        """
        # Load XYZ file.
        fp = open(file_name, mode="r", encoding="utf-8")
        nAtom = int(fp.readline().strip())
        fp.readline()
        # Creating list of atomic arrangement.
        atom = {'e': "", 'pos': np.zeros(3)}
        l_atom = []
        line = fp.readline()
        while line:
            line = [x.strip() for x in line.split()]
            atom['e'] = line[0]
            atom['pos'][0] = float(line[1])
            atom['pos'][1] = float(line[2])
            atom['pos'][2] = float(line[3])
            l_atom.append(copy.deepcopy(atom))
            line = fp.readline()
        fp.close()
        if nAtom != len(l_atom):
            print("Check the number of atoms in XYZ file.")
            exit()
        return l_atom

    def load_xtl_file(self, input_file="UnitCell.xyz"):
        """
        XTLファイルを読み込む。
        :param input_file: File name containing the extension ".xtl"
        :return: 原子配列リスト(list)と基本並進ベクトル(2D-ndarray)
        """
        atom = {'e': "", 'pos': np.zeros(3)}
        l_unit_frac = []
        l_lattice = []
        fr = open(input_file, mode="r", encoding="utf-8")
        line = ((fr.readline()).strip()).split()
        # 格子定数の情報の抽出
        while 0 < len(line):
            if "CELL" in line:
                line = ((fr.readline()).strip()).split()
                for i in line:
                    l_lattice.append(float(i))
                break
            else:
                line = ((fr.readline()).strip()).split()
        a = l_lattice[0]
        b = l_lattice[1]
        c = l_lattice[2]
        alpha = l_lattice[3]
        beta = l_lattice[4]
        gamma = l_lattice[5]
        trans_vec = self.translation_vector(a, b, c, alpha, beta, gamma)
        # 原子位置の情報の抽出
        while 0 < len(line):
            if "NAME" in line:
                line = ((fr.readline()).strip()).split()
                while 0 < len(line):
                    if "EOF" in line:
                        break
                    else:
                        atom['e'] = line[0]
                        atom['pos'][0] = float(line[1])
                        atom['pos'][1] = float(line[2])
                        atom['pos'][2] = float(line[3])
                        l_unit_frac.append(copy.deepcopy(atom))
                        line = ((fr.readline()).strip()).split()
            else:
                line = ((fr.readline()).strip()).split()
        fr.close()
        # Convert from fractional to cartesian
        l_unit = self.fractional_to_cartesian(l_unit_frac, a, b, c, alpha, beta, gamma)
        return l_unit, trans_vec

    def check_bond_length(self, l_unit, trans_vector, min_bond_length=0.8):
        """
        単位格子の情報が適切かを判定する
        :param l_unit: 単位格子の原子配列リスト (lise)
        :param trans_vector: 基本並進ベクトル (2D-ndarray)
        :param min_bond_length: 最小結合長 (float)
        :return: True or False
        """
        trans_vector = np.array(trans_vector)
        # ユニットセルを拡張して3x3x3クラスターを形成
        l_atom = []
        atom = {'e': "", 'pos': np.zeros(3)}
        for a in range(-1, 2, 1):
            for b in range(-1, 2, 1):
                for c in range(-1, 2, 1):
                    for n in range(len(l_unit)):
                        atom['e'] = l_unit[n]['e']
                        atom['pos'] = l_unit[n]['pos'] + a*trans_vector[0] + b*trans_vector[1] + c*trans_vector[2]
                        l_atom.append(copy.deepcopy(atom))
        # クラスターに含まれる全原子の距離を計算してリストに格納
        for j in range(len(l_atom)):
            for k in range(j+1, len(l_atom)):
                bond_length = np.linalg.norm(l_atom[j]['pos']-l_atom[k]['pos'], ord=2)
                if bond_length < min_bond_length:
                    print("Bond length == %9.6f" % bond_length)
                    return False
        return True

    def __vec_a(self, a):
        """
        直交座標表示のベクトルa
        :param a: 格子定数a (float)
        :return: ベクトルA (list)
        """
        a_vector = [a, 0.0, 0.0]
        return a_vector

    def __vec_b(self, b, gamma_degree):
        """
        直交座標表示のベクトルb
        :param b: 格子定数b (float)
        :param gamma_degree: 格子定数gamma (float)
        :return: ベクトルB (list)
        """
        gamma = np.radians(gamma_degree)
        b_vector = [b*np.cos(gamma), b*np.sin(gamma), 0.0]
        return b_vector

    def __vec_c(self, c, alpha_degrees, beta_degrees, gamma_degrees):
        """
        直交座標表示のベクトルc
        :param c: 格子定数c (float)
        :param alpha_degrees: 格子定数alpha (float)
        :param beta_degrees: 格子定数beta (float)
        :param gamma_degrees: 格子定数gamma (float)
        :return: ベクトルC (list)
        """
        alpha = np.radians(alpha_degrees)
        beta = np.radians(beta_degrees)
        gamma = np.radians(gamma_degrees)
        c_vector = [c*np.cos(beta),
                    c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma)) / np.sin(gamma),
                    c*np.sqrt(np.sin(beta)**2-((np.cos(alpha)-np.cos(beta)*np.cos(gamma))**2/np.sin(gamma)**2))]
        return c_vector

    def translation_vector(self, a, b, c, deg_alpha, deg_beta, deg_gamma):
        """
        格子定数から基本並進ベクトルを計算
        :param a: 格子定数a (float)
        :param b: 格子定数b (float)
        :param c: 格子定数c (float)
        :param deg_alpha: 格子定数alpha (float)
        :param deg_beta: 格子定数beta (float)
        :param deg_gamma: 格子定数gamma (float)
        :return: 基本並進ベクトル (2D-ndarray)
        """
        trans_vector = [self.__vec_a(a), self.__vec_b(b, deg_gamma), self.__vec_c(c, deg_alpha, deg_beta, deg_gamma)]
        return np.array(trans_vector)

    def lattice_constant(self, trans_vector):
        """
        基本並進ベクトルから格子定数を計算
        :param trans_vector: 基本並進ベクトル (2D-ndarray)
        :return: 格子定数のリスト (list)
        """
        a = np.linalg.norm(trans_vector[0], ord=2)
        b = np.linalg.norm(trans_vector[1], ord=2)
        c = np.linalg.norm(trans_vector[2], ord=2)
        ab_dot = np.dot(trans_vector[0], trans_vector[1])
        bc_dot = np.dot(trans_vector[1], trans_vector[2])
        ca_dot = np.dot(trans_vector[2], trans_vector[0])
        alpha_degrees = np.degrees(np.arccos(bc_dot/(b*c)))
        beta_degrees = np.degrees(np.arccos(ca_dot/(c*a)))
        gamma_degrees = np.degrees(np.arccos(ab_dot / (a*b)))
        return a, b, c, alpha_degrees, beta_degrees, gamma_degrees

    def __conversion_matrix(self, a, b, c, alpha_degrees, beta_degrees, gamma_degrees):
        """
        直交座標上の点から分率座標上の点に変換するための変換行列
        :param a: 格子定数a (float)
        :param b: 格子定数b (float)
        :param c: 格子定数c (float)
        :param alpha_degrees: 格子定数alpha (float)
        :param beta_degrees: 格子定数beta (float)
        :param gamma_degrees: 格子定数gamma (float)
        :return: 変換行列 (2D-ndarray)
        """
        alpha = np.radians(alpha_degrees)
        beta = np.radians(beta_degrees)
        gamma = np.radians(gamma_degrees)
        ax = a
        ay = 0.0
        az = 0.0
        bx = b * np.cos(gamma)
        by = b * np.sin(gamma)
        bz = 0.0
        cx = c * np.cos(beta)
        cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        cz = c * np.sqrt(np.sin(beta) ** 2 - ((np.cos(alpha) - np.cos(beta) * np.cos(gamma)) ** 2 / np.sin(gamma) ** 2))
        vol = ax * by * cz
        matrix = [[1.0/ax, -bx*cz/vol, (bx*cy-by*cx)/vol],
                  [0.0, 1.0/by, -ax*cy/vol],
                  [0.0, 0.0, 1.0/cz]]
        return np.array(matrix)

    def cartesian_to_fractional(self, l_atom, trans_vector):
        """
        原子配列を直交座標表示から分率座標表示に変更
        :param l_atom: 直交座標表示の原子配列リスト (list)
        :param trans_vector: 基本並進ベクトル (2D-ndarray)
        :return: 分率座標表示の原子配列リスト (list)
        """
        a, b, c, deg_alpha, deg_beta, deg_gamma = self.lattice_constant(trans_vector)
        matrix = self.__conversion_matrix(a, b, c, deg_alpha, deg_beta, deg_gamma)
        l_atom_frac = []
        atom = {'e': "", 'pos': np.zeros(3)}
        for i in range(len(l_atom)):
            atom['e'] = l_atom[i]['e']
            atom['pos'] = np.dot(matrix, l_atom[i]['pos'])
            l_atom_frac.append(copy.deepcopy(atom))
        return l_atom_frac

    def fractional_to_cartesian(self, l_atom_frac, a, b, c, alpha_degrees, beta_degrees, gamma_degrees):
        """
        原子配列を分率座標表示から直交座標表示に変更
        :param l_atom_frac: 分率座標表示の原子配列リスト (list)
        :param a: 格子定数a (float)
        :param b: 格子定数b (float)
        :param c: 格子定数c (float)
        :param alpha_degrees: 格子定数alpha (float)
        :param beta_degrees: 格子定数beta (float)
        :param gamma_degrees: 格子定数gamma (float)
        :return: 直交座標表示の原子配列リスト (list)
        """
        inv_matrix = np.linalg.inv(self.__conversion_matrix(a, b, c, alpha_degrees, beta_degrees, gamma_degrees))
        l_atom = []
        atom = {'e': "", 'pos': np.zeros(3)}
        for i in range(len(l_atom_frac)):
            atom['e'] = l_atom_frac[i]['e']
            atom['pos'] = np.dot(inv_matrix, l_atom_frac[i]['pos'])
            l_atom.append(copy.deepcopy(atom))
        return l_atom

    def make_emitter_list(self, l_atom, number_of_emitter=1):
        """
        Make a list described the number of emitter atom index.
        :param l_atom: 原子配列リスト (list)
        :param number_of_emitter: 励起原子の数 (int)
        :return l_emitter: 励起原子のインデックスリスト (list)
        """
        l_emitter = []
        for i, atom in enumerate(l_atom):
            print("%4d\t:\t%2s" % (i + 1, atom['e']))
        print("Input emitter atom number.")
        for j in range(1, number_of_emitter + 1, 1):
            number = int(input(str(j) + " / " + str(number_of_emitter) + "\t:\t")) - 1
            if not 0 <= number <= len(l_atom) - 1:
                print("Atom with that number do not exist.")
                exit()
            l_emitter.append(number)
        return l_emitter

    def move_emitter_atom_to_origin(self, l_atom, emitter_atom_index=0):
        """
        Move the atom of the selected number to the origin.
        :param l_atom: 原子配列リスト <list>
        :param emitter_atom_index: 励起原子のインデックス <int>
        :return: 励起原子を原点とした原子配列リストを返す
        """
        l_atom2 = []
        vector = copy.deepcopy(l_atom[emitter_atom_index]['pos'])
        for atom in l_atom:
            if np.array_equal(atom['pos'], vector):
                atom['pos'] = np.zeros(3)
                l_atom2.insert(0, atom)
                continue
            else:
                atom['pos'] -= vector
                l_atom2.append(atom)
        return l_atom2

    def build_slab_model(self, l_unit, trans_vector, radius=15.0, min_bond_length=0.5):
        """
        The structural model is built by translational operations on a and b axes with respect to unit cell.
        :param min_bond_length:
        :param trans_vector: 基本並進ベクトル (2D-ndarray)
        :param l_unit: 単位格子の原子配列リスト <2D-ndarray>
        :param radius: クラスターの半径 <float>
        :return: スラブモデルの原子配列リスト
        """
        if not self.check_bond_length(l_unit, trans_vector, min_bond_length):
            print("'ERROR': AtomEditor.check_bond_length: Check translation vector and atomic arrangement\n")
            exit()
        l_atom = []
        for a in range(-6, 7, 1):
            for b in range(-6, 7, 1):
                for value in l_unit:
                    atom = copy.deepcopy(value)
                    atom['pos'] += trans_vector[0] * a + trans_vector[1] * b
                    if np.all(atom['pos'] == np.zeros(3)):
                        l_atom.insert(0, copy.deepcopy(atom))
                    elif atom['pos'][0]**2 + atom['pos'][1]**2 < radius**2:
                        l_atom.append(copy.deepcopy(atom))
                    else:
                        continue
        return l_atom

    def build_spherical_cluster(self, l_unit, trans_vec, radius=15.0, min_bond_length=0.5):
        """
        球形のクラスターを構築
        :param min_bond_length:
        :param l_unit: 原子配列リスト (list[dict])
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param radius: クラスターの半径 (float)
        :return: 球形の原子配列リスト (list)
        """
        if not self.check_bond_length(l_unit, trans_vec, min_bond_length):
            print("'ERROR': AtomEditor.check_bond_length: Check translation vector and atomic arrangement\n")
            exit()
        l_atom = []
        for a in range(-7, 8, 1):
            for b in range(-7, 8, 1):
                for c in range(-7, 8, 1):
                    for value in l_unit:
                        atom = copy.deepcopy(value)
                        atom['pos'] += a * trans_vec[0] + b * trans_vec[1] + c * trans_vec[2]
                        if radius < np.linalg.norm(atom['pos'], ord=2):
                            continue
                        elif np.all(atom['pos'] == np.zeros(3)):
                            l_atom.insert(0, copy.deepcopy(atom))
                        else:
                            l_atom.append(copy.deepcopy(atom))
        return l_atom

    def build_column_cluster(self, l_unit, trans_vec, radius=15.0, z_max=5.0, z_min=-5.0, min_bond_length=0.5):
        """
        円柱形のクラスターを構築。
        :param min_bond_length:
        :param l_unit: 原子配列リスト <list>
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param radius: クラスターの半径 <float>
        :param z_max: 原点からの高さ <float>
        :param z_min: 原点からの深さ <float>
        :return: 円柱形の原子配列リスト <list>
        """
        if not self.check_bond_length(l_unit, trans_vec, min_bond_length):
            print("'ERROR': AtomEditor.check_bond_length: Check translation vector and atomic arrangement\n")
            exit()
        l_atom = []
        for a in range(-7, 8, 1):
            for b in range(-7, 8, 1):
                for c in range(-7, 8, 1):
                    for value in l_unit:
                        atom = copy.deepcopy(value)
                        atom['pos'] += trans_vec[0] * a + trans_vec[1] * b + trans_vec[2] * c
                        if not float(z_min) < atom['pos'][2] < float(z_max):
                            continue
                        elif radius ** 2 < atom['pos'][0] ** 2 + atom['pos'][1] ** 2:
                            continue
                        elif np.all(atom['pos'] == np.zeros(3)):
                            l_atom.insert(0, copy.deepcopy(atom))
                        else:
                            l_atom.append(copy.deepcopy(atom))
        return l_atom

    def build_cuboid_cluster(self, l_unit, trans_vec, x_max=5.0, x_min=5.0, y_max=5.0, y_min=5.0, z_max=5.0, z_min=5.0,
                             min_bond_length=0.5):
        """
        矩形のクラスターを構築。
        :param min_bond_length:
        :param l_unit: 原子配列リスト (list)
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param x_max: X軸の最大値 (float)
        :param x_min: X軸の最小値 (float)
        :param y_max: Y軸の最大値 (float)
        :param y_min: Y軸の最小値 (float)
        :param z_max: 原点からの高さ (float)
        :param z_min: 原点からの深さ (float)
        :return: 矩形の原子配列リスト (list)
        """
        if not self.check_bond_length(l_unit, trans_vec, min_bond_length):
            print("'ERROR': AtomEditor.check_bond_length: Check translation vector and atomic arrangement\n")
            exit()
        l_atom = []
        for a in range(-7, 8, 1):
            for b in range(-7, 8, 1):
                for c in range(-7, 8, 1):
                    for value in l_unit:
                        atom = copy.deepcopy(value)
                        atom['pos'] += trans_vec[0] * a + trans_vec[1] * b + trans_vec[2] * c
                        if np.all(atom['pos'] == np.zeros(3)):
                            l_atom.insert(0, copy.deepcopy(atom))
                        elif x_min < atom['pos'][0] < x_max and y_min < atom['pos'][1] < y_max \
                                and z_min < atom['pos'][2] < z_max:
                            l_atom.append(copy.deepcopy(atom))
                        else:
                            continue
        return l_atom

    def rotation_by_euler_angle(self, l_atom, alpha_degrees, beta_degrees, gamma_degrees):
        """
        オイラー角(ZYZ)でクラスターを回転
        :param l_atom: 原子配列リスト (list[dict])
        :param alpha_degrees:
        :param beta_degrees:
        :param gamma_degrees:
        :return:
        """
        pi = 3.14159265359
        alpha = np.radians(alpha_degrees)
        beta = np.radians(beta_degrees)
        gamma = np.radians(gamma_degrees)
        s1 = np.sin(alpha)
        s2 = np.sin(beta)
        s3 = np.sin(gamma)
        c1 = np.cos(alpha)
        c2 = np.cos(beta)
        c3 = np.cos(gamma)
        rot_matrix = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                              [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                              [-c3 * s2, s2 * s3, c2]])
        for i in range(len(l_atom)):
            l_atom[i]['pos'] = np.dot(rot_matrix, l_atom[i]['pos'])
        return l_atom

    def remove_atoms_below_emitter(self, l_atom, z_value):
        """
        渡したzの値以下に位置する原子を取り除く
        :param l_atom: 原子配列リスト (list)
        :param z_value: z軸の値
        :return: 原子配列リスト list[dict{}]
        """
        new_l_atom = []
        for atom in l_atom:
            if z_value <= atom['pos'][2]:
                new_l_atom.append(copy.deepcopy(atom))
            else:
                continue
        return new_l_atom

    def yz_plane_symmetric_cluster(self, l_atom):
        """
        Return cluster with yz-plane symmetry.
        :param l_atom: 原子配列リスト (list)
        :return: yz-plane symmetric Atom list <list>
        """
        l_atom_yz = []
        for i in range(len(l_atom)):
            atom = copy.deepcopy(l_atom[i])
            atom['pos'] *= np.array([-1.0, 1.0, 1.0])
            l_atom_yz.append(copy.deepcopy(atom))
        return l_atom_yz

    def zx_plane_symmetric_cluster(self, l_atom):
        """
        Return cluster with zx-plane symmetry.
        :param l_atom: Atomic arrangement list <list>
        :return: zx-plane symmetric Atom list <list>
        """
        l_atom_zx = []
        for i in range(len(l_atom)):
            atom = copy.deepcopy(l_atom[i])
            atom['pos'] *= np.array([1.0, -1.0, 1.0])
            l_atom_zx.append(copy.deepcopy(atom))
        return l_atom_zx

    def make_xyz_file_text(self, l_atom, comment="Generated by AtomicEditor"):
        """
        Convert an atom list to the list for storage in XYZ file.
        :param l_atom: Atomic arrangement list <list>
        :param comment: comment (str)
        :return: List to save as text <list>
        """
        lines = [str(len(l_atom)) + "\n", comment + "\n"]
        for atom in l_atom:
            line = "%2s" % atom['e']
            line += "%12f" % (float(Decimal(str(atom['pos'][0])).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)))
            line += "%12f" % (float(Decimal(str(atom['pos'][1])).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)))
            line += "%12f\n" % (float(Decimal(str(atom['pos'][2])).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)))
            lines.append(copy.deepcopy(line))
        return lines

    def XYZ_text(self, l_atom, comment="Generated by AtomicEditor"):
        text = str(len(l_atom)) + "\n" + comment + "\n"
        for atom in l_atom:
            text += "%2s" % atom['e']
            text += "%12f" % (float(Decimal(str(atom['pos'][0])).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)))
            text += "%12f" % (float(Decimal(str(atom['pos'][1])).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)))
            text += "%12f\n" % (float(Decimal(str(atom['pos'][2])).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)))
        return text

    def save_text(self, lines, output_file="default.xyz"):
        """
        Save the list in text file.
        :param lines: List to save as text <list>
        :param output_file: File name containing the extension ".xyz"
        :return: File with atomic arrangement
        """
        fw = open(output_file, mode="w", encoding="utf-8")
        fw.writelines(lines)
        fw.close()

    def save_spherical_clusters(self, l_atom, trans_vec, l_emitter_atom_index,
                                output_file, radius=15.0, yz_plane=False, zx_plane=False):
        """
        球形のクラスターファイルを作成して保存する。
        :param l_atom: 単位格子の原子配列リスト (list)
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param l_emitter_atom_index: 励起原子のインデックスリスト (list)
        :param output_file: 作成するクラスターファイルのパスとファイル名 (str)
        :param radius: クラスターの半径 (float)
        :param yz_plane: YZ面対称のクラスターを作るかどうか (True or False)
        :param zx_plane: ZX面対称のクラスターを作るかどうか (True or False)
        :return: 球形のクラスターファイル
        """
        lines = []
        for emt in l_emitter_atom_index:
            l_unit1 = self.move_emitter_atom_to_origin(l_atom, emt)
            cluster1 = self.build_spherical_cluster(l_unit1, trans_vec, radius)
            lines1 = self.make_xyz_file_text(cluster1)
            if yz_plane is not True and zx_plane is not True:
                lines += lines1
            elif yz_plane is True and zx_plane is True:
                print("ERROR: Set one of 'yz_plane' or 'zx_plane' to 'True'.")
                exit()
            elif yz_plane is True:
                cluster2 = self.yz_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
            elif zx_plane is True:
                cluster2 = self.zx_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
        self.save_text(lines, output_file)

    def save_column_clusters(self, l_atom, trans_vec, l_emitter_atom_index,
                             output_file, radius=15.0, z_max=5.0, z_min=-5.0, yz_plane=False, zx_plane=False):
        """
        円柱形のクラスターファイルを作成して保存する。
        :param l_atom: 単位格子の原子配列リスト (list)
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param l_emitter_atom_index: 励起原子のインデックスリスト (list)
        :param output_file: 作成するクラスターファイルのパスとファイル名 (str)
        :param radius: クラスターの半径 <float>
        :param z_max: 原点からの高さ <float>
        :param z_min: 原点からの深さ <float>
        :param yz_plane: YZ面対称のクラスターを作るかどうか <True or False>
        :param zx_plane: ZX面対称のクラスターを作るかどうか <True or False>
        :return: 円柱形のクラスターファイル
        """
        lines = []
        for emt in l_emitter_atom_index:
            l_unit1 = self.move_emitter_atom_to_origin(l_atom, emt)
            cluster1 = self.build_column_cluster(l_unit1, trans_vec, radius, z_max, z_min)
            lines1 = self.make_xyz_file_text(cluster1)
            if yz_plane is not True and zx_plane is not True:
                lines += lines1
            elif yz_plane is True and zx_plane is True:
                print("ERROR: Set one of 'yz_plane' or 'zx_plane' to 'True'.")
                exit()
            elif yz_plane is True:
                cluster2 = self.yz_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
            elif zx_plane is True:
                cluster2 = self.zx_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
        self.save_text(lines, output_file)

    def save_cuboid_clusters(self, l_atom, trans_vec, l_emitter_atom_index,
                             output_file, x_max=5.0, x_min=-5.0, y_max=5.0, y_min=-5.0, z_max=5.0, z_min=-5.0,
                             yz_plane=False, zx_plane=False):
        """
        矩形クラスターファイルを作成して保存する。
        :param l_atom: 単位格子の原子配列リスト (list)
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param l_emitter_atom_index: 励起原子のインデックスリスト (list)
        :param output_file: 作成するクラスターファイルのパスとファイル名 (str)
        :param x_max: X軸の最大値 <float>
        :param x_min: X軸の最小値 <float>
        :param y_max: Y軸の最大値 <float>
        :param y_min: Y軸の最小値 <float>
        :param z_max: 原点からの高さ <float>
        :param z_min: 原点からの深さ <float>
        :param yz_plane: YZ面対称のクラスターを作るかどうか <True or False>
        :param zx_plane: ZX面対称のクラスターを作るかどうか <True or False>
        :return: 矩形のクラスターファイル
        """
        lines = []
        for emt in l_emitter_atom_index:
            l_unit1 = self.move_emitter_atom_to_origin(l_atom, emt)
            cluster1 = self.build_cuboid_cluster(l_unit1, trans_vec, x_max, x_min, y_max, y_min, z_max, z_min)
            lines1 = self.make_xyz_file_text(cluster1)
            if yz_plane is not True and zx_plane is not True:
                lines += lines1
            elif yz_plane is True and zx_plane is True:
                print("ERROR: Set one of 'yz_plane' or 'zx_plane' to 'True'.")
                exit()
            elif yz_plane is True:
                cluster2 = self.yz_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
            elif zx_plane is True:
                cluster2 = self.zx_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
        self.save_text(lines, output_file)

    def save_slab_models(self, l_atom, trans_vec, l_emitter_atom_index,
                         output_file, radius=15.0, yz_plane=False, zx_plane=False):
        """
        スラブモデルのクラスターファイルを作成して保存する。
        :param l_atom: 単位格子の原子配列リスト (list)
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param l_emitter_atom_index: 励起原子のインデックスリスト (list)
        :param output_file: 作成するクラスターファイルのパスとファイル名 (str)
        :param radius: クラスターの半径 (float)
        :param yz_plane: YZ面対称のクラスターを作るかどうか (True or False)
        :param zx_plane: ZX面対称のクラスターを作るかどうか (True or False)
        :return: スラブモデルのクラスターファイル
        """
        lines = []
        for emitter in l_emitter_atom_index:
            l_unit1 = self.move_emitter_atom_to_origin(l_atom, emitter)
            cluster1 = self.build_slab_model(l_unit1, trans_vec, radius)
            lines1 = self.make_xyz_file_text(cluster1)
            if yz_plane is not True and zx_plane is not True:
                lines += lines1
            elif yz_plane is True and zx_plane is True:
                print("ERROR: Set one of 'yz_plane' or 'zx_plane' to 'True'.")
                exit()
            elif yz_plane is True:
                cluster2 = self.yz_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
            elif zx_plane is True:
                cluster2 = self.zx_plane_symmetric_cluster(cluster1)
                lines2 = self.make_xyz_file_text(cluster2)
                lines += (lines1 + lines2)
        self.save_text(lines, output_file)

    def save_unit_cell_as_xtl_file(self, input_filename, trans_vec, output_file):
        """
        直交座標表示の原子配列リストをXTL形式で保存
        :param input_filename:
        :param trans_vec: 基本並進ベクトル (2D-ndarray)
        :param output_file: 保存するファイルの名前 (str)
        :return: XTLファイル
        """
        lines = []
        # 原子配列リストと基本並進ベクトルの読み込み
        l_unit = self.load_xyz_file(input_filename)
        l_unit_frac = self.cartesian_to_fractional(l_unit, trans_vec)
        l_lattice = self.lattice_constant(trans_vec)
        # TITLE
        all_elem = []
        l_elem = [l_unit_frac[0]['e']]
        for i in range(len(l_unit_frac)):
            all_elem.append(copy.deepcopy(l_unit_frac[i]['e']))
            if l_unit_frac[i]['e'] in l_elem:
                continue
            else:
                l_elem.append(copy.deepcopy(l_unit_frac[i]['e']))
        line = "TITLE\t"
        for e in range(len(l_elem)):
            line += (l_elem[e] + str(all_elem.count(l_elem[e])) + " ")
        line += "\n"
        lines.append(copy.deepcopy(line))
        # CELL
        line = "CELL\n"
        for j in range(len(l_lattice)):
            line += ("\t" + str(Decimal(str(l_lattice[j])).quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP)))
        line += "\n"
        lines.append(copy.deepcopy(line))
        # SYMMETRY
        line = "SYMMETRY NUMBER\t1\nSYMMETRY LABEL\tP1\n"
        lines.append(copy.deepcopy(line))
        # ATOMS
        line = "ATOMS\nNAME\tX\tY\tZ\n"
        for k in range(len(l_unit_frac)):
            line += "%2s\t" % l_unit_frac[k]['e']
            line += "%12s\t" % (str(Decimal(str(l_unit_frac[k]['pos'][0])).quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP)))
            line += "%12s\t" % (str(Decimal(str(l_unit_frac[k]['pos'][1])).quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP)))
            line += "%12s\n" % (str(Decimal(str(l_unit_frac[k]['pos'][2])).quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP)))

            # line += str(Decimal(str(l_unit_frac[k]['pos'][0])).quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP))
            # line += "\t"
            # line += str(Decimal(str(l_unit_frac[k]['pos'][1])).quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP))
            # line += "\t"
            # line += str(Decimal(str(l_unit_frac[k]['pos'][2])).quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP))
            # line += "\n"
        line += "EOF\n"
        lines.append(copy.deepcopy(line))
        self.save_text(lines, output_file)

    def show_atomic_arrangement(self, filename):
        fig = plt.figure(dpi=100, figsize=(8, 8))
        ax = fig.add_subplot(projection='3d')
        # 軸の範囲 (min, max)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        l_atom = self.load_xyz_file(filename)

        ax.quiver(-50, 0, 0, 2*50, 0, 0, linestyle='-', arrow_length_ratio=0.1, color='black')

        x, y, z = [], [], []
        for i in range(len(l_atom)):
            x.append(copy.deepcopy(l_atom[i]['pos'][0]))
            y.append(copy.deepcopy(l_atom[i]['pos'][1]))
            z.append(copy.deepcopy(l_atom[i]['pos'][2]))
        ax.scatter(x, y, z, c='red')
        plt.show()
