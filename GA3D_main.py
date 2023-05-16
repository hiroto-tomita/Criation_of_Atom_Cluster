"""
Genetic Algorithmのメインプログラム
api_3d_air_image.py, GeneticAlgorithm.py, AtomicEditor.py を用いる
2022/11/8  Version 2
Hiroto Tomita
Native oxide on GaNの酸化膜原子のZ軸パラメータ最適化に使用
"""

import numpy as np
import subprocess
import csv
import copy
import time
import api_3d_air_image
import GeneticAlgorithm
import AtomicEditor
import LossFunction
from decimal import Decimal, ROUND_HALF_UP
import random


api = api_3d_air_image.Api()
api.connect()
ga = GeneticAlgorithm
ae = AtomicEditor.AE()
lf = LossFunction.LF()


""" /////////////////////////////////////////////////////////////////////////////////////////////////////////////// """

"""結晶構造の定数"""
# 最適化する単位格子のファイル名 (str)
UNIT_CELL = r".\trilayer_bilayer_supercell.xyz"
# 基本並進ベクトル (2D-ndarray)
TRANSLATION_VECTOR = np.array([[3.2162900*2, 0.0000000, 0.0000000],
                               [-1.6081450*2, 2.7853888*2, 0.0000000],
                               [0.0000000, 0.0000000, 40.0000000]])
# 単位格子内の励起原子のインデックスリスト (list[int])
EMITTER_ATOM_INDEX_LIST = [0, 5, 10, 14, 17, 22, 24, 32, 37, 40, 42, 48]
# 単位格子内の最適化する原子のインデックスリスト (list[int])
OPT_ATOM_INDEX_LIST = [2, 7, 12, 13, 14, 21, 22, 23, 24, 25, 32, 33, 36, 37, 40, 41, 42, 43, 48, 49]

"""クラスターの定数 [Angstrom]"""
# 形状(spherical, column, cuboid, slab_model) (str)
CLUSTER_SHAPE = "column"
# 半径  (float)
RADIUS = 15
# X軸  (float)
X_MAX = 5.0
X_MIN = -5.0
# Y軸  (float)
Y_MAX = 5.0
Y_MIN = -5.0
# Z軸  (float)
Z_MAX = 10.0
Z_MIN = -1.0
# 面対称のクラスターを作成 (True or False)
YZ_PLANE_SYMMETRY = False
ZX_PLANE_SYMMETRY = True
# 同じ高さの原子全てに同じ乱数を適用する (True or False)
LAYER_OPT = False
# 作成するクラスターのファイル名 (str)
CLUSTER_NAME = r".\cluster.xyz"

"""シミュレーション"""
# cmd_tmsp.exeのパス  (str)
CMD_TMSP = r"C:\Users\holo\3D-AIR-IMAGE-1100-64bit\plugins\qt5125_vs2015x64\cmd_tmsp.exe"
# シミュレーション用のファイル  (str)
EMITTER_WAVE_FUNCTION = r".\emitter.xml_emt"
PHASE_SHIFT = r".\phaseshift.xml_ph"
FLUCTUATION = r".\fluctuation.xml_flu"
ANGLE_RESOLUTION = "1"
MAPPING_METHOD = "AE"
IMFP = "14.29"
# 散乱波関数のパスとファイル名  (str)
SCATTERED_WAVE_FUNCTION = r"C:\Users\holo\OneDrive\pycharmProjects\GA3D_trilayer_bilayer\scatter.xml_sf"
# 実測ホログラムのパスとファイル名  (str)
EXP_HOLOGRAM = r"C:\Users\holo\OneDrive\pycharmProjects\GA3D_trilayer_bilayer\nGaN1_Ga3d_fit5_peak2.xml_air"
# 出力する計算ホログラムのパスとファイル名 (str)
CALC_PATTERN = r"C:\Users\holo\OneDrive\pycharmProjects\GA3D_trilayer_bilayer\calc_pattern.xml_air"

"""全般"""
# パラメータ初期値からの最大値変位量  (float)
MAX_VARIATION = 0.8
# パラメータの変位ステップ  (float)
VARIATION_STEP = 0.1
# 最小結合長  (float)
MIN_BOND_LENGTH = 1.2
# 最初に生成する個体数  (int)
MAX_GROUP_SIZE = 50
# 最終世代数  (int)
MAX_GENERATION = 1000
# 設定(CSVファイル)  (str)
RECORD = r".\optimization.csv"
HEADER = []

"""評価"""
# 損失関数(MSE, MAE, RMSE) (str)
LOSS_FUNCTION = "RMSE"

"""交叉"""
# 方式(one_point, two_point) (str)
CROSSOVER_METHOD = "two_point"

"""突然変異"""
# 個体の突然変異確率(0.00-1.00) (float)
INDIVIDUAL_MUTATION = 0.02
# 遺伝子の突然変異確率(0.00-1.00) (float)
GENE_MUTATION = 0.20

"""選択"""
# 方式(roulette, tournament, elite) (str)
SELECT_METHOD = "roulette"
# エリート戦略
# エリート個体選択数 <int>
# SELECT_GENE = 5
# ルーレット方式
# 5つのグループの選ばれる確率(合計1.00になるように設定) (float)
P_GROUP_A = 0.40
P_GROUP_B = 0.30
P_GROUP_C = 0.15
P_GROUP_D = 0.10
P_GROUP_E = 0.05

""" /////////////////////////////////////////////////////////////////////////////////////////////////////////////// """


def setting_text():
    lines = []
    # ユニットセルのファイル名
    line = "Unit.Cell\t" + str(UNIT_CELL.split("\\")[-1]) + "\n\n"
    lines.append(copy.deepcopy(line))
    # ユニットセルファイルの内容
    fr = open(UNIT_CELL, 'r', encoding='utf-8')
    line += "Number.of.Atoms\t" + fr.readline() + "\n"
    tmp = "<Unit.Cell\n"
    fr.readline()
    line = (fr.readline().strip()).split()
    num = 1
    while 0 < len(line):
        tmp += ("%3d\t%2s\t%10.6f\t%10.6f\t%10.6f\n" % (num, line[0], float(line[1]), float(line[2]), float(line[3])))
        line = (fr.readline().strip()).split()
        num += 1
    tmp += "UnitCellInfo>\n\n"
    lines.append(copy.deepcopy(tmp))
    # 基本並進ベクトル
    line = "<Primitive.Translation.Vector\n"
    line += "a:\t%10.6f\t%10.6f\t%10.6f\n" % (float(TRANSLATION_VECTOR[0][0]), float(TRANSLATION_VECTOR[0][1]),
                                              float(TRANSLATION_VECTOR[0][2]))
    line += "b:\t%10.6f\t%10.6f\t%10.6f\n" % (float(TRANSLATION_VECTOR[1][0]), float(TRANSLATION_VECTOR[1][1]),
                                              float(TRANSLATION_VECTOR[1][2]))
    line += "c:\t%10.6f\t%10.6f\t%10.6f\n" % (float(TRANSLATION_VECTOR[2][0]), float(TRANSLATION_VECTOR[2][1]),
                                              float(TRANSLATION_VECTOR[2][2]))
    line += "Primitive.Translation.Vector>\n\n"
    lines.append(copy.deepcopy(line))
    # 励起原子
    line = "<Emitter.Atoms\n"
    l_unit = ae.load_xyz_file(UNIT_CELL)
    for i in range(len(EMITTER_ATOM_INDEX_LIST)):
        line += "%2s\t%10.6f\t%10.6f\t%10.6f\n" % (l_unit[EMITTER_ATOM_INDEX_LIST[i]]['e'],
                                                   l_unit[EMITTER_ATOM_INDEX_LIST[i]]['pos'][0],
                                                   l_unit[EMITTER_ATOM_INDEX_LIST[i]]['pos'][1],
                                                   l_unit[EMITTER_ATOM_INDEX_LIST[i]]['pos'][2],)
    line += "Emitter.Atoms>\n\n"
    lines.append(copy.deepcopy(line))
    # 最適化する原子
    line = "<OptAtoms\n"
    l_unit = ae.load_xyz_file(UNIT_CELL)
    for i in range(len(OPT_ATOM_INDEX_LIST)):
        line += "%2s\t%10.6f\t%10.6f\t%10.6f\n" % (l_unit[OPT_ATOM_INDEX_LIST[i]]['e'],
                                                   l_unit[OPT_ATOM_INDEX_LIST[i]]['pos'][0],
                                                   l_unit[OPT_ATOM_INDEX_LIST[i]]['pos'][1],
                                                   l_unit[OPT_ATOM_INDEX_LIST[i]]['pos'][2],)
    line += "OptAtoms>\n\n"
    lines.append(copy.deepcopy(line))
    # クラスター
    line = "<Cluster\nShape:\t"
    if CLUSTER_SHAPE == "spherical":
        line += (CLUSTER_SHAPE + "\nRadius(A):\t" + str(RADIUS))
    elif CLUSTER_SHAPE == "column":
        line += (CLUSTER_SHAPE + "\nRadius(A):\t" + str(RADIUS)
                 + "\nMaxZ(A):\t" + str(Z_MAX) + "\nMinZ(A):\t" + str(Z_MIN))
    elif CLUSTER_SHAPE == "cuboid":
        line += (CLUSTER_SHAPE + "\nMaxX(A):\t" + str(X_MAX) + "\nMinX(A):\t" + str(X_MIN)
                 + "\nMaxY(A):\t" + str(Y_MAX) + "\nMinY(A):\t" + str(Y_MIN)
                 + "\nMaxZ(A):\t" + str(Z_MAX) + "\nMinZ(A):\t" + str(Z_MIN))
    elif CLUSTER_SHAPE == "slab":
        line += (CLUSTER_SHAPE + "\nRadius(A):\t" + str(RADIUS))
    line += "\nFileName:\t" + CLUSTER_NAME + "\nCluster>\n\n"
    lines.append(copy.deepcopy(line))
    # 最適化設定
    line = "<OptSetting\nMaxVariation(A):\t" + str(MAX_VARIATION) + "\nVariationStep(A):\t" + str(VARIATION_STEP)\
           + "\nMinBondLength(A):\t" + str(MIN_BOND_LENGTH) + "\nSelectedGroupSize:\t" + str(MAX_GROUP_SIZE)\
           + "\nMaxGeneration:\t" + str(MAX_GENERATION) + "\nOptSetting>\n\n"
    lines.append(line)
    # 評価
    line = "<Evaluation\nLossFunction:\t" + LOSS_FUNCTION + "\nEvaluation>\n\n"
    lines.append(copy.deepcopy(line))
    # 交叉
    line = "<Crossover\n" + CROSSOVER_METHOD + "\nCrossover>\n\n"
    lines.append(copy.deepcopy(line))
    # 突然変異
    line = "<Mutation\nIndividual(%):\t" + str(INDIVIDUAL_MUTATION * 100)\
           + "\nGene(%):\t" + str(GENE_MUTATION * 100) + "\nMutation>\n\n"
    lines.append(copy.deepcopy(line))
    # 選択
    # line = "<Selection\nSelectMethod:\t" + SELECT_METHOD + "\nSelection>\n\n"
    # lines.append(copy.deepcopy(line))
    fw = open("CalculationSettings.txt", 'w', encoding='utf-8')
    fw.writelines(lines)
    fw.close()


def load_chromosome(chromosome, l_unit, l_opt_atom_index):
    """
    染色体の遺伝子情報を適用した単位格子を生成
    :param chromosome: 染色体
    :param l_unit: 単位格子
    :param l_opt_atom_index: 最適化する原子のインデックスリスト (list[int])
    :return: 染色体の遺伝子情報を適用した単位格子
    """
    another_l_unit = copy.deepcopy(l_unit)
    # 遺伝子情報を単位格子に反映
    for i in range(len(l_opt_atom_index)):
        another_l_unit[l_opt_atom_index[i]]['pos'][0] = chromosome[i * 3]
        another_l_unit[l_opt_atom_index[i]]['pos'][1] = chromosome[i * 3 + 1]
        another_l_unit[l_opt_atom_index[i]]['pos'][2] = chromosome[i * 3 + 2]
    return another_l_unit


def __randomize(gene, max_variation, step):
    """
    遺伝子のパラメータを乱数で変化させる。
    :param gene: 変化させるパラメータ <float>
    :param max_variation: 最大変位量 <float>
    :param step: 変位ステップ <float>
    :return: 変化させたパラメータ <float>
    """
    tmp = Decimal(str(gene)) + Decimal(str(random.choice(list(np.arange(-max_variation, max_variation + step, step)))))
    new_gene = float(tmp.quantize(Decimal('0.0000001'), rounding=ROUND_HALF_UP))
    return new_gene


# def __check_bond_length(chromosome, l_unit, trans_vec, l_opt_atom_index, min_bond_length=1.2):
#     """
#     生成した遺伝子情報を適用したユニットセルの各原子が近すぎないかをチェックする。
#     :param chromosome: 遺伝子配列 (list)
#     :param l_unit: デフォルトの単位格子の原子配列リスト (list)
#     :param trans_vec: 基本並進ベクトル　(2D-array)
#     :param l_opt_atom_index: 最適化する原子のインデックスリスト (list)
#     :param min_bond_length: 最小結合長 (float)
#     :return: 条件を満たしていればTrue, 満たしていなければFalseを返す
#     """
#     default_unit = copy.deepcopy(l_unit)
#     # 遺伝子情報を単位格子に反映
#     for i in range(len(l_opt_atom_index)):
#         default_unit[l_opt_atom_index[i]]['pos'][0] = chromosome[i * 3]
#         default_unit[l_opt_atom_index[i]]['pos'][1] = chromosome[i * 3 + 1]
#         default_unit[l_opt_atom_index[i]]['pos'][2] = chromosome[i * 3 + 2]
#     # ユニットセルを拡張して小さなクラスターを形成
#     l_atom = []
#     atom = {'e': "", 'pos': np.zeros(3)}
#     for a in range(-1, 2, 1):
#         for b in range(-1, 2, 1):
#             for c in range(-1, 2, 1):
#                 for n in range(len(default_unit)):
#                     atom['e'] = default_unit[n]['e']
#                     atom['pos'] = default_unit[n]['pos'] + a * trans_vec[0] + b * trans_vec[1] + c * trans_vec[2]
#                     l_atom.append(copy.deepcopy(atom))
#     # クラスターに含まれる全原子の距離を計算してリストに格納
#     for j in range(len(l_atom)):
#         for k in range(j + 1, len(l_atom)):
#             if ( (l_atom[j]['pos'][0] - l_atom[k]['pos'][0]) ** 2
#                + (l_atom[j]['pos'][1] - l_atom[k]['pos'][1]) ** 2
#                + (l_atom[j]['pos'][2] - l_atom[k]['pos'][2]) ** 2) < min_bond_length ** 2:
#                 return False
#             else:
#                 continue
#     return True


def __check_variation_range(l_unit, chromosome, opt_atom_index_list, max_variation):
    """
    パラメータの変位量が最大変位量以下であるかをチェックする。
    :param l_unit: 単位格子の原子配列リスト <list>
    :param chromosome: パラメータリスト染色体 <list>
    :param opt_atom_index_list: 最適化する原子のインデックスのリスト <list>
    :param max_variation: 最大変位量 <float>
    :return: 条件を満たしていればTrue, 満たしていなければFalseを返す
    """
    # デフォルトのパラメータを単位格子から抽出
    default_gene = []
    for i in range(len(opt_atom_index_list)):
        default_gene.append(copy.deepcopy(l_unit[opt_atom_index_list[i]]['pos'][2]))
    # 読み込んだ遺伝子の値とデフォルト値を比較し、差が最大変位量以下かを判定
    for j in range(len(opt_atom_index_list)):
        variation = chromosome[j] - default_gene[j]
        if variation < -max_variation or max_variation < variation:
            return False
    return True


def generate_individual(l_unit, trans_vec, l_opt_atom_index, max_variation, step, min_bond_length):
    """
    最小結合長が設定値以上になるランダムな遺伝子を持つ個体を生成する。
    :param l_unit: デフォルト単位格子の原子配列 <list>
    :param trans_vec: 並進ベクトル
    :param l_opt_atom_index: 最適化する原子のインデックスリスト <list>
    :param max_variation: パラメータの最大変位量 <float>
    :param step: 変位ステップ <float>
    :param min_bond_length: 最小結合長 <float>
    :return: 生成した遺伝子を格納した個体 <class>
    """
    # デフォルトパラメータを単位格子リストから取得して染色体を生成
    default_chromosome = []
    for i in range(len(l_opt_atom_index)):
        default_chromosome.append(l_unit[l_opt_atom_index[i]]['pos'][0])
        default_chromosome.append(l_unit[l_opt_atom_index[i]]['pos'][1])
        default_chromosome.append(l_unit[l_opt_atom_index[i]]['pos'][2])
    while True:
        chromosome = []
        for j in range(len(default_chromosome)):
            chromosome.append(__randomize(default_chromosome[j], max_variation, step))
        # 作成した個体によるユニットセルの最小結合長が設定値以上かを判定
        l_atom = load_chromosome(chromosome, l_unit, l_opt_atom_index)
        if ae.check_bond_length(l_atom, trans_vec, min_bond_length):
            return ga.Individual(chromosome, 0)
        else:
            continue


def cluster_spherical(individual, l_unit, trans_vec, l_emitter_atom_index, l_opt_atom_index,
                      cluster_filename, radius, yz_plane, zx_plane, layer_opt=False):
    """
    個体の遺伝子を反映した球形のクラスターファイルを作成して保存する
    :param individual: 個体 (class)
    :param l_unit: デフォルト単位格子の原子配列リスト (list)
    :param trans_vec: 基本並進ベクトル (2D-ndarray)
    :param l_emitter_atom_index: 励起原子のインデックスリスト (list)
    :param l_opt_atom_index: 最適化する原子のインデックスリスト (list)
    :param cluster_filename: 生成するクラスターのパスとファイル名 (str)
    :param radius: クラスターの半径 (float)
    :param yz_plane: YZ面対称のクラスターを生成するかどうか (True or False)
    :param zx_plane: ZX面対称のクラスターを生成するかどうか (True or False)
    :param layer_opt: 同じ高さにある原子全てに遺伝子情報を適用する (True or False)
    :return: 球形のクラスターファイル
    """
    l_atom = copy.deepcopy(l_unit)
    # 遺伝子情報を原子配列に反映
    chromosome = individual.get_chromosome()
    for i in range(len(l_opt_atom_index)):
        l_atom[l_opt_atom_index[i]]['pos'][0] = chromosome[i * 3]
        l_atom[l_opt_atom_index[i]]['pos'][1] = chromosome[i * 3 + 1]
        l_atom[l_opt_atom_index[i]]['pos'][2] = chromosome[i * 3 + 2]
        # 同じ高さにある原子にも遺伝子情報を適用する
        if layer_opt:
            for j in range(len(l_atom)):
                if l_unit[l_opt_atom_index[i]]['pos'][2] == l_atom[j]['pos'][2]:
                    l_atom[j]['pos'][2] = chromosome[i]
    # 球形クラスターの作成
    ae.save_spherical_clusters(l_atom, trans_vec, l_emitter_atom_index, cluster_filename, radius, yz_plane, zx_plane)


def cluster_column(individual, l_unit, trans_vec, l_index_emitter, l_index_optAtom,
                   cluster_filename, radius, z_max, z_min, layer_opt=False):
    """
    円柱形のクラスターファイルを作成して保存する。
    :param individual: 個体 (class)
    :param l_unit: 単位格子の原子配列リスト (list)
    :param trans_vec: 基本並進ベクトル (2D-ndarray)
    :param l_index_emitter: 励起原子のインデックスリスト (list)
    :param l_index_optAtom: 最適化する原子のインデックスリスト (list)
    :param cluster_filename: 作成するクラスターファイルのパスとファイル名 (str)
    :param radius: クラスターの半径 (float)
    :param z_max: 原点からの高さ (float)
    :param z_min: 原点からの深さ (float)
    :return: 円柱形のクラスターファイル
    :param layer_opt:
    """
    l_atom = copy.deepcopy(l_unit)
    # 遺伝子情報を原子配列に反映
    chromosome = individual.get_chromosome()
    for i in range(len(l_index_optAtom)):
        l_atom[l_index_optAtom[i]]['pos'][0] = chromosome[i * 3]
        l_atom[l_index_optAtom[i]]['pos'][1] = chromosome[i * 3 + 1]
        l_atom[l_index_optAtom[i]]['pos'][2] = chromosome[i * 3 + 2]
        # 同じ高さにある原子にも遺伝子情報を適用する
        if layer_opt:
            for j in range(len(l_atom)):
                if l_unit[l_index_optAtom[i]]['pos'][2] == l_atom[j]['pos'][2]:
                    l_atom[j]['pos'][2] = chromosome[i]

    lines = []
    for emt in l_index_emitter:
        unitcell = ae.move_emitter_atom_to_origin(l_atom, emt)
        cluster = ae.build_column_cluster(unitcell, trans_vec, radius, z_max, z_min)
        lines += ae.make_xyz_file_text(cluster)
    ae.save_text(lines, cluster_filename)


def cluster_cuboid(individual, l_unit, trans_vec, l_emitter_atom_index, l_opt_atom_index,
                   cluster_filename, x_max, x_min, y_max, y_min, z_max, z_min, yz_plane, zx_plane, layer_opt=False):
    """
    個体の遺伝子を反映した矩形のクラスターファイルを作成して保存する。
    :param individual: 個体 (class)
    :param l_unit: デフォルト単位格子の原子配列リスト (list)
    :param trans_vec: 基本並進ベクトル (2D-ndarray)
    :param l_emitter_atom_index: 励起原子のインデックスリスト (list)
    :param l_opt_atom_index: 最適化する原子のインデックスリスト (list)
    :param cluster_filename: 生成するクラスターのパスとファイル名 (str)
    :param x_max: X軸の最大値 (float)
    :param x_min: X軸の最小値 (float)
    :param y_max: Y軸の最大値 (float)
    :param y_min: Y軸の最小値 (float)
    :param z_max: 原点からの高さ (float)
    :param z_min: 原点からの深さ (float)
    :param yz_plane: YZ面対称のクラスターを生成する (True or False)
    :param zx_plane: ZX面対称のクラスターを生成する (True or False)
    :param layer_opt: 同じ高さにある原子全てに遺伝子情報を適用する (True or False)
    :return: 矩形のクラスターファイル
    """
    l_atom = copy.deepcopy(l_unit)
    # 遺伝子情報を原子配列に反映
    chromosome = individual.get_chromosome()
    for i in range(len(l_opt_atom_index)):
        l_atom[l_opt_atom_index[i]]['pos'][0] = chromosome[i * 3]
        l_atom[l_opt_atom_index[i]]['pos'][1] = chromosome[i * 3 + 1]
        l_atom[l_opt_atom_index[i]]['pos'][2] = chromosome[i * 3 + 2]
        # 同じ高さにある原子にも遺伝子情報を適用する
        if layer_opt:
            for j in range(len(l_atom)):
                if l_unit[l_opt_atom_index[i]]['pos'][2] == l_atom[j]['pos'][2]:
                    l_atom[j]['pos'][2] = chromosome[i]
    # 矩形クラスターの作成
    ae.save_cuboid_clusters(l_atom, trans_vec, l_emitter_atom_index,
                            cluster_filename, x_max, x_min, y_max, y_min, z_max, z_min, yz_plane, zx_plane)


def cluster_slab_model(individual, l_unit, trans_vec, l_emitter_atom_index, l_opt_atom_index,
                       cluster_filename, radius, yz_plane, zx_plane, layer_opt=False):
    """
    個体の遺伝子を反映したスラブモデルのXYZファイルを作成して保存する
    :param individual: 個体 (class)
    :param l_unit: デフォルト単位格子の原子配列リスト (list)
    :param trans_vec: 基本並進ベクトル (2D-ndarray)
    :param l_emitter_atom_index: 励起原子のインデックスリスト (list)
    :param l_opt_atom_index: 最適化する原子のインデックスリスト (list)
    :param cluster_filename: 生成するクラスターのパスとファイル名 (str)
    :param radius: クラスターの半径 (float)
    :param yz_plane: YZ面対称のクラスターを生成するかどうか (True or False)
    :param zx_plane: ZX面対称のクラスターを生成するかどうか (True or False)
    :param layer_opt: 同じ高さにある原子全てに遺伝子情報を適用する (True or False)
    :return: スラブモデルのXYZファイル
    """
    l_atom = copy.deepcopy(l_unit)
    # 遺伝子情報を原子配列に反映
    chromosome = individual.get_chromosome()
    for i in range(len(l_opt_atom_index)):
        l_atom[l_opt_atom_index[i]]['pos'][0] = chromosome[i * 3]
        l_atom[l_opt_atom_index[i]]['pos'][1] = chromosome[i * 3 + 1]
        l_atom[l_opt_atom_index[i]]['pos'][2] = chromosome[i * 3 + 2]
        # 同じ高さにある原子にも遺伝子情報を適用する
        if layer_opt:
            for j in range(len(l_atom)):
                if l_unit[l_opt_atom_index[i]]['pos'][2] == l_atom[j]['pos'][2]:
                    l_atom[j]['pos'][2] = chromosome[i]
    # スラブモデルの作成
    ae.save_slab_models(l_atom, trans_vec, l_emitter_atom_index, cluster_filename, radius, yz_plane, zx_plane)


def selection_elite(population, elite_length):
    """
    選択関数。エリート戦略で選択を行う。
    :param population: 選択を行う個体集団 <list>
    :param elite_length: 選択するエリート個体の数 <int>
    :return: エリート個体集団 <list>
    """
    # 現行世代個体集団を誤差の小さい順にソート
    # sorted(): ソートした新たなリストを生成。デフォルトは昇順。reverse=Trueとすると、降順になる。
    # keyにlambdaを渡すと、各リストの絶対値の最大値を基準にソートする。
    sort_result = sorted(population, key=lambda u: u.evaluation)
    # 評価上位の遺伝子を抽出する
    result = []
    for i in range(elite_length):
        result.append(sort_result.pop(0))
    return result


def selection_roulette(population, selected_population_size, pa, pb, pc, pd, pe):
    """
    選択関数。ルーレット方式で選択を行う。
    :param population: 個体集団 <list>
    :param selected_population_size: 選択後の個体集団の大きさ <int>
    :param pa: Aグループが選ばれる確率 <float>
    :param pb: Bグループが選ばれる確率 <float>
    :param pc: Cグループが選ばれる確率 <float>
    :param pd: Dグループが選ばれる確率 <float>
    :param pe: Eグループが選ばれる確率 <float>
    :return: 選択された個体集団 <list>
    """
    # 個体集団の大きさ
    group_length, tmp = divmod(len(population), 5)
    # 適応度の高い(残差の小さい)順にソート
    l_sort = sorted(population, reverse=False, key=lambda u: u.evaluation)
    # 適応度上位から20%ずつの5つのグループに分割
    group = []
    for i in range(0, 5, 1):
        group.append(l_sort[(i*group_length):((i+1)*group_length)])
    if tmp != 0:
        group[4].extend(l_sort[-tmp:])
    # 各グループから選択される確率
    a = Decimal(str(pa))
    b = Decimal(str(pb))
    c = Decimal(str(pc))
    d = Decimal(str(pd))
    e = Decimal(str(pe))
    if a + b + c + d + e != Decimal('1.00'):
        print("'Selection Error'")
        print("Make the sum of the probabilities of each group being selected 1.00")
        print(exit)
    # 各グループから選択された個体集団の生成
    selected_group = []
    while len(selected_group) < selected_population_size:
        rnd = Decimal(str(random.random()))
        # Aグループ
        if Decimal('0.00') <= rnd < a:
            # Aグループに個体がない場合はもう一度ルーレットを回す
            if len(group[0]) == 0:
                continue
            # Aグループが選ばれた場合は適応度の高い順にselected_groupに追加される
            selected_group.append(group[0].pop(0))
        # Bグループ
        elif a <= rnd < (a + b):
            if len(group[1]) == 0:
                continue
            selected_group.append(group[1].pop(0))
        # Cグループ
        elif (a + b) <= rnd < (a + b + c):
            if len(group[2]) == 0:
                continue
            selected_group.append(group[2].pop(0))
        # Dグループ
        elif (a + b + c) <= rnd < (a + b + c + d):
            if len(group[3]) == 0:
                continue
            selected_group.append(group[3].pop(0))
        # Eグループ
        else:
            if len(group[4]) == 0:
                continue
            selected_group.append(group[4].pop(0))
    return selected_group


def crossover_original(parent_first, parent_second):
    """
    交叉関数。二つのベクトルの中間と、a+(a-b)/2の二つの子孫を生成する。
    :param parent_first: 一つ目の個体(親) <class>
    :param parent_second: 二つ目の個体(親) <class>
    :return: 二つの子孫個体 <list>
    """
    # 子孫個体を格納するリストを作成
    l_children = []
    # 遺伝子を取り出す
    parent_first_chromosome = np.array(parent_first.get_chromosome())
    parent_second_chromosome = np.array(parent_second.get_chromosome())
    # 交叉を行う
    child_first = (parent_first_chromosome + parent_second_chromosome) / 2
    child_second = parent_first_chromosome + ((parent_first_chromosome - parent_second_chromosome) / 2)
    # GA classインスタンスを生成して子孫をリストに格納
    l_children.append(ga.Individual(child_first.tolist(), 0))
    l_children.append(ga.Individual(child_second.tolist(), 0))
    return l_children


def crossover_one_point(parent_first, parent_second):
    """
    交叉関数。一点交叉。
    :param parent_first: 一つ目の個体(親) <class>
    :param parent_second: 二つ目の個体(親) <class>
    :return: 二つの子孫個体 <list>
    """
    # 子孫を格納するリスト
    l_children = []
    # 染色体の抽出
    chromosome_first = parent_first.get_chromosome()
    chromosome_second = parent_second.get_chromosome()
    chromosome_length = len(chromosome_first)
    # 交叉
    rd = random.randint(1, chromosome_length - 1)
    child_first = chromosome_first[:rd] + chromosome_second[rd:]
    child_second = chromosome_second[:rd] + chromosome_first[rd:]
    l_children.append(ga.Individual(child_first, 0))
    l_children.append(ga.Individual(child_second, 0))
    return l_children


def crossover_two_point(parent_first, parent_second):
    """
    交叉関数。特殊二点交叉。乱数で選ばれた二つのパラメータを入れ替える。
    :param parent_first: 一つ目の個体(親) <class>
    :param parent_second: 二つ目の個体(親) <class>
    :return: 二つの子孫個体 <list>
    """
    # 子孫を格納するリスト
    l_children = []
    # 染色体の抽出
    chromosome_first = parent_first.get_chromosome()
    chromosome_second = parent_second.get_chromosome()
    chromosome_length = len(chromosome_first)
    # 交叉
    rd1 = random.randint(0, chromosome_length - 1)
    rd2 = random.randint(0, chromosome_length - 1)
    if rd1 == rd2:
        while rd1 == rd2:
            rd2 = random.randint(0, chromosome_length - 1)
    chromosome_third = copy.deepcopy(chromosome_first)
    chromosome_third[rd1], chromosome_third[rd2] = chromosome_second[rd1], chromosome_second[rd2]
    chromosome_fourth = copy.deepcopy(chromosome_second)
    chromosome_fourth[rd1], chromosome_fourth[rd2] = chromosome_first[rd1], chromosome_first[rd2]
    # 子孫個体を生成
    l_children.append(ga.Individual(chromosome_third, 0))
    l_children.append(ga.Individual(chromosome_fourth, 0))
    return l_children


def mutation(population, individual_mutation, gene_mutation, max_variation, step, l_unit, trans_vector,
             l_opt_atom_index, min_bond_length):
    """
    突然変異関数
    :param population: 突然変異させる個体集団 <list>
    :param individual_mutation: 個体に対する突然変異確率 <float>
    :param gene_mutation: 遺伝子一つ一つに対する突然変異確率 <float>
    :param max_variation: 最大変位量 <float>
    :param step: 変位ステップ <float>
    :param l_unit: デフォルト単位格子の原子配列リスト <list>
    :param trans_vector: 並進ベクトル
    :param l_opt_atom_index: 最適化する原子のインデックスリスト <list>
    :param min_bond_length: 最小結合長 <float>
    :return: 突然変異処理を施した個体集団 <list>
    """
    default_chromosome = []
    for k in range(len(l_opt_atom_index)):
        default_chromosome.append(l_unit[l_opt_atom_index[k]]['pos'][0])
        default_chromosome.append(l_unit[l_opt_atom_index[k]]['pos'][1])
        default_chromosome.append(l_unit[l_opt_atom_index[k]]['pos'][2])
    new_population = []
    for individual in population:
        # 個体に対して一定の確率で突然変異を起こす
        if random.random() < individual_mutation:
            chromosome = individual.get_chromosome()
            while True:
                new_chromosome = []
                for j in range(len(chromosome)):
                    # 個体の遺伝子一つ一つに対して一定の確率で突然変異を起こす
                    if random.random() < gene_mutation:
                        new_chromosome.append(__randomize(default_chromosome[j], max_variation, step))
                    else:
                        new_chromosome.append(chromosome[j])
                # 結合長を確認
                l_atom = load_chromosome(new_chromosome, l_unit, l_opt_atom_index)
                if ae.check_bond_length(l_atom, trans_vector, min_bond_length):
                    break
            # 突然変異が適用された場合は適応度を0にして再度評価する
            if chromosome != new_chromosome:
                individual.set_evaluation(0)
            individual.set_chromosome(new_chromosome)
            new_population.append(copy.deepcopy(individual))
        else:
            new_population.append(copy.deepcopy(individual))
    return new_population


def image_get_as_ndarray(image_file):
    """
    XML_AIRファイルのホログラム画像を二次元配列として読み込む
    :param image_file: ホログラム画像のパスとファイル名 (str)
    :return: ホログラム画像データ (2D-array)
    """
    api.image_load(image_file)
    api.update()
    return api.image_get(api.window_get_list_wid_and_caption()[-1]['wid'])


def image_symmetrize_ZX_plane(image_wid):
    """
    Generate ZX plane symmetrized image
    :param image_wid: window id
    :return: symmetrized image (2D-ndarray)
    """
    image = api.image_get(image_wid)
    mirror_image = []
    for i in range(len(image)):
        mirror_image.append(image[-i-1])
    return image + np.array(mirror_image)


def get_image_data_from_3DAI(image_file):
    """
    XML_AIRファイルのホログラム画像を二次元配列として読み込む
    :param image_file: ホログラム画像のパスとファイル名 (str)
    :return: ホログラム画像データ (2D-array)
    """
    api.image_load(image_file)
    api.update()
    return api.image_get(api.window_get_list_wid_and_caption()[-1]['wid'])


def blank_out_window_on_3DAI():
    """
    Close all windows on 3D AIR IMAGE.
    :return:
    """
    api.update()
    l_window = api.window_get_list_wid_and_caption()
    blank_img = np.zeros([181, 181])
    if l_window:
        for i in range(len(l_window)):
            api.window_is_saved_set(l_window[i]['wid'], isSaved=True)
            api.image_set(l_window[i]['wid'], blank_img)
            api.window_close(l_window[i]['wid'])


def test():
    # 最適化を行う単位格子の読み込み
    l_unit = ae.load_xyz_file(input_filename=UNIT_CELL)
    # 全データを格納するリスト
    l_record = []

    # 最初の個体集団を生成
    print("Generating first generation population...\n")
    current_generation_population = []
    for i in range(MAX_GROUP_SIZE):
        current_generation_population.append(generate_individual(l_unit, TRANSLATION_VECTOR, OPT_ATOM_INDEX_LIST,
                                                                 MAX_VARIATION, VARIATION_STEP, MIN_BOND_LENGTH))
    # メインループ
    for count_ in range(1, MAX_GENERATION + 1):
        print("Loop " + str(count_) + " calculation starts...\n")
        # 交叉(Crossover)
        children_population = []
        for i in range(0, MAX_GROUP_SIZE, 2):
            children_population.extend(crossover_two_point(current_generation_population[i - 1],
                                                           current_generation_population[i]))
        # 子孫が結合長の条件を満たしているか判定
        for child in children_population:
            l_atom = load_chromosome(child.get_chromosome(), l_unit, OPT_ATOM_INDEX_LIST)
            if ae.check_bond_length(l_atom, TRANSLATION_VECTOR, MIN_BOND_LENGTH):
                current_generation_population.append(child)
            else:
                current_generation_population.append(generate_individual(l_unit, TRANSLATION_VECTOR,
                                                                         OPT_ATOM_INDEX_LIST, MAX_VARIATION,
                                                                         VARIATION_STEP, MIN_BOND_LENGTH))

        # 突然変異(Mutation)
        current_generation_population = mutation(current_generation_population, INDIVIDUAL_MUTATION, GENE_MUTATION,
                                                 MAX_VARIATION, VARIATION_STEP, l_unit, TRANSLATION_VECTOR,
                                                 OPT_ATOM_INDEX_LIST, MIN_BOND_LENGTH)
        # 評価(Evaluation)
        for i in range(len(current_generation_population)):
            sim_time = 0
            # 全てのデータをCSVファイルに記録
            line = [count_, current_generation_population[i].get_evaluation()]
            line.extend(current_generation_population[i].get_chromosome())
            line.append(sim_time)
            l_record.append(copy.deepcopy(line))
            fr = open(RECORD, 'w', encoding='utf-8', newline="")
            writer = csv.writer(fr)
            writer.writerows(l_record)
            fr.close()

        # 一世代の進化的計算終了

        # 淘汰(Selection)
        next_generation_population = []
        # ルーレット選択方式
        if SELECT_METHOD == "roulette":
            next_generation_population.extend(selection_roulette(current_generation_population, MAX_GROUP_SIZE,
                                                                 P_GROUP_A, P_GROUP_B, P_GROUP_C, P_GROUP_D, P_GROUP_E))
        # トーナメント方式(未完成)
        elif SELECT_METHOD == "tournament":
            print("selection error")
            exit()

        # 次世代に移行
        current_generation_population = next_generation_population


def main():
    # 設定値の記録
    setting_text()
    # 最適化を行う単位格子の読み込み
    l_unit = ae.load_xyz_file(file_name=UNIT_CELL)
    # 実験ホログラムのデータを二次元配列として読み込み
    exp_img = get_image_data_from_3DAI(EXP_HOLOGRAM)
    blank_out_window_on_3DAI()
    print("Experimental image loaded\n")
    # 全データを記録するリスト
    record = []
    header = ["Generation", LOSS_FUNCTION]
    for i in OPT_ATOM_INDEX_LIST:
        header.extend(["atom" + str(i) + "_" + l_unit[i]['e'] + "_x", "atom" + str(i) + "_" + l_unit[i]['e'] + "_y",
                       "atom" + str(i) + "_" + l_unit[i]['e'] + "_z"])
    header.extend(["Calc. time (sec)"])
    record.append(header)

    # 最初の個体集団を生成
    print("Generating first generation population...\n")
    current_generation_population = []
    for i in range(MAX_GROUP_SIZE):
        current_generation_population.append(generate_individual(l_unit, TRANSLATION_VECTOR, OPT_ATOM_INDEX_LIST,
                                                                 MAX_VARIATION, VARIATION_STEP, MIN_BOND_LENGTH))
    # メインループ
    for count_ in range(1, MAX_GENERATION + 1):
        print("Loop " + str(count_) + " calculation starts...\n")
        # 交叉(Crossover)
        children_population = []
        for i in range(0, MAX_GROUP_SIZE, 2):
            children_population.extend(crossover_two_point(current_generation_population[i - 1],
                                                           current_generation_population[i]))
        # 子孫が結合長の条件を満たしているか判定
        for child in children_population:
            l_atom = load_chromosome(child.get_chromosome(), l_unit, OPT_ATOM_INDEX_LIST)
            if ae.check_bond_length(l_atom, TRANSLATION_VECTOR, MIN_BOND_LENGTH):
                current_generation_population.append(child)
            else:
                current_generation_population.append(generate_individual(l_unit, TRANSLATION_VECTOR,
                                                                         OPT_ATOM_INDEX_LIST, MAX_VARIATION,
                                                                         VARIATION_STEP, MIN_BOND_LENGTH))

        # 突然変異(Mutation)
        current_generation_population = mutation(current_generation_population, INDIVIDUAL_MUTATION, GENE_MUTATION,
                                                 MAX_VARIATION, VARIATION_STEP, l_unit, TRANSLATION_VECTOR,
                                                 OPT_ATOM_INDEX_LIST, MIN_BOND_LENGTH)
        # 評価(Evaluation)
        for i in range(len(current_generation_population)):
            calc_time = 0
            # 評価されていない個体(evaluation=0)に評価を行う
            if current_generation_population[i].get_evaluation() == 0:
                # 個体の遺伝子情報を適用した単位格子からクラスターを作成
                if CLUSTER_SHAPE == "spherical":
                    cluster_spherical(current_generation_population[i], l_unit, TRANSLATION_VECTOR,
                                      EMITTER_ATOM_INDEX_LIST, OPT_ATOM_INDEX_LIST, CLUSTER_NAME, RADIUS,
                                      YZ_PLANE_SYMMETRY, ZX_PLANE_SYMMETRY, layer_opt=LAYER_OPT)
                elif CLUSTER_SHAPE == "column":
                    cluster_column(current_generation_population[i], l_unit, TRANSLATION_VECTOR,
                                   EMITTER_ATOM_INDEX_LIST, OPT_ATOM_INDEX_LIST, CLUSTER_NAME,
                                   RADIUS, Z_MAX, Z_MIN, layer_opt=LAYER_OPT)
                elif CLUSTER_SHAPE == "cuboid":
                    cluster_cuboid(current_generation_population[i], l_unit, TRANSLATION_VECTOR,
                                   EMITTER_ATOM_INDEX_LIST, OPT_ATOM_INDEX_LIST, CLUSTER_NAME,
                                   X_MAX, X_MIN, Y_MAX, Y_MIN, Z_MAX, Z_MIN, YZ_PLANE_SYMMETRY, ZX_PLANE_SYMMETRY,
                                   layer_opt=LAYER_OPT)
                elif CLUSTER_SHAPE == "slab_model":
                    cluster_slab_model(current_generation_population[i], l_unit, TRANSLATION_VECTOR,
                                       EMITTER_ATOM_INDEX_LIST, OPT_ATOM_INDEX_LIST, CLUSTER_NAME, RADIUS,
                                       YZ_PLANE_SYMMETRY, ZX_PLANE_SYMMETRY, layer_opt=LAYER_OPT)
                else:
                    print("'ERROR': Input CLUSTER_SHAPE does not exist.")
                    exit()
                # シミュレーションホログラムの作成
                start_time = time.time()
                subprocess.run(CMD_TMSP
                               + " -C " + CLUSTER_NAME
                               + " -F " + EMITTER_WAVE_FUNCTION
                               + " -P " + PHASE_SHIFT
                               + " -W " + SCATTERED_WAVE_FUNCTION
                               + " -T " + FLUCTUATION
                               + " -R " + ANGLE_RESOLUTION + " -M " + MAPPING_METHOD + " -MFPES " + IMFP
                               + " -I " + CALC_PATTERN)
                # 1回のシミュレーション時間を計算
                calc_time = round(time.time() - start_time)
                # 計算ホログラムを取得
                # 計算ホログラムを対称操作
                if ZX_PLANE_SYMMETRY:
                    api.image_new_window()
                    api.image_set(api.window_get_list_wid_and_caption()[-1]['wid'], image_get_as_ndarray(CALC_PATTERN))
                    api.update()
                    calc_image = image_symmetrize_ZX_plane(api.window_get_list_wid_and_caption()[-1]['wid'])
                else:
                    calc_image = image_get_as_ndarray(CALC_PATTERN)
                blank_out_window_on_3DAI()
                # 二つのホログラム画像の差分を計算し結果をGA Classに保存
                if LOSS_FUNCTION == "MSE":
                    current_generation_population[i].set_evaluation(lf.mean_square_error(exp_img, calc_image))
                elif LOSS_FUNCTION == "MAE":
                    current_generation_population[i].set_evaluation(lf.mean_absolute_error(exp_img, calc_image))
                elif LOSS_FUNCTION == "RMSE":
                    current_generation_population[i].set_evaluation(lf.root_mean_square_error(exp_img, calc_image))
                else:
                    print("ERROR: Set LOSS_FUNCTION to 'MAE', 'MSE', or 'RMSE'.")
            # 全てのデータをCSVファイルに記録
            line = [count_, current_generation_population[i].get_evaluation()]
            line.extend(current_generation_population[i].get_chromosome())
            line.append(calc_time)
            record.append(copy.deepcopy(line))
            fr = open(RECORD, 'w', encoding='utf-8', newline="")
            writer = csv.writer(fr)
            writer.writerows(record)
            fr.close()

        # 一世代の進化的計算終了

        # 淘汰(Selection)
        next_generation_population = []
        # ルーレット選択方式
        if SELECT_METHOD == "roulette":
            next_generation_population.extend(selection_roulette(current_generation_population, MAX_GROUP_SIZE,
                                                                 P_GROUP_A, P_GROUP_B, P_GROUP_C, P_GROUP_D, P_GROUP_E))
        # トーナメント方式(未完成)
        elif SELECT_METHOD == "tournament":
            print("selection error")
            exit()

        # 次世代に移行
        current_generation_population = next_generation_population


if __name__ == '__main__':
    main()
