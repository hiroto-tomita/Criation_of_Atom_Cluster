"""
Genetic Algorithmのメインプログラム
2023/2/19  Version 4
Hiroto Tomita
Native oxide on GaNの酸化膜原子のZ軸パラメータ最適化に使用
"""
import sys
import numpy as np
import subprocess
import csv
import copy
import time

sys.path.append(r"C:\Users\holo\OneDrive\pycharmProjects\general")
import api_3d_air_image
import GeneticAlgorithm
import AtomicEditor
import Image2D


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
WORK_DIRECTORY = r"C:\Users\holo\OneDrive\pycharmProjects\GA1D_2ML_HD_OS"
"""結晶構造の定数"""
# 最適化する単位格子のファイル名  str
UNIT_CELL_FILE = WORK_DIRECTORY+r"\trilayer_bilayer_supercell.xyz"
# 基本並進ベクトル  2D-ndarray
TRANSLATION_VECTOR = np.array([[9.564000, 0.0, 0.0],
                               [0.0, 5.521778, 0.0],
                               [0.0, 0.0, 35.1856100]])
# 単位格子内の励起原子のインデックスリスト (list[int])
EMITTER_ATOM_INDEX_LIST = [0, 1, 9, 11, 12, 21, 22]
# 単位格子内の最適化する原子のインデックスリスト (list[int])
OPT_ATOM_INDEX_LIST = [1, 3, 11, 15]
# 同じ高さの原子全てに同じ乱数を適用する  bool
LAYER_OPTIMIZATION = True


"""クラスターの定数 [Angstrom]"""
# 形状(spherical, column, cuboid)  str
CLUSTER_SHAPE = "column"
# 半径  float
RADIUS = 15
# X軸  float
X_MAX = 5.0
X_MIN = -5.0
# Y軸  float
Y_MAX = 5.0
Y_MIN = -5.0
# Z軸  float
Z_MAX = 10.0
Z_MIN = -2.0
# 作成するクラスターのファイル名  str
CLUSTER_FILE = WORK_DIRECTORY+r"\cluster.xyz"

"""シミュレーション"""
# cmd_tmsp.exeのパス  (str)
CMD_TMSP = r"C:\Users\holo\3D-AIR-IMAGE-1100-64bit\plugins\qt5125_vs2015x64\cmd_tmsp.exe"
# シミュレーション用のファイル  (str)
EMITTER_WAVE_FUNCTION = WORK_DIRECTORY+r"\emitter.xml_emt"
PHASE_SHIFT = WORK_DIRECTORY+r"\phaseshift.xml_ph"
FLUCTUATION = WORK_DIRECTORY+r"\fluctuation.xml_flu"
ANGLE_RESOLUTION = "1"
MAPPING_METHOD = "AE"
IMFP = "14.29"
# 散乱波関数のパスとファイル名  (str)
SCATTERED_WAVE_FUNCTION = WORK_DIRECTORY+r"\scatter.xml_sf"
# 実測ホログラムのパスとファイル名  (str)
EXP_IMAGE = WORK_DIRECTORY+r"\nGaN1_Ga3d_fit6_peak2.xml_air"
# 出力する計算ホログラムのパスとファイル名 (str)
CALC_IMAGE = WORK_DIRECTORY+r"\calc_pattern.xml_air"
# ホログラムの面対称操作  bool
SYMMETRIZE_VERTICAL_MIRROR = False
SYMMETRIZE_HORIZONTAL_MIRROR = True

"""全般"""
# 変位させる軸  int (X:0, Y:1, Z:2)
OPT_AXIS = 2
# パラメータ初期値からの最大値変位量  list[min, max, step]
VARIATION = [-1.5, 1.5, 0.1]
# 最小結合長  float
MIN_BOND_LENGTH = 1.2
# 一世代の個体数  int
POPULATION_SIZE = 50
# 終了させる世代  int
FINAL_GENERATION = 1000
# 記録(CSVファイル)  (str)
RECORD = WORK_DIRECTORY+r"\optimization_process.csv"

"""評価"""
# 損失関数(MSE, MAE, RMSE) (str)
LOSS_FUNCTION = "RMSE"

"""交叉"""
# 方式(uniform) (str) uniform, vector
CROSSOVER_METHOD = "vector"
N_PARAMETER = 2

"""突然変異"""
# 個体の突然変異確率(0.00-1.00) (float)
P_INDIVIDUAL_MUTATION = 0.20
# 遺伝子の突然変異確率(0.00-1.00) (float)
P_GENE_MUTATION = 0.10

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
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 3D AIR IMAGEと接続
api = api_3d_air_image.Api()
api.connect()

ae = AtomicEditor.AE()
img = Image2D.Image2D()

opt1d = GeneticAlgorithm.Opt1D(UNIT_CELL_FILE,
                               TRANSLATION_VECTOR,
                               EMITTER_ATOM_INDEX_LIST,
                               OPT_ATOM_INDEX_LIST,
                               OPT_AXIS,
                               VARIATION,
                               MIN_BOND_LENGTH,
                               LAYER_OPTIMIZATION)


def main():
    # 実験像読み込み
    api.image_load(EXP_IMAGE)
    api.update()
    exp_image = api.image_get(api.window_get_list_wid_and_caption()[-1]['wid'])
    api.window_close(api.window_get_list_wid_and_caption()[-1]['wid'])
    print("Experimental image loaded\n")

    # 全データを記録するリスト
    l_csv = []
    header = ["Generation", LOSS_FUNCTION]
    for i in OPT_ATOM_INDEX_LIST:
        header.extend(["atom" + str(i) + "_" + opt1d.l_unit[i]['e'] + "_" + str(opt1d.axis)])
    header.extend(["CalcTime (sec)"])
    l_csv.append(header)

    # 最初の個体集団を生成
    print("Generating first generation population...\n")
    population = []
    for i in range(POPULATION_SIZE):
        print("Generating individual %d..." % (i + 1))
        population.append(opt1d.individual_generate())
        print("...Completed")

    # 仮の最適個体
    best_individual = copy.deepcopy(population[0])
    best_individual.set_evaluation(10.0)

    #
    # メインループ
    #
    for count_ in range(1, FINAL_GENERATION + 1):
        print("START LOOP %d CALCULATION...\n" % count_)
        # 交叉(Crossover)
        offspring_population = []
        if CROSSOVER_METHOD == "uniform":
            for i in range(0, POPULATION_SIZE, 2):
                offspring_population.extend(opt1d.crossover_uniform(population[i], population[i+1], N_PARAMETER))
        elif CROSSOVER_METHOD == "vector":
            for i in range(0, POPULATION_SIZE, 2):
                offspring_population.extend(opt1d.crossover_vector(population[i], population[i+1]))
        population.extend(offspring_population)
        # 突然変異(Mutation)
        population = opt1d.mutation(population, P_INDIVIDUAL_MUTATION, P_GENE_MUTATION)
        # 評価(Evaluation)
        for i in range(len(population)):
            calc_time = 0.0
            # 評価されていない個体(evaluation=0)に評価を行う
            if population[i].get_evaluation() == 0.0:
                # 個体の遺伝子情報を適用した単位格子からクラスターを作成
                if CLUSTER_SHAPE == "spherical":
                    opt1d.cluster_spherical(population[i], CLUSTER_FILE, RADIUS)
                elif CLUSTER_SHAPE == "column":
                    opt1d.cluster_column(population[i], CLUSTER_FILE, RADIUS, Z_MAX, Z_MIN)
                elif CLUSTER_SHAPE == "cuboid":
                    opt1d.cluster_cuboid(population[i], CLUSTER_FILE, X_MAX, X_MIN, Y_MAX, Y_MIN, Z_MAX, Z_MIN)
                else:
                    print("ERROR: Input CLUSTER_SHAPE does not exist.")
                    exit(1)
                # シミュレーションホログラムの作成
                start_time = time.time()
                # 散乱計算
                subprocess.run(CMD_TMSP
                               + " -C " + CLUSTER_FILE
                               + " -F " + EMITTER_WAVE_FUNCTION
                               + " -P " + PHASE_SHIFT
                               + " -W " + SCATTERED_WAVE_FUNCTION
                               + " -T " + FLUCTUATION
                               + " -R " + ANGLE_RESOLUTION
                               + " -M " + MAPPING_METHOD
                               + " -MFPES " + IMFP
                               + " -I " + CALC_IMAGE)
                # 1回のシミュレーション時間を計算
                calc_time = round(time.time() - start_time)
                # Load calculation hologram
                api.image_load(CALC_IMAGE)
                api.update()
                calc_image = api.image_get(api.window_get_list_wid_and_caption()[-1]['wid'])
                api.window_close(api.window_get_list_wid_and_caption()[-1]['wid'])
                # 計算ホログラムの対称操作
                if SYMMETRIZE_HORIZONTAL_MIRROR:
                    calc_image = img.image2D_symmetrize_horizontal_mirror(calc_image)
                elif SYMMETRIZE_VERTICAL_MIRROR:
                    calc_image = img.image2D_symmetrize_vertical_mirror(calc_image)
                # 二つのホログラム画像の差分を計算し結果をGA Classに保存
                if LOSS_FUNCTION == "MSE":
                    population[i].set_evaluation(img.image2D_MSE(exp_image, calc_image))
                elif LOSS_FUNCTION == "MAE":
                    population[i].set_evaluation(img.image2D_MAE(exp_image, calc_image))
                elif LOSS_FUNCTION == "RMSE":
                    population[i].set_evaluation(img.image2D_RMSE(exp_image, calc_image))
                else:
                    print("ERROR: Set LOSS_FUNCTION to 'MAE', 'MSE', or 'RMSE'.")
                # 最適解を保存
                if population[i].get_evaluation() < best_individual.get_evaluation():
                    best_individual = copy.deepcopy(population[i])
                    optimized_unit_cell = opt1d.chromosome_to_unit_cell(best_individual.get_chromosome())
                    ae.save_text(ae.XYZ_text(optimized_unit_cell), WORK_DIRECTORY+r"\optimized_unit_cell.xyz")
            # 全てのデータをCSVファイルに記録
            row = [count_, population[i].get_evaluation()]
            row.extend(population[i].get_chromosome())
            row.append(calc_time)
            l_csv.append(copy.deepcopy(row))
            fp = open(RECORD, 'w', encoding='utf-8', newline="")
            writer = csv.writer(fp)
            writer.writerows(l_csv)
            fp.close()
        #
        # 一世代の計算終了
        #
        print("...COMPLETE LOOP %d CALCULATION\n" % count_)
        # 淘汰(Selection)
        # ルーレット選択方式
        if SELECT_METHOD == "roulette":
            population = opt1d.selection_roulette(population, POPULATION_SIZE,
                                                  P_GROUP_A, P_GROUP_B, P_GROUP_C, P_GROUP_D, P_GROUP_E)
        # トーナメント方式(未完成)
        elif SELECT_METHOD == "tournament":
            print("selection error")
            exit()
        # 次世代に移行


if __name__ == '__main__':
    main()
