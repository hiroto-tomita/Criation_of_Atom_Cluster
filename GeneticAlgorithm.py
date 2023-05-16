import copy
import random
import numpy as np
import AtomicEditor
from decimal import Decimal, ROUND_HALF_UP


ae = AtomicEditor.AE()


class Individual:

    def __init__(self, chromosome, evaluation):
        self.chromosome = chromosome  # 最適化するパラメータ(遺伝子)を格納したリスト(染色体)
        self.evaluation = evaluation  # 評価(適応度)

    def get_chromosome(self):
        return self.chromosome

    def get_evaluation(self):
        return self.evaluation

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome

    def set_evaluation(self, evaluation):
        self.evaluation = evaluation


class Opt1D:

    def __init__(self, unit_cell_file, translation_vector, l_emitter_index, l_opt_atom_index, opt_axis, variation, bond_length_min, layer_optimization):
        """

        :param unit_cell_file: (str)
        :param translation_vector: 基本並進ベクトル [2D-ndarray]
        :param l_emitter_index: エミッタ原子のインデックスリスト [list(int)]
        :param l_opt_atom_index: 変位させる原子のインデックスリスト [list(int)]
        :param opt_axis: 最適化する軸 (0:X, 1:Y, 2:Z)
        :param variation: 変位量 [list(min, max, step)]
        :param bond_length_min: 最小結合長 [float]
        """
        self.l_unit = ae.load_xyz_file(unit_cell_file)
        if len(translation_vector) != 3:
            print("ERROR:  'translation vector' argument should be (3x3) ndarray.")
            exit(1)
        self.trans_vector = translation_vector
        self.l_emitter_atom_index = l_emitter_index
        self.l_opt_atom_index = l_opt_atom_index
        if opt_axis != 0 and opt_axis != 1 and opt_axis != 2:
            print("ERROR:  'optimization axis' argument should be integer. (0: X-axis, 1: Y-axis, 2: Z-axis)")
            exit(1)
        self.axis = opt_axis
        if len(variation) != 3:
            print("ERROR:  'variation' argument should be [min, max, step] array.")
            exit(1)
        self.variation = variation
        self.bond_length_min = bond_length_min
        self.layer_opt = layer_optimization

    def gene_randomize(self, gene):
        """
        遺伝子のパラメータを乱数で変化させる。
        :param gene: 変位させるパラメータ (float)
        :return: 変位させたパラメータ (float)
        """
        d_min = 0.0
        d_max = 0.0
        step = self.variation[2]
        if self.variation[0] < self.variation[1]:
            d_max = self.variation[1]
            d_min = self.variation[0]
        elif self.variation[1] < self.variation[0]:
            d_max = self.variation[0]
            d_min = self.variation[1]
        else:
            print("ERROE:  Please check 'range_of_variation' ")
            exit(1)
        new_gene = Decimal(str(gene)) + Decimal(str(random.choice(list(np.arange(d_min, d_max + step, step)))))
        new_gene = float(new_gene.quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP))
        return new_gene

    def chromosome_check_bond_length(self, chromosome):
        """
        パラメータセットを適用したユニットセル内の原子間距離が近すぎないかをチェック
        :param chromosome: パラメータセット
        :return: (bool)
        """
        unit_cell = self.chromosome_to_unit_cell(chromosome)
        return ae.check_bond_length(unit_cell, self.trans_vector, self.bond_length_min)

    def chromosome_to_unit_cell(self, chromosome):
        """
        パラメータセットの値を適用したユニットセルを返す
        :param chromosome: パラメータセット list[float]
        :return: 原子配列リスト list
        """
        unit_cell = copy.deepcopy(self.l_unit)
        for i, opt in enumerate(self.l_opt_atom_index):
            unit_cell[opt]['pos'][self.axis] = chromosome[i]
            # 同じ高さの原子全体にパラメータを適用
            if self.layer_opt:
                for atom in unit_cell:
                    if self.l_unit[opt]['pos'][self.axis] == atom['pos'][self.axis]:
                        atom['pos'][self.axis] = chromosome[i]
        return unit_cell

    def individual_generate(self):
        """
        最小結合長が設定値以上の個体を生成
        :return: 個体  class
        """
        # デフォルトパラメータセットを単位格子リストから取得して乱数で変位させたパラメータセットを生成
        default_chromosome = []
        unit_cell = copy.deepcopy(self.l_unit)
        for opt in self.l_opt_atom_index:
            default_chromosome.append(copy.deepcopy(self.l_unit[opt]['pos'][self.axis]))
        while True:
            chromosome = []
            for gene in default_chromosome:
                chromosome.append(self.gene_randomize(gene))
            # パラメータセットをユニットセルに反映
            for i, opt in enumerate(self.l_opt_atom_index):
                unit_cell[opt]['pos'][self.axis] = chromosome[i]
            # 作成した個体によるユニットセルの最小結合長が設定値以上かを判定
            if ae.check_bond_length(unit_cell, self.trans_vector, self.bond_length_min):
                return Individual(chromosome, 0.0)
            else:
                continue

    def cluster_spherical(self, individual, cluster_file, radius):
        """
        個体の遺伝子を反映した球形のクラスターファイルを作成して保存する
        :param individual: 個体 (class)
        :param cluster_file: 生成するクラスターのパスとファイル名 (str)
        :param radius: クラスターの半径 (float)
        :return: 球形クラスターのファイルを保存
        """
        # 遺伝子情報を原子配列に反映
        unit_cell = self.chromosome_to_unit_cell(individual.get_chromosome())
        # クラスターファイルを作成して保存
        ae.save_spherical_clusters(unit_cell, self.trans_vector, self.l_emitter_atom_index, cluster_file, radius)

    def cluster_column(self, individual, cluster_file, radius, z_max, z_min):
        """
        個体の遺伝子を反映した円柱形のクラスターファイルを作成して保存する
        :param individual: 個体 (class)
        :param cluster_file: 生成するクラスターのパスとファイル名 (str)
        :param radius: クラスターの半径 (float)
        :param z_max: 原点からの高さ (float)
        :param z_min: 原点からの深さ (float)
        :return: 円柱形のクラスターファイルを保存
        """
        # 遺伝子情報を原子配列に反映
        unit_cell = self.chromosome_to_unit_cell(individual.get_chromosome())
        # クラスターファイルを作成して保存
        ae.save_column_clusters(unit_cell, self.trans_vector, self.l_emitter_atom_index, cluster_file,
                                radius,
                                z_max,
                                z_min)

    def cluster_cuboid(self, individual, cluster_file, x_max, x_min, y_max, y_min, z_max, z_min):
        """
        個体の遺伝子を反映した矩形のクラスターファイルを作成して保存する
        :param individual: 個体 (class)
        :param cluster_file: 生成するクラスターのパスとファイル名 (str)
        :param x_max: X軸の最大値 (float)
        :param x_min: X軸の最小値 (float)
        :param y_max: Y軸の最大値 (float)
        :param y_min: Y軸の最小値 (float)
        :param z_max: 原点からの高さ (float)
        :param z_min: 原点からの深さ (float)
        :return: 矩形のクラスターファイル
        """
        # 遺伝子情報を原子配列に反映
        unit_cell = self.chromosome_to_unit_cell(individual.get_chromosome())
        # クラスターファイルを作成して保存
        ae.save_cuboid_clusters(unit_cell, self.trans_vector, self.l_emitter_atom_index, cluster_file,
                                x_max,
                                x_min,
                                y_max,
                                y_min,
                                z_max,
                                z_min)

    def selection_roulette(self, population, selected_population_size,
                           pa=0.20, pb=0.20, pc=0.20, pd=0.20, pe=0.20):
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
            group.append(l_sort[(i * group_length):((i + 1) * group_length)])
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

    def crossover_uniform(self, parent_individual_1, parent_individual_2, nExchange=2):
        """
        一様交叉
        :param parent_individual_1: 一つ目の個体(親) (class)
        :param parent_individual_2: 二つ目の個体(親) (class)
        :param nExchange: 入れ替えるパラメータ数
        :return: 二つの子孫個体 <list>
        """
        # 子孫を格納するリスト
        l_offspring = []
        # 染色体の抽出
        chromosome_1 = parent_individual_1.get_chromosome()
        chromosome_2 = parent_individual_2.get_chromosome()
        nParam = len(chromosome_1)
        # 交換するパラメータの決定
        l_exchange_param_index = []  # 交換するパラメータのパラメータセット内のインデックス
        for i in range(nExchange):
            while True:
                num = random.randint(0, nParam-1)
                if num in l_exchange_param_index:
                    continue
                else:
                    l_exchange_param_index.append(copy.deepcopy(num))
                    break
        if len(l_exchange_param_index) != nExchange:
            print("ERROR:  Check 'Crossover section'")
            exit(1)
        # 交叉
        chromosome_3 = copy.deepcopy(chromosome_1)
        chromosome_4 = copy.deepcopy(chromosome_2)
        for j in l_exchange_param_index:
            chromosome_3[j] = copy.deepcopy(chromosome_2[j])
            chromosome_4[j] = copy.deepcopy(chromosome_1[j])
        # 子孫個体を検査。結合長が設定値以下の場合、入れ替えたパラメータを乱数で変位させる。
        if not self.chromosome_check_bond_length(chromosome_3):
            while True:
                for k in l_exchange_param_index:
                    chromosome_3[k] = self.gene_randomize(self.l_unit[self.l_opt_atom_index[k]]['pos'][self.axis])
                if self.chromosome_check_bond_length(chromosome_3):
                    break
        if not self.chromosome_check_bond_length(chromosome_4):
            while True:
                for k in l_exchange_param_index:
                    chromosome_4[k] = self.gene_randomize(self.l_unit[self.l_opt_atom_index[k]]['pos'][self.axis])
                if self.chromosome_check_bond_length(chromosome_4):
                    break
        # 子孫個体を生成
        l_offspring.append(Individual(chromosome_3, 0.0))
        l_offspring.append(Individual(chromosome_4, 0.0))
        print("...Generated offspring individuals")
        return l_offspring

    def crossover_vector(self, parent_individual_1, parent_individual_2):
        """

        :param parent_individual_1: 一つ目の個体(親) (class)
        :param parent_individual_2: 二つ目の個体(親) (class)
        :return: 二つの子孫個体 <list>
        """
        # 子孫を格納するリスト
        l_offspring = []
        # 染色体の抽出
        parent_chromosome_1 = np.array(parent_individual_1.get_chromosome())
        parent_chromosome_2 = np.array(parent_individual_2.get_chromosome())
        # 交叉
        offspring_chromosome_1 = list((parent_chromosome_1+parent_chromosome_2)/2.0)
        offspring_chromosome_2 = list((2*parent_chromosome_1-parent_chromosome_2))
        if self.chromosome_check_bond_length(offspring_chromosome_1):
            l_offspring.append(Individual(offspring_chromosome_1, 0.0))
        else:
            l_offspring.append(self.individual_generate())
        if self.chromosome_check_bond_length(offspring_chromosome_2):
            l_offspring.append(Individual(offspring_chromosome_2, 0.0))
        else:
            l_offspring.append(self.individual_generate())
        print("...Generated offspring individuals")
        return l_offspring

    def mutation(self, population, P_individual_mutation=0.10, P_gene_mutation=0.10):
        """
        突然変異関数
        :param population: 突然変異させる個体集団 (list(individual))
        :param P_individual_mutation: 個体に対する突然変異確率 (float)
        :param P_gene_mutation: 遺伝子一つ一つに対する突然変異確率 (float)
        :return: 突然変異処理を施した個体集団 (list(individual))
        """
        default_chromosome = []
        for i in self.l_opt_atom_index:
            default_chromosome.append(copy.deepcopy(self.l_unit[i]['pos'][self.axis]))
        new_population = []
        for individual in population:
            # 個体に対して一定の確率で突然変異を起こす
            if random.random() < P_individual_mutation:
                chromosome = individual.get_chromosome()
                while True:
                    new_chromosome = []
                    for j in range(len(chromosome)):
                        # 個体の遺伝子一つ一つに対して一定の確率で突然変異を起こす
                        if random.random() < P_gene_mutation:
                            new_chromosome.append(self.gene_randomize(default_chromosome[j]))
                        else:
                            new_chromosome.append(chromosome[j])
                    # 突然変異したパラメータセットを検査
                    if self.chromosome_check_bond_length(new_chromosome):
                        break
                # 突然変異が適用された場合は適応度を0にして再度評価する
                if chromosome != new_chromosome:
                    individual.set_evaluation(0.0)
                individual.set_chromosome(new_chromosome)
                new_population.append(copy.deepcopy(individual))
            else:
                new_population.append(copy.deepcopy(individual))
        return new_population
