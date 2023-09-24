
import numpy as np
class TSP_GA:
    def __init__(self,
                 iteration,
                 cost_mat,
                 pop_size,
                 mutation_rate,
                 elite_rate,
                 cross_rate):
        self.iteration = iteration # 迭代次数
        self.cost_mat = cost_mat # 成本矩阵
        self.cities = self.cost_mat.shape[0]  # 城市个数
        self.pop_size = pop_size # 种群数量
        self.mutate_rate = mutation_rate # 变异率
        self.elite_rate = elite_rate # 精英率

    def initial_population(self): # 种群初始化
        population = []
        for i in range(self.pop_size):
            cities_list = list(range(self.cities))
            random.shuffle(cities_list)
            population.append(cities_list)
        return population

    def fitness(self, individual): # 计算每个个体的适应度
        route_cost = 0
        for i in range(len(individual) - 1):
            start = individual[i]
            end = individual[i + 1]
            route_cost += self.cost_mat[start][end]
        route_cost += self.cost_mat[individual[-1]][individual[0]]
        return route_cost

    @staticmethod
    def __crossover(parent_a, parent_b): # 交叉
        """
        :param parent_a: 父代基因A
        :param parent_b: 父代基因B
        :return:
        """
        child = [None] * len(parent_a)  # 创建一个子代个体
        start_index, end_index = random.sample(range(len(parent_a)), 2)  # 随机生成两个索引点，用于截取基因片段
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        child[start_index:end_index] = parent_a[start_index:end_index]  # 截取父代基因A中的片段，并赋给子代基因的对应位置
        remaining_genes = [gene for gene in parent_b if gene not in child]  # 从另一个父代基因组中选择剩余的基因
        i = 0
        for gene in remaining_genes:
            while child[i] is not None:  # 找到子代基因组中的空位
                i += 1
            child[i] = gene  # 将基因填充到子代基因组中
        return child  # 返回子代基因组

    def select_elites(self, population, fitnesses): # 精英选择
        # 计算被选出精英个体的数量
        num_elites = int(len(population) * self.elite_rate)  # 精英数量

        # 根据适应度对种群进行排序
        sorted_population = [individual for _, individual in sorted(zip(fitnesses, population))]

        # 选取适应度大的前几个
        elites = sorted_population[:num_elites]

        return elites

    def select_two_parents(self, population, fitnesses): #
        total_fitness = sum(fitnesses)
        selection_probability = [fitness / total_fitness for fitness in fitnesses]

        # 选择父代 A
        parent_a_index = random.choices(range(len(population)), weights=selection_probability, k=1)[0]
        parent_a = population[parent_a_index]

        # 选择父代 B
        population_without_a = population[:parent_a_index] + population[parent_a_index + 1:]
        fitnesses_without_a = fitnesses[:parent_a_index] + fitnesses[parent_a_index + 1:]
        total_fitness = sum(fitnesses_without_a)
        selection_probability = [fitness / total_fitness for fitness in fitnesses_without_a]
        parent_b_index = random.choices(range(len(population_without_a)), weights=selection_probability, k=1)[0]
        parent_b = population_without_a[parent_b_index]

        return parent_a, parent_b


    def displacement_mutation(self, individual):
        """
        置换变异
        :param individual: 个体
        """
        i, j = sorted(random.sample(range(len(individual)), 2))
        k = random.randint(0, len(individual) - (j - i + 1))
        genes = individual[i:j + 1]
        del individual[i:j + 1]
        individual[k:k] = genes
        return individual



    def solve(self):
        population = self.initial_population()  # init polpulation
        best_fitness = []
        for i in range(self.iteration):  # iteration
            fitnesses = [self.fitness(individual) for individual in population]  # 求解每个个体的适应度并保存为列表
            next_population = self.select_elites(population, fitnesses)  # 精英选择
            while len(next_population) < self.pop_size:
                parent_a, parent_b = self.select_two_parents(population, fitnesses)
                child_a = self.__crossover(parent_a, parent_b)  # 交叉，生成子代个体a
                child_b = self.__crossover(parent_b, parent_a)  # 交叉，生成子代个体b
                if random.random() < self.mutate_rate:
                    child_a = self.displacement_mutation(child_a)  # 变异
                if random.random() < self.mutate_rate:
                    child_b = self.displacement_mutation(child_b)  # 变异
                next_population.append(child_a)  # 将子代基因组添加到新的种群中
                next_population.append(child_b)
            population = next_population
            fitnesses = [self.fitness(individual) for individual in population]  # 求解每个个体的适应度并保存为列表
            best_fitness.append(min(fitnesses))
            print('当前迭代进度:{}/{},最佳适应度为:{}'.format(i, self.iteration, min(fitnesses)))
        fitnesses = [self.fitness(individual) for individual in population]
        best_individual = population[fitnesses.index(min(fitnesses))]
        return best_individual, min(fitnesses), best_fitness
distmat = np.array([[0,350,290,670,600,500,660,440,720,410,480,970],
                 [350,0,340,360,280,375,555,490,785,760,700,1100],
                 [290,340,0,580,410,630,795,680,1030,695,780,1300],
                 [670,360,580,0,260,380,610,805,870,1100,1000,1100],
                 [600,280,410,260,0,610,780,735,1030,1000,960,1300],
                 [500,375,630,380,610,0,160,645,500,950,815,950],
                 [660,555,795,610,780,160,0,495,345,820,680,830],
                 [440,490,680,805,735,645,495,0,350,435,300,625],
                 [720,785,1030,870,1030,500,345,350,0,475,320,485],
                 [410,760,695,1100,1000,950,820,435,475,0,265,745],
                 [480,700,780,1000,960,815,680,300,320,265,0,585],
                 [970,1100,1300,1100,1300,950,830,625,485,745,585,0]])

TSP = TSP_GA(teration=10000,
             cost_mat=distmat,
             pop_size=150,
             mutation_rate=0.05,
             elite_rate=0.4,
             cross_rate=0.7)  # 实例化，初始化参数

TSP_RESULT = TSP.solve()  # 调用TSP_GA中的 solve求解函数
data = pd.read_csv('data.tsp', header=None) # 测试数据集
data_dic = list(data.itertuples(index=False, name=None))
distances = squareform(pdist(data_dic)) # 生成距离矩阵
for point in data_dic:
    label, x, y = point
    plt.scatter(x, y,c='darkturquoise',s=220)
    plt.text(x, y, label, ha='center', va='center_baseline',fontdict={'size': 10, 'color': 'black'})
