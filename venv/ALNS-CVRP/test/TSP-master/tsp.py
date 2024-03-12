from ortools.constraint_solver import routing_enums_pb2, pywrapcp

def create_data_model():
    """创建数据模型"""
    data = {}
    locations = [(-664600, -1122100), (-681100, -1205800), (-628200, -1119600)]
    data['locations'] = locations
    data['num_locations'] = len(locations)
    data['depot'] = 0  # 起点为第一个位置
    return data

def distance(position_1, position_2):
    """计算两点之间的距离"""
    return ((position_1[0] - position_2[0]) ** 2 + (position_1[1] - position_2[1]) ** 2) ** 0.5

def tsp():
    """求解 TSP 问题"""
    # 创建数据模型
    data = create_data_model()

    # 创建求解器
    manager = pywrapcp.RoutingIndexManager(data['num_locations'], 1, [data['depot']])
    routing = pywrapcp.RoutingModel(manager)

    # 计算距离矩阵
    distance_matrix = {}
    for from_node in range(data['num_locations']):
        distance_matrix[from_node] = {}
        for to_node in range(data['num_locations']):
            if from_node == to_node:
                distance_matrix[from_node][to_node] = 0
            else:
                distance_matrix[from_node][to_node] = distance(data['locations'][from_node], data['locations'][to_node])
    def distance_callback(from_index, to_index):
        """回调函数，返回两点之间的距离"""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # 定义约束条件
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    dimension_name = 'Distance'
    routing.AddDimension(transit_callback_index, 0, 1000000, True, dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # 设置求解参数
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 10

    # 开始求解
    solution = routing.SolveWithParameters(search_parameters)

    # 输出结果
    if solution:
        print('Total distance: {} meters'.format(solution.ObjectiveValue()))
        index = routing.Start(0)
        plan_output = 'Route for vehicle 0:\n'
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        plan_output += 'Total distance: {} meters'.format(route_distance)
        print(plan_output)

tsp()
