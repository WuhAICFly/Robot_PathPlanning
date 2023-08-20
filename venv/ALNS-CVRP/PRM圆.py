import numpy as np
import csv
import random
import math
from pathfinding.finder.a_star import AStarFinder



class PRMPathPlanner(object):
    def __init__(self, obstacle_path, N, k):
        self.N = N
        self.k = k
        self.x_start = [-0.5, -0.5]
        self.goal = [0.5, 0.5]
        distance = self.distance_between_two_point(self.x_start, self.goal)
        self.nodes = [[1, self.x_start[0], self.x_start[1], distance]]
        self.epslon = 0.03
        self.edges_and_cost = []
        self.edges = []
        self.obstacles = []
        with open(obstacle_path, "rt") as f_obj:
            contents = csv.reader(f_obj)
            for row in contents:
                if row[0][0] != '#':
                    self.obstacles.append([float(row[0]), float(row[1]), float(row[2])])

    def distance_between_two_point(self, point_1, point_2):
        x1, y1 = point_1
        x2, y2 = point_2
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def calculate_vector_angle(self, a, b):
        a = np.array(a)
        b = np.array(b)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        a_dot_b = a.dot(b)
        value = a_dot_b / (a_norm * b_norm)
        if value > 1:
            value = 1
        if value < -1:
            value = -1
        # print(value)
        theta = np.arccos(value)
        return theta * 180 / np.pi

    def point_in_circle(self, point, circle):
        x, y, d = circle
        r = d / 2
        # self.epslon is accounted for the robot radius itself
        if self.distance_between_two_point(point, [x, y]) < r + self.epslon:
            return True
        else:
            return False

    # find foot of perpendicular of the point to the line
    def find_vertical_point(self, point, line):
        [x0, y0] = point
        [x1, y1, x2, y2] = line
        k = -1 * ((x1 - x0) * (x2 - x1) + (y1 - y0) * (y2 - y1)) / ((x2 - x1) ** 2 + (y2 - y1) ** 2)
        x = k * (x2 - x1) + x1
        y = k * (y2 - y1) + y1
        return [x, y]

    def point_within_line_segment(self, point, point1, point2):
        a = [point1[i] - point[i] for i in range(len(point))]
        b = [point2[i] - point[i] for i in range(len(point))]
        if self.calculate_vector_angle(a, b) > 90:
            return True
        else:
            return False

    # check if a line with point1 and point2 as its end
    # is in collision with a circle
    def in_collision_with_circle(self, point1, point2, circle):
        [x1, y1] = point1
        [x2, y2] = point2
        line = [x1, y1, x2, y2]
        [x0, y0, diameter] = circle
        radius = diameter / 2
        center = [x0, y0]
        vertical_point = self.find_vertical_point(center, line)

        # only when both point1 and point2 are outside the obstacle, the path might be collision free
        # otherwise the path must be in collision
        if not self.point_in_circle(point1, circle) and \
                not self.point_in_circle(point2, circle):
            distance_to_line = self.distance_between_two_point(vertical_point, center)

            if distance_to_line > radius:
                return False

            if self.point_within_line_segment(vertical_point, point1, point2):
                return True
            else:
                return False
        else:
            return True

    # check if specified line segment is in collision with any obstacles
    def in_collision(self, point1, point2):
        collision = False
        for each_circle in self.obstacles:
            if self.in_collision_with_circle(point1, point2, each_circle):
                collision = True
                break
        return collision

    def construct_roadmap(self):
        index = 1
        while len(self.nodes) < self.N:
            x = random.uniform(-0.5, 0.5)
            y = random.uniform(-0.5, 0.5)
            x_sample = [x, y]
            is_free = True
            for each_circle in self.obstacles:
                if self.point_in_circle(x_sample, each_circle):
                    is_free = False
            if is_free:
                index = index + 1
                distance = self.distance_between_two_point(x_sample, self.goal)
                self.nodes.append([index, x_sample[0], x_sample[1], distance])

        # add goal to nodes
        self.nodes.append([index + 1, self.goal[0], self.goal[1], 0])
        self.write_to_nodes_file()
        # print(self.nodes)

        # find k closest neighbors of each sample
        for each_node in self.nodes:
            current_index, x1, y1, cost1 = each_node
            current_pos = [x1, y1]
            neighbor_distance = []
            for other_node in self.nodes:
                other_index, x2, y2, cost2 = other_node
                if (current_index != other_index):
                    d = self.distance_between_two_point(current_pos, [x2, y2])
                    neighbor_distance.append([other_index, d])
            neighbor_distance = sorted(neighbor_distance, key=lambda d: d[1])

            # find k closest collision-free neighbors for current node
            del neighbor_distance[3:]
            for neighbor in neighbor_distance:
                neighbor_index, neighbor_cost = neighbor
                # extract neighbor information
                node, x, y, cost = self.nodes[neighbor_index - 1]
                if not self.in_collision(current_pos, [x, y]) \
                        and [current_index, neighbor_index] not in self.edges \
                        and [neighbor_index, current_index] not in self.edges:
                    self.edges.append([current_index, neighbor_index])
                    self.edges_and_cost.append([current_index, neighbor_index, neighbor_cost])
            # print(self.edges_and_cost)
        self.write_to_edges_file()

    def write_to_edges_file(self):
        with open('edges.csv', 'wt') as f_obj:
            writer = csv.writer(f_obj, delimiter=',')
            for each_row in self.edges_and_cost:
                writer.writerow(each_row)

    def write_to_nodes_file(self):
        with open('nodes.csv', 'wt') as f_obj:
            writer = csv.writer(f_obj, delimiter=',')
            for each_row in self.nodes:
                writer.writerow(each_row)


if __name__ == "__main__":
    prm = PRMPathPlanner('a.csv', 200, 3)
    prm.construct_roadmap()

    # now we have got a roadmap representing C-free
    # call A star algorithm to find the shortest path
    planner = AStarFinder("nodes.csv", "edges.csv")
    success, path = planner.search_for_path()
    if success:
        print(path[::-1])
        planner.save_path_to_file("path.csv")
    else:
        print("no solution found")
