import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sqrt
import numpy as np

class Path(object):
    def __init__(self, node, cost, path = None):
        if not path:
            self.road = [node]
            self.gn = cost
        else:
            self.road = path.road + [node]
            self.gn = path.gn + cost
        self.hn = h_n(node, end)
        self.fn = self.gn + self.hn

    def set_gn(self, cost):
        self.gn = cost
        self.fn = self.gn + self.hn

    def set_hn(self, cost):
        self.hn = cost
        self.fn = self.gn + self.hn




def search_plot(map, start, end, wall, path=None):
    fig = plt.figure(0)
    axes = fig.add_subplot(111)
    axes.plot(start[0], start[1], 'ro')
    axes.plot(end[0], end[1], 'ko')
    if len(path[0])!=0:
        axes.plot(path[0][:,0], path[0][:,1],'-',color = 'orange')
    if len(path[1]) != 0:
        axes.plot(path[1][:,0], path[1][:,1], '-', color='green')

    for brick in wall:
        rec = patches.Rectangle(brick, 1, 1, fc='silver')
        axes.add_patch(rec)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([0, map[0]])
    axes.set_ylim([0, map[1]])
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_xticks(list(range(map[0] + 1)))
    axes.set_yticks(list(range(map[1] + 1)))
    axes.grid()
    axes.tick_params(length=0)

    plt.show()


def h_n(node1, node2):
    return(sqrt(pow(node1[0] - node2[0], 2) + pow(node1[1]-node2[1], 2)))

def next_position(node, position):
    if position == 0:
        return [node[0], node[1]+1]
    elif position == 1:
        return [node[0]+1, node[1]+1]
    elif position == 2:
        return [node[0]+1, node[1]]
    elif position == 3:
        return [node[0]+1, node[1]-1]
    elif position == 4:
        return [node[0], node[1]-1]
    elif position == 5:
        return [node[0]-1, node[1]-1]
    elif position == 6:
        return [node[0]-1, node[1]]
    elif position == 7:
        return [node[0]-1, node[1]+1]

def A_star_search(start,end, blocks, map):
    frontier = [Path(start, 0)]
    explored = []
    while True:
        node = min(frontier, key=lambda x: x.fn)
        frontier.remove(node)
        explored.append(node.road[-1])

        if node.road[-1] == end:
            return node.road

        for i in range(8):
            path_next_node = next_position(node.road[-1], i)
            if path_next_node not in blocks and path_next_node not in explored \
                    and path_next_node[0]>=0 and path_next_node[0] <= map[0] \
                    and path_next_node[1]>=0 and path_next_node[1] <= map[1]:
                action_cost = h_n(node.road[-1], path_next_node)
                newPath = Path(path_next_node, action_cost, node)
                frontier.append(newPath)

def greedy_search(start, end, blocks, map):
    frontier = [Path(start, 0)]
    explored = []
    while True:
        node = min(frontier, key=lambda x: (x.hn, x.fn))
        frontier.remove(node)
        explored.append(node.road[-1])
        # print(node.road)
        if node.road[-1] == end:
            return node.road

        for i in range(8):
            path_next_node = next_position(node.road[-1], i)
            if path_next_node not in blocks and path_next_node not in explored \
                    and path_next_node[0] >= 0 and path_next_node[0] <= map[0] \
                    and path_next_node[1] >= 0 and path_next_node[1] <= map[1]:
                action_cost = h_n(node.road[-1], path_next_node)
                newPath = Path(path_next_node, action_cost, node)
                frontier.append(newPath)

def wall_trans(draw_wall):
    wall = []
    for corner in draw_wall:
        if corner not in wall:
            wall.append(corner)
        if [corner[0]+1, corner[1]] not in wall:
            wall.append([corner[0]+1, corner[1]])
        if [corner[0] + 1, corner[1]+1] not in wall:
            wall.append([corner[0]+1, corner[1]+1])
        if [corner[0], corner[1]+1] not in wall:
            wall.append([corner[0], corner[1]+1])
    return wall


map=[20, 12]
start = [2, 7]
end = [18, 5]
wall1 = [[6, i] for i in range(4, 12)] + [[5,2]]
wall2 = [[13,i] for i in range(0, 8)]

path1 = greedy_search(start, end, wall_trans(wall1+wall2), map)
path2 = A_star_search(start, end, wall_trans(wall1+wall2), map)
# path1 = []
# path2 = []
search_plot(map,start,end,wall1+ wall2, [np.array(path1),np.array(path2)])
print("OK")