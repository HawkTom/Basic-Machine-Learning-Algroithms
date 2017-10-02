import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sqrt


def graph_plot(map, start, end, wall, path=None):
    fig = plt.figure(0)
    axes = fig.add_subplot(111)

    axes.plot(start[0], start[1], 'ro')
    axes.plot(end[0], end[1], 'ko')

    for brick in wall:
        rec = patches.Rectangle(brick, 1, 1, fc = 'silver')
        axes.add_patch(rec)

    if path:
        axes.plot(path[:,0], path[:,1],'-',color = 'orange')

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([0, map[0]])
    axes.set_ylim([0, map[1]])
    # axes.set_xticklabels([])
    # axes.set_yticklabels([])
    axes.set_xticks(list(range(map[0]+1)))
    axes.set_yticks(list(range(map[1]+1)))
    axes.grid()
    axes.tick_params(length=0)
    plt.show()

def h_n(node1, node2):
    return(sqrt(pow(node1[0] - node2[0], 2) + pow(node1[1]-node2[1], 2)))


def next_position(node, position):
    if position == 0:
        return [node[0], node[1]+1]
    elif position == 1:
        return [node[0]+1, node[1]]
    elif position == 1:
        return [node[0], node[1]-1]
    elif position == 1:
        return [node[0]-1, node[1]]

def greedy_search(start, end, blocks, map):
    frontier = [(start,h_n(start,end))]
    for node in frontier:
        for i in range(4):
            node_next = next_position(node, i)
            if node_next not in blocks:
                frontier.append((node_next, h_n(node_next, end)))


def A_star_search(start,end, blocks, map):
    pass

map=[20, 12]
start = [2,7]
end = [18, 5]
draw_wall1 = [(4, i) for i in range(4, 12)]
draw_wall2 = [(13,i) for i in range(0, 8)]

graph_plot(map,start,end,draw_wall1+ draw_wall2)
greedy_search(start, end, draw_wall1, map)