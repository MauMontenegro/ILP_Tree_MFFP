# Original Firefighter Problem on Trees

from pulp import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import json
import random

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


# Generate Random Tree with initial fire_root
seed = 4576
N = 100
starting_fire = 0
T = nx.balanced_tree(2,4)

# Induce a BFS path to get the fire propagation among levels
T = nx.bfs_tree(T, starting_fire)

# Saving Position Layout of T
pos = hierarchy_pos(T, 0)
AdjacencyT = nx.to_numpy_array(T)

# Save Original Tree in a file
nx.write_adjlist(T, "FF_Tree.adjlist")

# Just Showing NX Tree
nx.draw_networkx(T, pos=pos, with_labels=True)
plt.show()

# Saving Layout in to a json file
for element in pos:
    pos[element] = list(pos[element])

with open('layout.json', 'w') as layout_file:
    layout_file.write(json.dumps(pos))

# Build Node Structure for LP
Nodes = list(T.nodes)
Nodes.pop(0)                               # We do not consider root as part of the problem
print(Nodes)

# Obtain weights of each node
weights = {}
for node in Nodes:
    weights[node] = len(nx.descendants(T, node)) + 1

print('weights')
print(weights)

# Creates LP problem variable
prob = LpProblem("Firefighter_Tree", LpMaximize)

# Create Decision Variables for LP with restrictions in range and type
variables = LpVariable.dicts("Defend", Nodes, 0, 1, LpBinary)
print(variables)

# Construct optimization problem
prob += (
    lpSum([variables[i] * weights[i] for i in Nodes]),
    "Sum_of_Defended_Nodes",
)
print(prob)

# Build Constraints
# Build Level Constraint (Only one defended node per level or time)
levels = nx.single_source_shortest_path_length(T, starting_fire)
levels.pop(0)
max_level = max(levels.values())
print('levels')
print(levels)

# Restriction Levels: For each Level has a list of nodes that are in that lvel
restriction_levels = {}
for level in range(1, max_level + 1):
    restriction_levels[level] = [i for i, j in levels.items() if j == level]

# Create a constraint for each level of T
for level_ in restriction_levels:
    prob += (
        lpSum([variables[i] for i in restriction_levels[level_]]) <= 1,
        "Sum_of_Defended_level%s" %level_,
    )

print('Restricted Levels')
print(restriction_levels)

# Build Leaf Path to root Constraints
# Obtaining all Leaf nodes from T
leaf_nodes = [node for node in T.nodes() if T.in_degree(node)!= 0 and T.out_degree(node) == 0]

# restricted_Ancestors: Keys are leaves of T and their values are his path to the root
restricted_ancestors = {}
for leaf in leaf_nodes:
    restricted_ancestors[leaf] = list(nx.ancestors(T, leaf))
    restricted_ancestors[leaf].remove(0)
    restricted_ancestors[leaf].append(leaf)
    if len(restricted_ancestors[leaf]) == 0:
        restricted_ancestors.pop(leaf)

print(restricted_ancestors)

# FOr each leaf path add a restriction to problem
for leaf in restricted_ancestors:
    prob += (
        lpSum([variables[i] for i in restricted_ancestors[leaf]]) <= 1,
        "Sum_of_Leaf_Path%s" %leaf,
    )

print(prob)

# The problem data is written to an .lp file
prob.writeLP("FFP_Tree.lp")

# The problem is solved using PuLP's choice of Solver (Default is CBC: Coin or branch and cut)
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
solution = {}
for v in prob.variables():
    print(v.name, "=", v.varValue)
    node_name = v.name.split("_")
    solution[node_name[1]] = v.varValue

# Nodes that are defended during solution
sol_nodes=[k for k,v in solution.items() if v == 1]
# Nodes that are burned during Solution
burned_nodes=[k for k,v in solution.items() if v == 0]

temp={}
temp2={}

# Get Level of defended nodes and save in temp
for node in sol_nodes:
    temp[node] = levels[int(node)]

# Sort Levels of defended nodes (To give temporal importance)
temp = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1])}
# Only want the keys, not the levels (we know that is sequential solution)
temp = [k for k in temp.keys()]

# Nodes burned per level after defend a node in that level
for i in range(1, len(temp)+1):
    # Remove defended node for each level (from temp{})
    remove = int(temp[i - 1])
    levels.pop(remove)
    # Obtain descendants of defended node and remove from graph
    # This is for not consider them and subyacent levels
    saved_nodes = nx.descendants(T, remove)
    for saved in saved_nodes:
        levels.pop(saved)

# sol_restriction_levels: Nodes that burn at each level (as we remove defended and all his descendants)
sol_restriction_levels = {}
for level in range(1, len(temp)+1):
    sol_restriction_levels[level] = [i for i, j in levels.items() if j == level]

with open('solution.json', 'w') as sol_file:
    sol_file.write(json.dumps([temp, sol_restriction_levels]))
print(sol_restriction_levels)

# The optimised objective function value is printed to the screen
print("Total Saved Trees = ", value(prob.objective))




# Now for next consecutive phases
for i in range(0,N-1):
    for item in items_per_phase[i]:
        keys=[]
        k_pos_var = lpvariables_per_phase[i][item]
        valid_input_edge = GDN(item)[1]
        sum = k_pos_var
        for item_ in lpvariables_per_phase[i+1]:
            valid_input_edge_ = GDN(item_)[0]
            if int(valid_input_edge) != int(valid_input_edge_):  # Restriction over other nodes
                keys.append(item_)
        sum += lpSum(lpvariables_per_phase[i+1][j] for j in keys)
        #print("Restriction in Phase: {p} for element {n}".format(p=i+1,n=item))
        #print(sum)
        prob += (
            sum <= 1,
            "Continuity_Restriction_{p},{n}".format(p=i,n=item),
        )

