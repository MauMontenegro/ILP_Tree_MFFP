# Moving Firefighter Problem on Trees
    # Author: Mauro Alejandro Montenegro Meza

from pulp import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import json
import random as rnd
import re

def GDN(dv_string):
    new_string= re.sub(r"[\[]","",dv_string)
    new_string = re.sub(r"[\]]", "", new_string)
    split_key=new_string.split(',')
    return split_key

# Generate Random Tree with initial fire_root
N = 20  # Number of Nodes
seed = 140  # Experiment Seed
scale = 10  # Scale of distances
starting_fire = rnd.randint(0, N - 1) # Fire in random node
print('Starting fire in Node: {sf}'.format(sf=starting_fire))

# Adding Bulldozer
a_x_pos = rnd.uniform(-1, 1) * scale
a_y_pos = rnd.uniform(-1, 1) * scale
print('Initial Bulldozer Position: [{bx},{by}]'.format(bx=a_x_pos,by=a_y_pos))

# Create a Random Tree (nx use a Prufer Sequence) and get 'pos' layout for nodes
T = nx.random_tree(n=N, seed=seed)
# Induce a BFS path to get the fire propagation among levels
T = nx.bfs_tree(T, starting_fire)
# Could use spring or spectral Layout
pos = nx.spring_layout(T, seed=seed)

# Create Empty Adjacency Matrix for Full Distances
T_Ad = np.zeros((N + 1, N + 1))

# Save Original Tree in a "adjlist" file
nx.write_adjlist(T, "MFF_Tree.adjlist")
nx.draw_networkx(T,pos=pos)
plt.show()
# Create Adjacency Matrix with escalated distances in layout
for row in range(0, N):
    for column in range(row, N):
        if row == column:
            T_Ad[row][column] = 0
        else:
            x_1 = pos[row][0]
            x_2 = pos[column][0]
            y_1 = pos[row][1]
            y_2 = pos[column][1]
            dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
            T_Ad[row][column] = dist * scale  # Scale factor of 10

#Scale Distances in Layout to better plot
for element in pos:
    pos[element][0] = pos[element][0] * scale
    pos[element][1] = pos[element][1] * scale

# Adding Bulldozer distances to Full Adjacency Matrix
for node in range(0, N):
    x_1 = pos[node][0]
    x_2 = a_x_pos
    y_1 = pos[node][1]
    y_2 = a_y_pos
    dist = np.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
    T_Ad[node][N] = dist

# Create a Symmetric Matrix with upper part of T_Ad (For symmetric distances)
T_Ad_Sym = np.triu(T_Ad) + np.tril(T_Ad.T)

# Add Bulldozer Node to Tree and add his escalated position
T.add_node(N)
pos[N] = [a_x_pos, a_y_pos]

#Saving Full Distance Matrix
f = open("Full_Matrix.txt", "w")
f.write(str(T_Ad_Sym))

# Just Showing NX Tree
#nx.draw_networkx(T, pos=pos, with_labels=True)
#plt.show()

# Saving Layout in to a json file
for element in pos:
    pos[element] = list(pos[element])

with open('layout_MFF.json', 'w') as layout_file:
    layout_file.write(json.dumps(pos))

# Build Node Structure for LP
Nodes = list(T.nodes)
Nodes.remove(starting_fire)
Nodes.remove(N)
print(Nodes)

# Pre-Compute Data
###########################################################################################################
# Pre-Compute Burning_Times for each node in T
levels = nx.single_source_shortest_path_length(T, starting_fire) #Obtain Level in Tree for each node
print(levels)

# Pre-Compute time from node to node in already saved in Full_Adjacency Matrix

# Pre-Compute cardinality of a node sub-tree (saved nodes if defend)
weights = {}
for node in Nodes:
    weights[node] = len(nx.descendants(T, node)) + 1

print(weights)
############################################################################################################

# Create LP Problem
prob = LpProblem("Moving_Firefighter_Tree", LpMaximize)

# Create Decision Variables: (X_k,u,v)
# Create all edges (without initial position)
all_edges = []
for node1 in Nodes:
    for node2 in Nodes:
        if node1!=node2:
            edge = [node1, node2]
            all_edges.append(edge)
edge_number = len(all_edges)

# Create all edges for initial position
all_initial_edges = []
for node1 in Nodes:
    edge=[N,node1]
    all_initial_edges.append(edge)

# Create all phases for all edges without initial pos
variables = []
# Array Containing all edges per phase
phases = [x for x in range(1, N+1)]           # Array of total phases (Max node number)
print(phases)
for phase in phases:
    edges_per_phase={}
    for edge in all_edges:                    # Here, we fill all edges in this phase
        x = edge.copy()
        x.append(phase)
        edges_per_phase[str(x)] = edge[1]
    variables.append(edges_per_phase)

items_per_phase = []
for phase in phases:
    items_per_phase.append(variables[phase-1].keys())

# Create initial phase for all edges
variables_init = {}
for node in Nodes:
    x = [N, node]
    x.append(0)
    variables_init[str(x)] = node
items_init = variables_init.keys()

# Create Initial Decision Variables for LP with restrictions in range and type
lpvariables_init = LpVariable.dicts("Defend", items_init, 0, 1, LpBinary)

# Create Decision Variables for LP with restrictions in range and type for each phase
lpvariables_per_phase=[]
for phase in phases:
    lpvariables = LpVariable.dicts("Defend", items_per_phase[phase-1], 0, 1, LpBinary)
    lpvariables_per_phase.append(lpvariables)

# Sum Decision Variables
lps=0
counter=0
for phase in lpvariables_per_phase:
    lps+= lpSum([phase[i] * weights[variables[counter][i]] for i in variables[counter]])
    counter+=1

lps_init = lpSum([lpvariables_init[f] * weights[variables_init[f]] for f in variables_init])
lps_total = lps + lps_init

# Construct optimization problem
prob += (lps_total,
    "Sum_of_Defended_Edges",
)

# Constraints
#################################################################################################################
# 1) At phase 0, we only enable at most one edge to be active from p_0 to any node v
prob += (
        lpSum([lpvariables_init[i] for i in variables_init]) <= 1,
        "Initial_Edges",
    )

# 2) From phase 1 to N we only enable at most one edge to be active per phase
counter=0
for lpvariables_ in lpvariables_per_phase:
    prob += (
        lpSum([lpvariables_[i] for i in variables[counter]]) <= 1,
        "Edges_Phase_%s" %counter,
    )
    counter+=1

# 3) At phase 0, we only enable edge transitions that lead B from his initial position p_0
#    to nodes which B can reach before fire does.
prob += (
        lpSum([lpvariables_init[i] * T_Ad_Sym[GDN(i)[0]][GDN(i)[1]] for i in variables_init]) <=
        lpSum([lpvariables_init[i] * levels[GDN(i)[1]] for i in variables_init]),
        "Initial_Distance_Restriction",
    )




