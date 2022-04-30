# Moving Firefighter Problem on Trees
    # Author: Mauro Alejandro Montenegro Meza

from pulp import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import random as rnd
import re


BN = 1000 # Big Number for Restrictions

# Function that split string of [u,v,p]
# where (u,v) is an edge and p is the phase
def GDN(dv_string):
    new_string= re.sub(r"[\[]","",dv_string)
    new_string = re.sub(r"[\]]", "", new_string)
    split_key=new_string.split(',')
    return split_key

load = True

if load:
    T = nx.read_adjlist("instance/MFF_Tree.adjlist")
    # Relabeling Nodes
    mapping={}
    for node in T.nodes:
        mapping[node] = int(node)
    T= nx.relabel_nodes(T, mapping)
    T_Ad_Sym = np.load("instance/FDM_MFFP.npy")
    lay = open('instance/layout_MFF.json')
    pos={}
    pos_=json.load(lay)

    for position in pos_:
        pos[int(position)]=pos_[position]
    # Get Instance Parameters
    p = open('instance/instance_info.json')
    parameters = json.load(p)
    N = parameters['N']
    seed = parameters['seed']
    scale = parameters['scale']
    starting_fire = parameters['start_fire']

    a_x_pos=parameters['a_pos_x']
    a_y_pos = parameters['a_pos_y']

    T = nx.bfs_tree(T, starting_fire)
    T.add_node(N)

    #pos[N] = [a_x_pos, a_y_pos]
    nx.draw_networkx(T, pos=pos)
    nx.draw_networkx_nodes(T, pos, T.nodes, node_color="tab:red")
    # plt.show()
    plt.savefig('Graph_Test.png')


else:
    # Generate Random Tree with initial fire_root
    N = 12  # Number of Nodes
    seed = 150  # Experiment Seed
    scale = 1  # Scale of distances
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
    #plt.show()
    plt.savefig('Graph_Test.png')

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

    print(T_Ad_Sym)

    # Add Bulldozer Node to Tree and add his escalated position
    T.add_node(N)
    pos[N] = [a_x_pos, a_y_pos]

    #Saving Full Distance Matrix
    f = open("Full_Matrix.txt", "w")
    f.write(str(T_Ad_Sym))

    # Just Showing NX Tree
    nx.draw_networkx(T, pos=pos, with_labels=True)
    #plt.show()
    plt.savefig('Graph_Test.png')

    # Saving Layout in to a json file
    for element in pos:
        pos[element] = list(pos[element])

    with open('layout_MFF.json', 'w') as layout_file:
        layout_file.write(json.dumps(pos))

# Build Node Structure for LP
Nodes = list(T.nodes)
print(Nodes)

print(starting_fire)
Nodes.remove(starting_fire)
print(Nodes)
print(N)
Nodes.remove(N)
print(Nodes)

# Pre-Compute Data
###########################################################################################################
# Pre-Compute Burning_Times for each node in T
levels = nx.single_source_shortest_path_length(T, starting_fire) #Obtain Level in Tree for each node
print(levels)
print("Levels of each node in directed Tree:")
print(levels)

# Pre-Compute time from node to node in Full_Adjacency Matrix

############################################################################################################

# Create LP Problem
prob = LpProblem("Moving_Firefighter_Tree", LpMaximize)

# Create Decision Variables: (X_k,u,v)
# Create all edges (without initial position)
all_edges = []
for node1 in Nodes:
    for node2 in Nodes:
        #if node1 != node2:
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

for phase in phases:
    edges_per_phase={}
    for edge in all_edges:                    # Here, we fill all edges in this phase
        x = edge.copy()
        x.append(phase)
        if edge[0] == edge[1]:
            edges_per_phase[str(x)] = 'f'
        else:
            edges_per_phase[str(x)] = edge[1]
    variables.append(edges_per_phase)

# Pre-Compute cardinality of a node sub-tree (saved nodes if defended)
weights = {}
for node in Nodes:
    weights[node] = len(nx.descendants(T, node)) + 1
weights['f'] = 0

print("Weights for each node")
print(weights)

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
lps = 0
counter=0

lps_init = lpSum([lpvariables_init[f] * weights[variables_init[f]] for f in variables_init])

print(variables[0])

for phase in lpvariables_per_phase:
    lps += lpSum([phase[i] * weights[variables[counter][i]] for i in variables[counter]])
    counter += 1



lps_total = lps + lps_init

# Construct optimization problem
prob += (lps_total,
    "Sum_of_Defended_Edges",
)


# Constraints
#################################################################################################################
# 1) At phase 0, we only enable at most one edge to be active from p_0 to any node v

first_constraint = lpSum([lpvariables_init[i] for i in variables_init])

prob += (
         first_constraint == 1,
        "Initial_Edges",
    )

# 2) From phase 1 to N we only enable at most one edge to be active per phase
counter=0
for lpvariables_ in lpvariables_per_phase:
    prob += (
        lpSum([lpvariables_[i] for i in variables[counter]]) == 1,
        "Edges_Phase_%s" %counter,
    )
    counter+=1

# 3) At phase 0, we only enable edge transitions that lead B from his initial position p_0
#    to nodes which B can reach before fire does.


prob += (
        lpSum([lpvariables_init[i] * T_Ad_Sym[int(GDN(i)[0])][int(GDN(i)[1])] for i in variables_init]) <=
        lpSum([lpvariables_init[i] * levels[int(GDN(i)[1])] for i in variables_init]),
        "Initial_Distance_Restriction",
    )

# 4) From phase 1 to n we enable only edges that lead B to valid nodes from his current position. The sum of distances
#    from p0 to current position following active edges must be less that the time it takes the fire to reach a node from
#    the nearest fire root.
r_init= lpSum([lpvariables_init[i] * T_Ad_Sym[int(GDN(i)[0])][int(GDN(i)[1])] for i in variables_init])
counter=0
dist_r_i = r_init
dist_r_d=0

print("\\Assad  sadasd  asdsd\\")
for lpvariables_ in lpvariables_per_phase: # At each loop sum one new phase (Cumulative)
    dist_r_i += lpSum([lpvariables_[i] * T_Ad_Sym[int(GDN(i)[0])][int(GDN(i)[1])] for i in variables[counter]])

    dist_r_d = lpSum([lpvariables_[i] * levels[int(GDN(i)[1])] for i in variables[counter]])
    disable_r = BN * (1 - lpSum([lpvariables_[i] for i in variables[counter]]))
    dist_r_d += disable_r
    prob += (
        dist_r_i <= dist_r_d,
        "Distance_Restriction_%s" %counter,
    )
    counter+=1

# 5) We only enable one defended node in the path of each leaf to the root
leaf_nodes = [node for node in T.nodes() if T.in_degree(node)!= 0 and T.out_degree(node) == 0]
restricted_ancestors={}
for leaf in leaf_nodes:
    restricted_ancestors[leaf] = list(nx.ancestors(T , leaf))
    restricted_ancestors[leaf].remove(starting_fire)
    restricted_ancestors[leaf].insert(0,leaf)


p0 = str(N)

for leaf in restricted_ancestors:
    r=0
    for node in restricted_ancestors[leaf]:
        # Generate only edges that goes to 'node'
        valid_nodes = Nodes.copy()
        valid_nodes.remove(node)
        valid_edges=[[int(i),int(node)] for i in valid_nodes]
        l = str(node)
        key_init_string='[' +p0 + ', ' + l + ', ' + '0]'
        r+=lpvariables_init[key_init_string]
        counter=0
        for lpvariables_ in lpvariables_per_phase:
            valid_edges_tmp = valid_edges.copy()
            for edge in valid_edges_tmp:
                if len(edge) >2:
                    edge.pop(2)
                edge = edge.insert(2,counter+1)
            valid_edges_keys = [str(element) for element in valid_edges_tmp]
            lpv_edges_phase = lpSum(lpvariables_[i] for i in valid_edges_keys)
            r+=lpv_edges_phase
            counter += 1

    prob += (
        r <= 1,
        "Leaf_Restriction_{l},{n}".format(l=leaf,n=node),
    )

# 6) If we choose an edge at phase K, next phase must include the last node in the edge, others will
#   be invalid edges.

for element in variables_init:
    initial_pos_var = lpvariables_init[element]
    valid_input_edge = GDN(element)[1]
    for j in range(0, N ):
        sum = initial_pos_var
        keys = []
        for element_ in lpvariables_per_phase[j]: # Phase K+1 = 1
            valid_input_edge_ = GDN(element_)[0]
            if int(valid_input_edge) != int(valid_input_edge_): # Restriction over other nodes
                    keys.append(element_)
        sum+=lpSum(lpvariables_per_phase[j][i] for i in keys)
        prob+=(
            sum <= 1,
            "Initial_Continuity_Restriction_{l}_{p}".format(l=element,p=j),
        )


# Now for next consecutive phases
for node in Nodes:
    for phase in range(0, N-1):
        keys_k=[]
        keys_kp1=[]
        sum=0
        # Sum variables that end in node v at phase K
        for item in items_per_phase[phase]:
            valid_input_edge = GDN(item)[1]
            if int(valid_input_edge) == int(node):
                keys_k.append(item)
        sum += lpSum(lpvariables_per_phase[phase][i] for i in keys_k)
        # Sum all variables that not start at v at phase k+1
        for item_ in lpvariables_per_phase[phase+1]:
            valid_input_edge_ = GDN(item_)[0]
            if int(node) != int(valid_input_edge_):  # Restriction over other nodes
                keys_kp1.append(item_)
        sum += lpSum(lpvariables_per_phase[phase+1][j] for j in keys_kp1)
        prob += (
            sum <= 1,
            "Continuity_Restriction_{p},{n}".format(p=node, n=phase),
        )


# 7) We force solution to start at Agent initial position
force_init = lpSum(lpvariables_init[i] for i in variables_init)
force_init_r = lpSum(lpvariables_per_phase[0][i] for i in items_per_phase[0])
prob += (
                force_init >= force_init_r,
                "Force_Initial",
            )

# Force consecutive phases
for phase in range(0,N-1):
    force_k=lpSum(lpvariables_per_phase[phase][i] for i in items_per_phase[phase])
    force_k_p_1 = lpSum(lpvariables_per_phase[phase+1][i] for i in items_per_phase[phase+1])
    prob += (
        force_k >= force_k_p_1,
        "Force_next_{p}".format(p=phase),
    )

##################################

# The problem data is written to an .lp file
prob.writeLP("MFFP_Tree.lp")

# The problem is solved using PuLP's choice of Solver (Default is CBC: Coin or branch and cut)
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
solution = {}
for v in prob.variables():
    #print(v.name, "=", v.varValue)
    solution[v.name] = v.varValue

# The optimised objective function value is printed to the screen
print("Total Saved Trees = ", value(prob.objective))

# Nodes that are defended during solution
sol_nodes=[k for k,v in solution.items() if v == 1]

print(sol_nodes)



