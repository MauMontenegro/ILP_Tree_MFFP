import re
import networkx as nx
import numpy as np
import json
import random as rnd
import matplotlib.pyplot as plt


# Function that split string of [u,v,p]
# where (u,v) is an edge and p is the phase
def GDN(dv_string):
    new_string = re.sub(r"[\[]", "", dv_string)
    new_string = re.sub(r"[\]]", "", new_string)
    split_key = new_string.split(",")
    return split_key


def generateInstance(load,directory):
    # Return [Tree,s_fire,Dist_Matrix,seed,scale,ax_pos,ay_pos]
    if load:
        T = nx.read_adjlist("Instances/"+ directory +"/MFF_Tree.adjlist")
        # Relabeling Nodes
        mapping = {}
        for node in T.nodes:
            mapping[node] = int(node)
        T = nx.relabel_nodes(T, mapping)
        T_Ad_Sym = np.load("Instances/"+ directory +"/FDM_MFFP.npy")
        lay = open("Instances/"+ directory +"/layout_MFF.json")
        pos = {}
        pos_ = json.load(lay)

        for position in pos_:
            pos[int(position)] = pos_[position]
        # Get Instance Parameters
        p = open("Instances/"+ directory +"/instance_info.json")
        parameters = json.load(p)
        N = parameters["N"]
        seed = parameters["seed"]
        scale = parameters["scale"]
        starting_fire = parameters["start_fire"]

        a_x_pos = parameters["a_pos_x"]
        a_y_pos = parameters["a_pos_y"]

        T = nx.bfs_tree(T, starting_fire)
        T.add_node(N)

        # pos[N] = [a_x_pos, a_y_pos]
        #nx.draw_networkx(T, pos=pos)
        #nx.draw_networkx_nodes(T, pos, T.nodes, node_color="tab:red")
        # plt.show()
        #plt.savefig("Graph_Test.png")

        return T, N, starting_fire, T_Ad_Sym, seed, scale, a_x_pos, a_y_pos

    else:
        # Generate Random Tree with initial fire_root
        N = 10  # Number of Nodes
        seed = 150  # Experiment Seed
        scale = 1  # Scale of distances
        starting_fire = rnd.randint(0, N - 1)  # Fire in random node
        print("Starting fire in Node: {sf}".format(sf=starting_fire))

        # Adding Bulldozer
        a_x_pos = rnd.uniform(-1, 1) * scale
        a_y_pos = rnd.uniform(-1, 1) * scale
        print("Initial Bulldozer Position: [{bx},{by}]".format(bx=a_x_pos, by=a_y_pos))

        # Create a Random Tree (nx use a Prufer Sequence) and get 'pos' layout for nodes
        T = nx.random_tree(n=N, seed=seed,create_using=nx.erdos_renyi_graph(N, 0.5, seed=None, directed=False))
        # Induce a BFS path to get the fire propagation among levels
        T = nx.bfs_tree(T, starting_fire)
        # Could use spring or spectral Layout
        pos = nx.spring_layout(T, seed=seed)

        # Create Empty Adjacency Matrix for Full Distances
        T_Ad = np.zeros((N + 1, N + 1))

        # Save Original Tree in a "adjlist" file
        #nx.write_adjlist(T, "MFF_Tree.adjlist")
        #nx.draw_networkx(T, pos=pos)
        # plt.show()
        #plt.savefig("Graph_Test.png")

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

        # Scale Distances in Layout to better plot
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

        # Saving Full Distance Matrix
        f = open("Full_Matrix.txt", "w")
        f.write(str(T_Ad_Sym))

        # Just Showing NX Tree
        nx.draw_networkx(T, pos=pos, with_labels=True)
        # plt.show()
        plt.savefig("Graph_Test.png")

        # Saving Layout in to a json file
        for element in pos:
            pos[element] = list(pos[element])

        with open("layout_MFF.json", "w") as layout_file:
            layout_file.write(json.dumps(pos))

        return T, N, starting_fire, T_Ad_Sym, seed, scale, a_x_pos, a_y_pos
