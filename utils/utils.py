import argparse
import re
import networkx as nx
import numpy as np
import json
import random as rnd
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
import tracemalloc
import time as tm
import os

def argParser(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
        List of Available Integer Programming Solutions:
            -IQP
            -ILP
            ''',
        epilog='''python ipromoff.py -s ILP_MFF -m batch -l True -l True'''
    )

    parser.add_argument(
        '--solver', '-s', type=str,
        help="Type of Solver for MFFP (IQP or ILP).")
    parser.add_argument(
        '--mode', '-m', type=str,
        help="Batch instances for stats or only one for testing")
    parser.add_argument(
        '--load', '-l', type=str,
        help="True for load instance or False to create one")
    parser.add_argument(
        '--config', '-c', type=str,
        help="Solver config file")

    return parser.parse_known_args(args)[0]


def createSolver(solver):
    import solvers as solvers
    target_class = solver
    if hasattr(solvers, target_class):
        inputClass = getattr(solvers, target_class)
    else:
        raise AssertionError('There is no implemented Solver called {}'.format(target_class))
    return inputClass

def GDN(dv_string):
    new_string = re.sub(r"[\[]", "", dv_string)
    new_string = re.sub(r"[\]]", "", new_string)
    split_key = new_string.split(",")
    return split_key

def generateInstance(load,path,directory):
    # Return [Tree,s_fire,Dist_Matrix,seed,scale,ax_pos,ay_pos]
    if load:
        print(path)
        T = nx.read_adjlist(path + "/"+ directory +"/MFF_Tree.adjlist")
        # Relabeling Nodes
        mapping = {}
        for node in T.nodes:
            mapping[node] = int(node)
        T = nx.relabel_nodes(T, mapping)
        T_Ad_Sym = np.load(path + "/" + directory +"/FDM_MFFP.npy")
        lay = open(path + "/" + directory +"/layout_MFF.json")
        pos = {}
        pos_ = json.load(lay)

        for position in pos_:
            pos[int(position)] = pos_[position]
        # Get Instance Parameters
        p = open(path + "/" + directory +"/instance_info.json")
        parameters = json.load(p)
        N = parameters["N"]
        seed = parameters["seed"]
        scale = parameters["scale"]
        starting_fire = parameters["start_fire"]

        a_x_pos = parameters["a_pos_x"]
        a_y_pos = parameters["a_pos_y"]

        T = nx.bfs_tree(T, starting_fire)
        T.add_node(N)

        degrees = T.degree()
        max_degree = max(j for (i, j) in degrees)
        root_degree = T.degree[starting_fire]

        # pos[N] = [a_x_pos, a_y_pos]
        #nx.draw_networkx(T, pos=pos)
        #nx.draw_networkx_nodes(T, pos, T.nodes, node_color="tab:red")
        # plt.show()
        #plt.savefig("Graph_Test.png")

        return T, N, starting_fire, T_Ad_Sym, seed, scale, a_x_pos, a_y_pos, max_degree, root_degree

    else:
        # Generate Random Tree with initial fire_root
        N = 100  # Number of Nodes
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

def getExpConfig(name, defpath=None):
    if defpath is None:
        path = Path.cwd() / 'config'
    else:
        path = Path(defpath)
    pathFile = path / (name.strip() + '.yaml')

    if not pathFile.exists() or not pathFile.is_file():
        raise ValueError('Config Path either does not exists or is not a File')

    config = yaml.safe_load(pathFile.open('r'))

    return config

def generateGraph(config):
    threads = config['experiment']['threads']
    nfilestart = config['experiment']['nodefilestart']

    path_ILP="exp_results_ILP_" + str(threads) + '_' + str(nfilestart) + '.npy'
    path_IQP="exp_results_IQP_" + str(threads) + '_' + str(nfilestart) + '.npy'
    ILP = np.load(path_ILP,allow_pickle=True)
    IQP = np.load(path_IQP,allow_pickle=True)

    print('ILP times:')
    print(ILP)
    print('IQP times')
    print(IQP)

    fig = plt.figure(figsize=(20, 10))
    rows = 1
    columns = 3 # Graphs for Time, variables and restrictions
    print(IQP[2])
    node_instances=[10,20,30,40]
    #Plotting Time Solutions
    fig.add_subplot(rows,columns,1)
    plt.plot(ILP[0], 'o-',color='green',label='ILP Solver')
    plt.plot(IQP[0], 'o-',color='blue',label='IQP Solver')
    plt.legend(loc="upper left")
    plt.xlabel("Node Instances")
    plt.ylabel("Time in seconds")
    plt.title("Times")
    plt.grid()

    #Plotting Number of Variables
    fig.add_subplot(rows, columns, 2)
    plt.plot(ILP[2],'o-', color='green', label='ILP Solver')
    plt.plot(IQP[2],'o-', color='blue', label='IQP Solver')
    plt.legend(loc="upper left")
    plt.xlabel("Node Instances")
    plt.ylabel("Decision Variables")
    plt.title("D.V.")
    plt.grid()

    #Plotting Number of Restrictions
    fig.add_subplot(rows, columns, 3)
    plt.plot(ILP[3],'o-', color='green', label='ILP Solver')
    plt.plot(IQP[3],'o-', color='blue', label='IQP Solver')
    plt.legend(loc="upper left")
    plt.xlabel("Node Instances")
    plt.ylabel("Restrictions")
    plt.title("Restrictions")
    plt.grid()

    plt.savefig("ILP_IQP_{t}_{n}".format(t=threads, n=nfilestart), format='png')

def generateGraphSeeds(config):
    path=os.walk("Results")
    for root, directories, files in path:
        print(directories)


# Performance
def tracing_start():
    tracemalloc.stop()
    print("nTracing Status : ", tracemalloc.is_tracing())
    tracemalloc.start()
    print("Tracing Status : ", tracemalloc.is_tracing())


def tracing_mem():
    first_size, first_peak = tracemalloc.get_traced_memory()
    peak = first_peak / (1024 * 1024)
    print("Peak Size in MB - ", peak)
    return peak