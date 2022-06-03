"""
Moving Firefighter Problem on Trees
Integer Quadratic Programming Solution
Author: Mauro Alejandro Montenegro Meza
"""

import numpy
from gurobipy import *
import networkx as nx
import numpy as np
from utils.utils import generateInstance
import os
from pathlib import Path

BN = 10000  # Big Number for Restrictions

class IQP_MFF():
    def __init__(self, mode, load, path,config):
        self.config = config
        self.mode = mode
        self.load = load
        self.times = []
        self.solutions = []
        self.saved = []
        if self.mode == 'batch':
            self.w_path = os.walk(Path.cwd() / 'Instances')
            self.path = path + '/Instances'
        else:
            self.w_path = os.walk(Path.cwd() / 'Instance')
            self.path = path + '/Instance'

    def solve(self):
        for root, directories, files in self.w_path:
            directories.sort()
            for directory in directories:
                print("\n\nCompute solution for {n}".format(n=directory))
                instance = generateInstance(self.load, self.path, str(directory))

                T = instance[0]
                N = instance[1]
                starting_fire = instance[2]
                T_Ad_Sym = instance[3]
                seed = instance[4]
                scale = instance[5]
                a_x_pos = instance[6]
                a_y_pos = instance[7]

                print("Starting FIre")
                print(instance[2])

                # --- MODEL---
                m = Model("mip1")
                m.Params.outputFlag = 1  # 0 - Off  //  1 - On
                m.setParam("MIPGap", self.config['experiment']['mip_gap'])
                m.setParam("Method", self.config['experiment']['method'])
                m.setParam("Presolve", self.config['experiment']['presolve'])  # -1 - Automatic // 0 - Off // 1 - Conservative // 2 - Aggresive
                m.setParam("NodefileStart", self.config['experiment']['nodefilestart'])
                m.setParam("Threads", self.config['experiment']['threads'])
                # m.setParam("PreQLinearize", -1); # -1 - Automatic // 0 - Off // 1 - Strong LP relaxation // 2 - Compact relaxation
                # m.params.BestObjStop = k

                # ---VARIABLES----
                vars = []
                for i in range(N):
                    temp = []
                    for j in range(N):
                        temp.append(0)
                    vars.append(temp)

                for phase in range(N):
                    for node in range(N):
                        vars[phase][node] = m.addVar(vtype=GRB.BINARY, name="x,%s" % str(phase) + "," + str(node))
                m.update()

                # -------- OBJECTIVE FUNCTION ----------
                Nodes = list(T.nodes)
                Nodes.remove(N)
                Nodes.sort()
                weights = np.zeros(N)
                i = 0

                for node in Nodes:
                    weights[i] = len(nx.descendants(T, node)) + 1
                    i += 1
                weights = np.delete(weights, starting_fire)

                objective = 0
                weights_transpose = np.array(weights).T
                for i in range(N):
                    vars_tmp = np.delete(vars[i], starting_fire)
                    objective += np.dot(weights_transpose, vars_tmp)
                m.setObjective(objective, GRB.MAXIMIZE)

                # ----------------------First Constraint---------------------------------
                m.update()
                sum_vars = 0
                for phase in range(N):
                    sum_vars = 0
                    for node in range(N):
                        sum_vars += vars[phase][node]
                    m.addConstr(sum_vars <= 1)

                # --------------------------Second Constraint--------------------------------

                # Obtain level for each node with starting fire as root
                levels = nx.single_source_shortest_path_length(
                    T, starting_fire
                )
                sorted_burning_times = numpy.zeros(N)

                # Sorted Burnig time for each node (from 0 to N)
                for i in range(N):
                    sorted_burning_times[i] = levels[i]

                # Constraint for initial Position
                initial_const = np.dot(T_Ad_Sym[N, 0:N], vars[0])
                initial_const_ = np.dot(sorted_burning_times.T, vars[0])
                m.addConstr(initial_const <= initial_const_, name="Init_Const")

                for phase in range(1, N):
                    q_1 = 0
                    for phase_range in range(0, phase):
                        for node_i in range(N):
                            for node_j in range(N):
                                q_1 += T_Ad_Sym[node_i][node_j] * (
                                        vars[phase_range][node_i] * vars[phase_range + 1][node_j])
                    q_1 += initial_const

                    q_2 = np.dot(sorted_burning_times.T, vars[phase])
                    d = 0
                    for node in range(N):
                        d += vars[phase][node]
                    d = BN * (1 - d)
                    q_2 += d

                    m.addConstr(q_1 <= q_2, name="Q,%s" % str(phase))

                # ----------------------Third Constraint --------------------------
                leaf_nodes = [
                    node for node in T.nodes() if T.in_degree(node) != 0 and T.out_degree(node) == 0
                ]

                restricted_ancestors = {}
                for leaf in leaf_nodes:
                    restricted_ancestors[leaf] = list(nx.ancestors(T, leaf))
                    restricted_ancestors[leaf].remove(starting_fire)
                    restricted_ancestors[leaf].insert(0, leaf)

                for leaf in restricted_ancestors:
                    l_q = 0
                    for node in restricted_ancestors[leaf]:
                        for phase in range(N):
                            l_q += vars[phase][node]
                    m.addConstr(l_q <= 1)

                # ----------------------------------Fourth Constrain-----------------------------------------
                # Force Consecutive Node Strategy
                for phase in range(N - 1):
                    vn_ = 0
                    v_ = 0
                    for node in range(N):
                        v_ += vars[phase][node]
                        vn_ += vars[phase + 1][node]
                    m.addConstr(v_ >= vn_)

                # ----------------- Optimize Step--------------------------------
                m.optimize()
                runtime = m.Runtime
                print("The run time is %f" % runtime)
                print("Obj:", m.ObjVal)
                self.saved.append(m.ObjVal)
                self.times.append(runtime)
                sol=[]
                for v in m.getVars():
                    if v.X > 0:
                        sol.append(v)
                self.solutions.append(sol)

    def getSolution(self):
        return self.solutions

    def getTimes(self):
        return self.times

    def getSaved(self):
        return self.saved

    def saveSolution(self):
        threads = self.config['experiment']['threads']
        nfilestart = self.config['experiment']['nodefilestart']
        name = 'exp_results_IQP_' + str(threads) +'_'+ str(nfilestart)
        numpy.save(name, numpy.array([self.times, self.saved]))
