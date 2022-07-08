"""
Moving Firefighter Problem on Trees
Integer Linear Programming Solution
Author: Mauro Alejandro Montenegro Meza
"""
import numpy
import pulp as pl
import networkx as nx
from utils.utils import GDN
from utils.utils import generateInstance
from utils.utils import tracing_start
from utils.utils import tracing_mem
from pathlib import Path
import time as tm
from gurobipy import *

BN = 10000  # Big Number for Restrictions
import numpy as np


class ILP_MFF():
    def __init__(self, mode, load, path, config):
        self.config = config
        self.mode = mode
        self.load = load
        self.times = []
        self.solutions = []
        self.saved = []
        self.n_restrictions = []
        self.n_variables = []
        self.root_degree = []
        self.max_degree = []
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
                print("\n\nCompute solution for {n} nodes".format(n=directory))
                instance = generateInstance(self.load, self.path, str(directory))
                T = instance[0]
                N = instance[1]
                starting_fire = instance[2]
                T_Ad_Sym = instance[3]
                seed = instance[4]
                scale = instance[5]
                a_x_pos = instance[6]
                a_y_pos = instance[7]
                self.max_degree.append(instance[8])
                self.root_degree.append(instance[9])

                # Pre-Compute Burning_Times for each node in T
                levels = nx.single_source_shortest_path_length(
                    T, starting_fire
                )  # Obtain Level in Tree for each node

                # --- MODEL---
                m = Model("ILP_FF")
                m.Params.outputFlag = 1  # 0 - Off  //  1 - On
                m.setParam("MIPGap", self.config['experiment']['mip_gap'])
                m.setParam("Method", self.config['experiment']['method'])
                m.setParam("Presolve", self.config['experiment'][
                    'presolve'])  # -1 - Automatic // 0 - Off // 1 - Conservative // 2 - Aggresive
                m.setParam("NodefileStart", self.config['experiment']['nodefilestart'])
                m.setParam("Threads", self.config['experiment']['threads'])

                # ---InitialPos_Node_Variables----
                # (X_phase0_node1), (X_phase0_node2), .... ,(X_phase0_nodeN)
                initial_vars = []
                for i in range(N):
                    initial_vars.append(0)
                    if i != starting_fire:
                        initial_vars[i] = m.addVar(vtype=GRB.BINARY, name="x,%s" % str(0) + "," + str(i))
                m.update()
                print(initial_vars)
                # ---InitialPos_Node_Variables----
                # (X_phase1_node0_node0), (X_phase1_node0_node1), .... ,(X_phase1_node0_nodeN),(X_phase1_node1_node0),...,(X_phase1_nodeN_nodeN)
                # (X_phase2_node0_node0), (X_phase2_node0_node1), ...., (X_phase2_node0_nodeN),(X_phase2_node1_node0),...,(X_phase2_nodeN_nodeN)
                vars = []
                for i in range(N - 1):
                    temp_1 = []
                    for j in range(N):
                        temp_2 = []
                        for k in range(N):
                            temp_2.append(0)
                        temp_1.append(temp_2)
                    vars.append(temp_1)

                for phase in range(N - 1):
                    for node_1 in range(N):
                        for node_2 in range(N):
                            if (node_1 != starting_fire & node_2 != starting_fire):
                                vars[phase][node_1][node_2] = m.addVar(vtype=GRB.BINARY,
                                                                       name="x,%s" % str(phase + 1) + "," + str(
                                                                           node_1) + "," + str(node_2))
                m.update()
                print('Decision Variables No Pulp')
                print(vars)
                # -------- OBJECTIVE FUNCTION ----------
                Nodes = list(T.nodes)
                Nodes.remove(N)
                Nodes.sort()
                weights = np.zeros(N)
                i = 0

                for node in Nodes:
                    weights[i] = len(nx.descendants(T, node)) + 1
                    i += 1

                # Sum initial vars to objective
                objective = 0
                weights_transpose = np.array(weights).T
                objective += np.dot(weights_transpose, initial_vars)

                # Sum rest of variables
                for i in range(N - 1):
                    for j in range(N):
                        w_copy = weights_transpose.copy()
                        w_copy[j] = 0
                        objective += np.dot(w_copy, vars[i][j])
                m.setObjective(objective, GRB.MAXIMIZE)

                count_const = 0
                # ----------SUBJECT TO---------------------
                # Constraint 1
                sum_initial_vars = 0
                for i in range(N):
                    sum_initial_vars += initial_vars[i]
                count_const += 1
                m.addConstr(sum_initial_vars == 1)

                # Constraint 2
                sum_vars = 0
                for phase in range(N-1):
                    sum_vars = 0
                    for node_1 in range(N):
                        for node_2 in range(N):
                          sum_vars += vars[phase][node_1][node_2]
                    count_const += 1
                    m.addConstr(sum_vars == 1)

                # Constraint 3
                levels = nx.single_source_shortest_path_length(
                    T, starting_fire
                )
                sorted_burning_times = numpy.zeros(N)

                # Sorted Burning time for each node (from 0 to N)
                for i in range(N):
                    sorted_burning_times[i] = levels[i]

                # Constraint for initial Position
                initial_time_const = np.dot(T_Ad_Sym[N, 0:N], initial_vars)
                initial_time_const_ = np.dot(sorted_burning_times.T, initial_vars)
                count_const += 1
                m.addConstr(initial_time_const <= initial_time_const_, name="Init_time_Const")

                # Constraint for next phases
                for phase in range(N - 1):
                    q_1 = 0
                    q_2 = 0
                    for phase_range in range(0, phase + 1):
                        for node_i in range(N):
                            for node_j in range(N):
                                q_1 += T_Ad_Sym[node_i][node_j] * vars[phase_range][node_i][node_j]
                    q_1 += initial_time_const
                    for i in range(N):
                        q_2 += np.dot(sorted_burning_times.T, vars[phase][i])
                    count_const += 1
                    m.addConstr(q_1 <= q_2, name="Q,%s" % str(phase))

                # Constraint 4
                leaf_nodes = [node for node in T.nodes() if T.in_degree(node) != 0 and T.out_degree(node) == 0]
                restricted_ancestors = {}
                for leaf in leaf_nodes:
                    restricted_ancestors[leaf] = list(nx.ancestors(T, leaf))
                    restricted_ancestors[leaf].remove(starting_fire)
                    restricted_ancestors[leaf].insert(0, leaf)

                for leaf in restricted_ancestors:
                    l_q = 0
                    for node in restricted_ancestors[leaf]:
                        for phase in range(N - 1):
                            for input_node in range(N):
                                if input_node!= node:
                                    l_q += vars[phase][input_node][node]
                        l_q += initial_vars[node]
                    count_const += 1
                    m.addConstr(l_q <= 1)

                # Constraint 5
                for i in range(N):
                    l_q = 0
                    l_q += initial_vars[i]
                    for j in range(N):
                        for k in range(N):
                            if j != i:
                                l_q += vars[0][j][k]
                    count_const += 1
                    m.addConstr(l_q <= 1)

                for i in range(N):  # For each node v
                    l_q = 0
                    for j in range(N - 2):  # For each phase
                        l_q = 0
                        for k in range(N):  # For each node u
                            l_q += vars[j][k][i]
                        for z in range(N):
                            for p in range(N):
                                if z != i:
                                    l_q += vars[j + 1][z][p]
                        count_const += 1
                        m.addConstr(l_q <= 1)

                # Constraint 6
                c_1 = 0
                c_2 = 0
                for i in range(N):
                    c_1 += initial_vars[i]
                for j in range(N):
                    for k in range(N):
                        c_2 += vars[0][j][k]
                count_const += 1
                m.addConstr(c_1 >= c_2)

                print('Total Constraints')
                print(count_const)
                # ----------------- Optimize Step--------------------------------
                m.optimize()
                runtime = m.Runtime
                print("The run time is %f" % runtime)
                print("Obj:", m.ObjVal)
                self.saved.append(m.ObjVal)
                self.times.append(runtime)
                sol = []
                for v in m.getVars():
                    if v.X > 0:
                        sol.append(v)
                        print(v.varName)
                self.solutions.append(sol)

    def solve_pulp(self):
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
                self.max_degree.append(instance[8])
                self.root_degree.append(instance[9])

                # Check and change LP Solver
                #solver_list = pl.listSolvers(onlyAvailable=True)
                #print(solver_list)
                # Build Node Structure for LP
                Nodes = list(T.nodes)
                Nodes.remove(starting_fire)
                Nodes.remove(N)

                # Pre-Compute Data
                ###########################################################################################################
                # Pre-Compute Burning_Times for each node in T
                levels = nx.single_source_shortest_path_length(
                    T, starting_fire
                )  # Obtain Level in Tree for each node
                ############################################################################################################

                # Create LP Problem
                prob = pl.LpProblem("Moving_Firefighter_Tree", pl.LpMaximize)

                # Create Decision Variables: (X_k,u,v)
                # Create all edges (without initial position)
                all_edges = []
                for node1 in Nodes:
                    for node2 in Nodes:
                        # if node1 != node2:
                        edge = [node1, node2]
                        all_edges.append(edge)
                edge_number = len(all_edges)

                # Create all edges for initial position
                all_initial_edges = []
                for node1 in Nodes:
                    edge = [N, node1]
                    all_initial_edges.append(edge)

                # Create all phases for all edges without initial pos
                variables = []
                # Array Containing all edges per phase
                phases = [x for x in range(1, N + 1)]  # Array of total phases (Max node number)

                for phase in phases:
                    edges_per_phase = {}
                    for edge in all_edges:  # Here, we fill all edges in this phase
                        x = edge.copy()
                        x.append(phase)
                        if edge[0] == edge[1]:
                            edges_per_phase[str(x)] = "f"
                        else:
                            edges_per_phase[str(x)] = edge[1]
                    variables.append(edges_per_phase)

                # Pre-Compute cardinality of a node sub-tree (saved nodes if defended)
                weights = {}
                for node in Nodes:
                    weights[node] = len(nx.descendants(T, node)) + 1
                weights["f"] = 0

                items_per_phase = []
                for phase in phases:
                    items_per_phase.append(variables[phase - 1].keys())

                # Create initial phase for all edges
                variables_init = {}
                for node in Nodes:
                    x = [N, node]
                    x.append(0)
                    variables_init[str(x)] = node
                items_init = variables_init.keys()

                # Create Initial Decision Variables for LP with restrictions in range and type
                lpvariables_init = pl.LpVariable.dicts("Defend", items_init, 0, 1, pl.LpBinary)

                # Create Decision Variables for LP with restrictions in range and type for each phase
                lpvariables_per_phase = []
                for phase in phases:
                    lpvariables = pl.LpVariable.dicts("Defend", items_per_phase[phase - 1], 0, 1, pl.LpBinary)
                    lpvariables_per_phase.append(lpvariables)

                self.n_variables.append((N-1)+(N-1)*(N-1)*N)

                # Sum Decision Variables
                lps = 0
                counter = 0

                lps_init = pl.lpSum(
                    [lpvariables_init[f] * weights[variables_init[f]] for f in variables_init]
                )

                for phase in lpvariables_per_phase:
                    lps += pl.lpSum(
                        [phase[i] * weights[variables[counter][i]] for i in variables[counter]]
                    )
                    counter += 1

                lps_total = lps + lps_init

                # Construct optimization problem
                prob += (
                    lps_total,
                    "Sum_of_Defended_Edges",
                )

                count_const = 0
                # Constraints
                #################################################################################################################
                # 1) At phase 0, we only enable at most one edge to be active from p_0 to any node v

                first_constraint = pl.lpSum([lpvariables_init[i] for i in variables_init])
                count_const += 1
                prob += (
                    first_constraint == 1,
                    "Initial_Edges",
                )

                # 2) From phase 1 to N we only enable at most one edge to be active per phase
                counter = 0
                for lpvariables_ in lpvariables_per_phase:
                    cons = pl.lpSum([lpvariables_[i] for i in variables[counter]])
                    count_const += 1
                    prob += (
                        cons == 1,
                        "Edges_Phase_%s" % counter,
                    )
                    counter += 1

                # 3) At phase 0, we only enable edge transitions that lead B from his initial position p_0
                #    to nodes which B can reach before fire does.
                cons_i = pl.lpSum(
                    [
                        lpvariables_init[i] * T_Ad_Sym[int(GDN(i)[0])][int(GDN(i)[1])]
                        for i in variables_init
                    ]
                )
                cons_d = pl.lpSum([lpvariables_init[i] * levels[int(GDN(i)[1])] for i in variables_init])

                count_const += 1
                prob += (
                    cons_i <= cons_d,
                    "Initial_Distance_Restriction",
                )

                # 4) From phase 1 to n we enable only edges that lead B to valid nodes from his current position. The sum of distances
                #    from p0 to current position following active edges must be less that the time it takes the fire to reach a node from
                #    the nearest fire root.
                # r_init= lpSum([lpvariables_init[i] * T_Ad_Sym[int(GDN(i)[0])][int(GDN(i)[1])] for i in variables_init])
                counter = 0
                dist_r_i = cons_i
                dist_r_d = 0

                for (
                        lpvariables_
                ) in lpvariables_per_phase:  # At each loop sum one new phase (Cumulative)
                    dist_r_i += pl.lpSum(
                        [
                            lpvariables_[i] * T_Ad_Sym[int(GDN(i)[0])][int(GDN(i)[1])]
                            for i in variables[counter]
                        ]
                    )
                    dist_r_d = pl.lpSum(
                        [lpvariables_[i] * levels[int(GDN(i)[1])] for i in variables[counter]]
                    )
                    disable_r = BN * (1 - pl.lpSum([lpvariables_[i] for i in variables[counter]]))
                    dist_r_d += disable_r
                    count_const += 1
                    prob += (
                        dist_r_i <= dist_r_d,
                        "Distance_Restriction_%s" % counter,
                    )
                    counter += 1

                # 5) We only enable one defended node in the path of each leaf to the root
                leaf_nodes = [
                    node for node in T.nodes() if T.in_degree(node) != 0 and T.out_degree(node) == 0
                ]
                restricted_ancestors = {}
                for leaf in leaf_nodes:
                    restricted_ancestors[leaf] = list(nx.ancestors(T, leaf))
                    restricted_ancestors[leaf].remove(starting_fire)
                    restricted_ancestors[leaf].insert(0, leaf)

                p0 = str(N)

                for leaf in restricted_ancestors:
                    #print(restricted_ancestors[leaf])
                    r = 0
                    for node in restricted_ancestors[leaf]:
                        # Generate only edges that goes to 'node'
                        valid_nodes = Nodes.copy()
                        valid_nodes.remove(node)
                        valid_edges = [[int(i), int(node)] for i in valid_nodes]
                        l = str(node)
                        key_init_string = "[" + p0 + ", " + l + ", " + "0]"
                        r += lpvariables_init[key_init_string]
                        counter = 0
                        for lpvariables_ in lpvariables_per_phase:
                            valid_edges_tmp = valid_edges.copy()
                            for edge in valid_edges_tmp:
                                if len(edge) > 2:
                                    edge.pop(2)
                                edge = edge.insert(2, counter + 1)
                            valid_edges_keys = [str(element) for element in valid_edges_tmp]
                            lpv_edges_phase = pl.lpSum(lpvariables_[i] for i in valid_edges_keys)
                            r += lpv_edges_phase
                            counter += 1
                        #print(r)
                    count_const += 1
                    prob += (
                        r <= 1,
                        "Leaf_Restriction_{l},{n}".format(l=leaf, n=node),
                    )

                # 6) If we choose an edge at phase K, next phase must include the last node in the edge, others will
                #   be invalid edges.

                for element in variables_init:
                    initial_pos_var = lpvariables_init[element]
                    valid_input_edge = GDN(element)[1]
                    sum = initial_pos_var
                    keys = []
                    for element_ in lpvariables_per_phase[0]:  # Phase K+1 = 1
                        valid_input_edge_ = GDN(element_)[0]
                        if int(valid_input_edge) != int(
                                valid_input_edge_
                        ):  # Restriction over other nodes
                            keys.append(element_)
                    sum += pl.lpSum(lpvariables_per_phase[0][i] for i in keys)
                    count_const += 1
                    prob += (
                        sum <= 1,
                        "Initial_Continuity_Restriction_{l}".format(l=element),
                    )

                rest = 1
                # Now for next consecutive phases
                for node in Nodes:
                    # print("Analyzing Node {n}".format(n=node))
                    for phase in range(0, N - 1):
                        sum = 0
                        # print("Phase {n}".format(n=phase))
                        keys_k = []
                        keys_kp1 = []
                        # Sum variables that end in node v at phase K
                        for item in items_per_phase[phase]:
                            valid_input_edge = GDN(item)[1]
                            if int(valid_input_edge) == int(node):
                                keys_k.append(item)
                        # print("Actual Phase")
                        sum += pl.lpSum(lpvariables_per_phase[phase][i] for i in keys_k)
                        # print(keys_k)
                        # Sum all variables that not start at v at phase k+1
                        for item_ in lpvariables_per_phase[phase + 1]:
                            valid_input_edge_ = GDN(item_)[0]
                            if int(node) != int(valid_input_edge_):  # Restriction over other nodes
                                keys_kp1.append(item_)
                        # print("Next Phase")
                        # print(keys_kp1)
                        sum += pl.lpSum(lpvariables_per_phase[phase + 1][j] for j in keys_kp1)
                        count_const += 1
                        prob += (
                            sum <= rest,
                            "Continuity_Restriction_{p},{n}".format(p=node, n=phase),
                        )

                # 7) We force solution to start at Agent initial position
                force_init = pl.lpSum(lpvariables_init[i] for i in variables_init)
                force_init_r = pl.lpSum(lpvariables_per_phase[0][i] for i in items_per_phase[0])
                count_const += 1
                prob += (
                    force_init >= force_init_r,
                    "Force_Initial",
                )

                # Force consecutive phases
                # for phase in range(0,N-1):
                #    force_k=lpSum(lpvariables_per_phase[phase][i] for i in items_per_phase[phase])
                #    force_k_p_1 = lpSum(lpvariables_per_phase[phase+1][i] for i in items_per_phase[phase+1])
                #    prob += (
                #        force_k >= force_k_p_1,
                #        "Force_next_{p}".format(p=phase),
                #    )

                ##################################

                # The problem data is written to an .lp file
                prob.writeLP("ILP_Tree.lp")
                # The problem is solved using PuLP's choice of Solver (Default is CBC: Coin or branch and cut)
                solver = pl.GUROBI_CMD(options=[("Method", self.config['experiment']['method']),
                                                ("NodefileStart", self.config['experiment']['nodefilestart']),
                                                ("Threads", self.config['experiment']['threads']),
                                                ("NodefileDir", os.getcwd() + '/' + 'gurobi_log'),
                                                ("PreSparsify", self.config['experiment']['presparsify'])])
                tracing_start()
                start = tm.time()

                prob.solve(solver)

                end = tm.time()
                print("Time elapsed solving model {} milli seconds".format((end - start) * 1000))
                peak = tracing_mem()
                print("Memory COnsumed by Model solver: {}".format(peak))

                print("Solution Time")
                print(prob.solutionTime)
                self.times.append(prob.solutionTime)

                # The status of the solution is printed to the screen
                print("Status:", pl.LpStatus[prob.status])

                # Each of the variables is printed with it's resolved optimum value
                solution = {}
                for v in prob.variables():
                    # print(v.name, "=", v.varValue)
                    solution[v.name] = v.varValue

                # The optimised objective function value is printed to the screen
                print("Total Saved Trees = ", pl.value(prob.objective))
                self.saved.append(pl.value(prob.objective))

                # Nodes that are defended during solution
                sol_nodes = [k for k, v in solution.items() if v == 1]

                s = {}
                for u_v_x in sol_nodes:
                    x_ = GDN(u_v_x)
                    x = re.sub(r"[\_]", "", x_[2])
                    s[u_v_x] = int(x)

                sorted_sol = sorted(s.items(), key=operator.itemgetter(1))
                self.solutions.append(sorted_sol)
                self.n_restrictions.append(count_const)

            print('Total Constraints')
            print(self.n_restrictions)
            print('Total Variables')
            print(self.n_variables)

    def getSolution(self):
        return self.solutions

    def getTimes(self):
        return self.times

    def getSaved(self):
        return self.saved

    def getVariables_Restrictions(self):
        return self.n_variables, self.n_restrictions

    def getDegrees(self):
        return self.root_degree, self.max_degree

    def saveSolution(self):
        threads = self.config['experiment']['threads']
        nfilestart = self.config['experiment']['nodefilestart']
        name = 'exp_results_ILP_' + str(threads) + '_' + str(nfilestart)
        print(type(self.saved))
        print(type(self.n_variables))
        print(type(self.n_restrictions))
        numpy.save(name, numpy.array([self.times, self.saved, self.n_variables, self.n_restrictions,self.root_degree,self.max_degree]))
