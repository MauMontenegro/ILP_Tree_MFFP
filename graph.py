from manim import *
import networkx as nx
import json
import numpy as np



class graphAnimator(Scene):
    def construct(self):
        # Get Original Tree Structure
        T = nx.read_adjlist("FF_Tree.adjlist")
        # Initial Node Fire
        initial_fire = 0
        T = nx.bfs_tree(T, str(initial_fire))

        # Get layout to draw
        lay = open('layout.json')
        layout = json.load(lay)
        # Get Solution provided By ILP Solver
        sol= open('solution.json')
        solution = json.load(sol)
        # Defended Nodes by Time
        sol_saved = solution[0]
        #Burned Nodes by Time
        sol_burned = solution[1]

        #Get indexes for nodes in Layout
        nodes_index=[i for i in layout.keys()]

        points = []
        lines=[]

        self.Presentation()
        self.ProblemDef()

        # For each node in Layout obtain coordinates (x,y,z)
        for node in layout:
            x = (3*layout[node][0])
            y = (3*layout[node][1]) +2
            z = 0
            points.append(Dot(np.array([x, y, z])).set_color(GREEN_A))

        # Obtain points to connect nodes with edges
        for edge in T.edges:
            p0=edge[0]
            p1=edge[1]
            idx_p0=nodes_index.index(str(p0))
            idx_p1 = nodes_index.index(str(p1))
            p0 = points[int(idx_p0)]
            p1 = points[int(idx_p1)]
            lines.append(Line(p0,p1).set_color(GREEN_A))

        self.add(*points)
        self.play(FadeIn(*points))
        self.add(*lines)
        self.play(FadeIn(*lines))
        self.wait(1)

        # Add Initial Fire
        idx_fire = nodes_index.index(str(initial_fire))
        p_fire=points[int(idx_fire)]

        self.add(p_fire.set_fill(RED, opacity=1))
        self.wait(1)
        self.Solution(0, 0, 1)
        self.wait(2)
        burned=0
        #Add Saved Nodes and Burned Nodes by time
        for i in range(1, len(sol_saved)+1):
            burned += len(sol_burned[str(i)])
            idx_saved = nodes_index.index(str(sol_saved[i-1]))
            p_saved = points[int(idx_saved)]
            vg_burned_vertices = VGroup()
            vg_burned_edges = VGroup()
            # Add a burning edge to saved node
            father = [pred for pred in T.predecessors(str(sol_saved[i-1]))]
            idx_p0 = nodes_index.index(str(father[0]))
            idx_p1 = nodes_index.index(str(sol_saved[i-1]))
            p0 = points[int(idx_p0)]
            p1 = points[int(idx_p1)]
            fire_edge = Line(p0, p1)
            fire_edge.set_color(ORANGE)
            vg_burned_edges.add(fire_edge)
            if str(i) in sol_burned:
                for burn_node in sol_burned[str(i)]:
                    idx_burned = nodes_index.index(str(burn_node))
                    # Find parent of burned node
                    father = [pred for pred in T.predecessors(str(burn_node))]
                    idx_p0 = nodes_index.index(str(father[0]))
                    idx_p1 = nodes_index.index(str(burn_node))
                    p0 = points[int(idx_p0)]
                    p1 = points[int(idx_p1)]
                    fire_edge = Line(p0, p1)
                    fire_edge.set_color(ORANGE)
                    vg_burned_edges.add(fire_edge)
                    fire = points[int(idx_burned)]
                    fire.set_fill(RED, opacity=1)
                    vg_burned_vertices.add(fire)
            self.add(p_saved.set_fill(BLUE, opacity=1))
            self.wait(2)
            self.add(vg_burned_vertices)
            self.wait(2)
            self.play(Create(vg_burned_edges))
            self.wait(1)
            self.Solution(i, sol_saved[i - 1], burned)
            self.wait(2)

    def Presentation(self):
        text = Text("FireFighter Problem In Trees", font_size=30)
        text.set_color(RED)
        self.play(FadeIn(text))
        self.wait(3)
        self.play(FadeOut(text))
        self.wait(1)

    def ProblemDef(self):
        text=Text("Problem Definition",font_size=20)
        text.move_to(UP*3.3)
        self.play(FadeIn(text))
        self.wait(5)
        definition_1=Tex(r'Given a rooted Tree T(V,E,r), where $r \in V$ is the root of the fire, which is labeled as $burned$ at time $t=0$ and $d$ is a firefighter \\'
                       r' that can defend only one vertex $v \in V$ at time. ',font_size=20)


        definition_2 = Tex(r'At each time step $t>0$ the firefighter chooses a vertex $i$ to defend and is labeled as $defended$. All of \\'
                       r'his descendants T[i] are now saved and this sub-tree is out of the fire dynamic propagation.', font_size=20)

        definition_3 = Tex(r'After this, fire is propagated to all the neighbors labeled with $burned$ N(v=burned). \\'
                       r'This dynamic ends when $t= h(T,r)$ where $h(T,r)$ is the height of the Tree rooted at $r$ \\'
                       r'or fire can not propagate anymore.',font_size=20)

        definition_4 = Tex(r'Our Goal is to give a defence sequence $\phi={v_1,v_2,...,v_T}$ of vertices such that \\'
                       r'$max \sum_{i=1} |T[v_i]|$.',font_size=20)

        definition_1.move_to(UP * 2.5)
        self.play(FadeIn(definition_1))
        self.wait(3)
        definition_2.move_to(UP * 1)
        self.play(FadeIn(definition_2))
        self.wait(3)
        definition_3.move_to(UP * -1)
        self.play(FadeIn(definition_3))
        self.wait(3)
        definition_4.move_to(UP * -2.5)
        self.play(FadeIn(definition_4))
        self.wait(3)
        self.play(FadeOut(text))
        self.play(FadeOut(definition_1))
        self.play(FadeOut(definition_2))
        self.play(FadeOut(definition_3))
        self.play(FadeOut(definition_4))

    def Solution(self,time,def_node,burned_nodes):

        text = MathTex(r"t :" + str(time), font_size=20)
        text.move_to(DOWN * 0.5)
        text2 = MathTex(r"Saved Node : " + str(def_node), font_size=20)
        text2.move_to(DOWN*1)
        text3 = MathTex(r" Burned Nodes : " + str(burned_nodes), font_size=20)
        text3.move_to(DOWN * 1.5)

        txtvg = VGroup()
        txtvg.add(text, text2, text3)
        txtvg.move_to(LEFT*2)
        self.add(txtvg)
        self.wait(2)
        self.play(FadeOut(text,text2,text3))
