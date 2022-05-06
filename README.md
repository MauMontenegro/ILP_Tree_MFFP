# Moving Fire Fighter Problem: Integer Lineal Programming Solution for Trees

The Moving Firefighter problem (MFFP) is a generalization of the original Moving Firefighter problem proposed by Bert Hartnell in 1995 which 
consists in a defense strategy against a propagation model (fire, flood, infectious disease, etc.) in a graph.
Given a graph *G=(E,V)* , a set of starting fire nodes $f \in V$, and a initial agent position *p*, at each consecutive discrete time step, 
the fire propagates from each burning vertex to all neighbor nodes unless they are defended by the agent. Once a vertex is burned or defended, they will
reamin on this state until fire can no longer spread. On the original problem the agent can move instantaneously between each pair of nodes, on *MFFP*
we have a notion of distance or cost to travel making the problem more restrictive.
