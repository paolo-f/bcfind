import math
import networkx


def distance((x1,y1,z1),(x2,y2,z2)):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def match_markers(C1,C2, max_distance):
    """Match true and predicted markers using max-weight bipartite matching

    Parameters
    ----------
    C1 : list
      first list of markers
    C2 : list
      second list of markers
    max_distance : float
      distance below which to markers are never matched

    return:
      G : bipartite graph
      mate: list of matches
      node2center: dict mapping graph nodes to markers
    """

    G = networkx.Graph()
    node2center = {}
    for i,c in enumerate(C1):
        node = 't_%d' % i
        G.add_node(node, x=c.x, y=c.y, z=c.z, label=c.name)
        node2center[node] = c
    for i,c in enumerate(C2):
        node = 'p_%d' % i
        G.add_node(node, x=c.x, y=c.y, z=c.z, label=c.name)
        node2center[node] = c
    # print("Computing pairwise distances")
    for ni in [n for n in G.nodes() if n[0] == 't']:
        for nj in [n for n in G.nodes() if n[0] == 'p']:
            d = distance((G.node[ni]['x'],G.node[ni]['y'],G.node[ni]['z']),
                         (G.node[nj]['x'],G.node[nj]['y'],G.node[nj]['z']))
            if d < max_distance:
                w = 1.0/max(0.001,d)
                G.add_edge(ni,nj,weight=w)
    print("Solving max weight matching (%d nodes, %d edges)" % (len(G.nodes()), len(G.edges())))
    mate = networkx.algorithms.matching.max_weight_matching(G,maxcardinality=False)
    return G,mate,node2center


