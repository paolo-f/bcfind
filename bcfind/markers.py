import math
import networkx
from scipy.spatial.distance import cdist 
import numpy as np

from bcfind.volume import Center
from clsm_registration.estimate_registration import horn_method

def distance((x1,y1,z1),(x2,y2,z2)):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)


def match_markers(C1,C2, max_distance,verbose=False):
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
    if verbose:
	print("Solving max weight matching (%d nodes, %d edges)" % (len(G.nodes()), len(G.edges())))
    mate = networkx.algorithms.matching.max_weight_matching(G,maxcardinality=False)
    return G,mate,node2center


def compute_matches(c1,c2,max_distance):
    """
    Method that finds all the pairs of points from two arrays of shape (n_points, 3) which have a distance below
    a maximum value.

    Parameters
    ----------
    c1 : numpy array of shape (n_points, 3)
	source point cloud
    c2 : numpy array of shape (n_points, 3)
	target point cloud
    max_distance : float
      distance below which to markers are never matched

    Returns
    -------
    good1 : list
      list of indices of rows of the first array that have been matched with the second array
    good1 : list
      list of indices of rows of the second array that have been matched with the first array
    distances: numpy array of shape (n_points,1)
      array of distances from all the points of the first array to the closest one of the second array

    References
    ----------
    Simon, D.A.: Fast and accurate shape-based registration. Ph.D. thesis, Pittsburgh, PA, USA
    (1996)
    """
    dist_matrix = cdist(c1, c2, 'euclidean') 
    range_col=np.arange(len(c2),dtype=int)
    range_row=np.arange(len(c1),dtype=int)
    good1=[]
    good2=[]
    distances=[]
    end_match=False
    j=min(len(c1),len(c2))
    while not end_match and j>0:
        min_dist=np.amin(dist_matrix)
        if min_dist < max_distance:
            ind_row=int(np.where(dist_matrix==min_dist)[0][0])
            ind_col=int(np.where(dist_matrix==min_dist)[1][0])
            dist_matrix[ind_row,:]=max_distance
            dist_matrix[:,ind_col]=max_distance
            good1.append(ind_row)
            good2.append(ind_col)
	    distances.append(min_dist)
        else:
            end_match=True
        j-=1

    return good1,good2,distances 
    

def match_markers_with_icp(C1,C2, max_distance,num_iterations = 100, eps=1e-8, verbose=False):
    """
    Method that matches two point clouds using an implementation of the Iterative Closest Point (ICP)
    procedure. 

    Parameters
    ----------
    C1 : list
      first list of markers
    C2 : list
      second list of markers
    max_distance : float
      distance below which to markers are never matched
    num_iterations : int
      number of iterations of ICP procedure (Default: 100)
    eps : float
      maximum allowable difference between two consecutive transformations (Default: 1e-8)

    Returns
    -------
    C2_t : list
      transformed second list of markers
    good1 : list
      list of indices of the points of the first cloud that have been matched with the second cloud
    good1 : list
      list of indices of the points of the second cloud that have been matched with the first cloud
    R : numpy array of shape (3, 3)
      rotational component of the estimated rigid transformation
    t : numpy array of shape (3)
      translational component of the estimated rigid transformation

    References
    ----------
    Simon, D.A.: Fast and accurate shape-based registration. Ph.D. thesis, Pittsburgh, PA, USA
    (1996)
    """
    c1 = np.array([[c.x, c.y, c.z] for c in C1])
    c2 = np.array([[c.x, c.y, c.z] for c in C2])

    if c1.ndim == 1:
        c1=np.expand_dims(c1, axis=0)
    if c2.ndim == 1:
        c2=np.expand_dims(c2, axis=0)

    good1,good2,distances = compute_matches(c1,c2,max_distance)
    c1_good = c1[good1]
    c2_good = c2[good2]

    R=np.eye(3)
    t=np.zeros(3)
    hom_matrix=np.vstack((np.hstack((R,np.expand_dims(t, axis=0).T)),np.array([0.,0.,0.,1.])))
    hom_matrix_tot=hom_matrix
    for i in xrange(num_iterations):

        hom_previous_matrix=hom_matrix
        
        weights=1./(np.array(distances)+1) 
        if len(c2_good) >3.:
            R,t,_,_,_ =horn_method(c2_good, c1_good, weights)
        else:
            break
        c2=np.dot(R,c2.T).T+t
        hom_matrix=np.vstack((np.hstack((R,np.expand_dims(t, axis=0).T)),np.array([0.,0.,0.,1.])))
        hom_matrix_tot=np.dot(hom_matrix_tot,hom_matrix)
        incr_froeb_diff=np.linalg.norm(hom_matrix - hom_previous_matrix)
        good1,good2,distances = compute_matches(c1,c2,max_distance)
        c1_good = c1[good1]
        c2_good = c2[good2] 
        if verbose:
            print('iter_icp:%d R:%s t:%s froeb_norm:%f'%(i,R,t,incr_froeb_diff))
        if incr_froeb_diff <= eps:
            break

    R = hom_matrix_tot[0:3,0:3]
    t = hom_matrix_tot[0:3,3]
    C2_t=[]
    for c,C in zip(c2,C2):
        c_t=Center(c[0],c[1],c[2])
        c_t.name=C.name
        c_t.hue=C.hue
        C2_t.append(c_t)

    return C2_t,good1,good2,R,t




