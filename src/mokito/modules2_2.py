import numpy as np
import networkx as nx
import matplotlib.cm as cm
import mdtraj as md 
from mdtraj.geometry.alignment import compute_average_structure

from sklearn.cluster import KMeans
from sklearn_extra.cluster import CommonNNClustering
# from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.manifold import MDS

from collections import Counter

from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from scipy.signal import find_peaks
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import scipy

## Modules to load data

def most_frequent(numbers):

    # remove numbers equal to -1
    filtered_numbers = [num for num in numbers if num != -1]
    # Count the frequency of each number in the list
    counts = Counter(filtered_numbers)
    # Find the number with the highest frequency
    most_common = counts.most_common(1)
    # Return the number with the highest frequency
    return most_common[0][0] if most_common else None

def distance_matrix(X, metric='L2norm', R=None, periodic=False):
    """
    Create a symmetric distance matrix where each element (i,j) represents the 
    L2norm between conformation i and conformation j.
    """
    num_conformations = len(X)
    distance_matrix = np.zeros((num_conformations, num_conformations))
    
    for i in range(num_conformations):
        for j in range(i + 1, num_conformations):
            if metric =='L2norm':
                value = np.linalg.norm(X[i] - X[j], axis = None)
        
            distance_matrix[i, j] = value
            distance_matrix[j, i] = value

    return distance_matrix


def distance_matrix2(X, metric='L2norm', R=None, periodic=False):
    """
    Create a symmetric distance matrix where each element (i,j) represents the 
    RMSD between conformation i and conformation j.
    This function needs MDTRAJ
    """
    num_conformations = X.n_frames
    distance_matrix = np.zeros((num_conformations, num_conformations))
    
    for i in range(num_conformations):
        distance_matrix[i] = md.rmsd(X, X, i)

    return distance_matrix



def distance_matrix2(X, metric='L2norm', R=None, periodic=False):
    """
    Create a symmetric distance matrix where each element (i,j) represents the 
    RMSD between conformation i and conformation j.
    This function needs MDTRAJ
    """
    num_conformations = X.n_frames
    distance_matrix = np.zeros((num_conformations, num_conformations))
    
    for i in range(num_conformations):
        distance_matrix[i] = md.rmsd(X, X, i)

    return distance_matrix


def coord_rmsd_matrix(X):

    num_conformations = len(X)
    RMSD = np.zeros((num_conformations, num_conformations))

    for i in range(num_conformations):

        ref = X[i]
        diff = X - ref
        rmsd_vals = np.sqrt(np.mean(diff**2, axis=(1,2)))

        RMSD[i] = rmsd_vals

    
    return RMSD

def angle_rmsd_matrix(X):

    num_conformations = len(X)
    RMSD = np.zeros((num_conformations, num_conformations))

    for i in range(num_conformations):

        ref = X[i]
        diff = X - ref

        diff[:, 1] = np.arctan2(np.sin(diff[:, 1].copy()), np.cos(diff[:, 1].copy()))

        rmsd_vals = np.sqrt(np.mean(diff**2, axis=1))

        RMSD[i] = rmsd_vals

    
    return RMSD
 
###########################################################################################
## Class to store fundamental parameters
class OrganizeData:
    def __init__(self, X0, chi0, MDtraj=None):
        
        """
        OrganizeData(X0, chi0, chit, MDtraj, frame=0)
        """        


        self.N               = X0.shape[0]
        self.Ndims           = X0.shape[1]

        self.X0 = X0
        self.chi0 = chi0
        self.MDtraj = MDtraj

        print("Check shape of input data")
        print("X0.shape   = ", X0.shape)
        print("chi0.shape = ", chi0.shape)

        print("  ")
        
###########################################################################################
## Modules to create intervals
class FindIntervals:
    def __init__(self, data, Nintervals=10, clustering = 'grid'):
    
        """
        ....
        """
        
        chi = np.copy(data.chi0).reshape(-1, 1)
            
        if clustering == 'grid':
            # Divide chi function in Nbins-1
            chi_min     =  np.min(chi) - 0.0001
            chi_max     =  np.max(chi) + 0.0001
            chi_edges   = np.linspace(chi_min, chi_max, Nintervals + 1)
            dchi        = chi_edges[1] - chi_edges[0]
            chi_centers = chi_edges + 0.5 * dchi
            chi_centers = chi_centers[0:-1]
            
            chi_intervals = np.digitize(chi, chi_edges) - 1
            chi_intervals = chi_intervals[:,0]
            labels_clusters, size_intervals = np.unique(chi_intervals, return_counts=True)
            
        elif clustering == 'kmeans': 
            KMintervals             = KMeans(n_clusters=Nintervals).fit(chi)
            chi_centers             = np.copy(KMintervals.cluster_centers_[:,0])
            chi_intervals           = np.copy(KMintervals.labels_)
        
            # Sort centroids
            idx                  = np.argsort(chi_centers)
            chi_centers          = chi_centers[idx]
            
            for nn,n in enumerate(idx):
                chi_intervals[KMintervals.labels_==n] = nn
                
            labels_clusters, size_intervals = np.unique(chi_intervals, return_counts=True)
        
        elif clustering == 'uniform':

            id_chi        = np.column_stack((np.arange(data.N), chi))
            sorted_id_chi = id_chi[id_chi[:,1].argsort()]
            size_intervals   = data.N // Nintervals

            chi_intervals = np.zeros(data.N)
            chi_centers   = np.zeros(Nintervals)

            for n in range(Nintervals):
                idx                            = sorted_id_chi[n*size_intervals:(1+n)*size_intervals,0]
                chi_intervals[idx.astype(int)] = n
                chi_centers[n]                 = np.mean(chi[idx.astype(int)])

        self.chi_centers       =   chi_centers                     # centers of intervals
        self.Nintervals        =   Nintervals                      # Number of intervals
        self.chi_intervals     =   chi_intervals                   # each entry is the interval of a chi-value 

        self.size_intervals     =   size_intervals                   # size of intervals
        
        
###########################################################################################

###########################################################################################
def assign_noisy_states(labels, di):
    svm_classifier = SVC(kernel='rbf', C=1, gamma='scale')

    PWD_0_clutered = di[labels>-1]
    PWD_0_noisy    = di[labels==-1]
    
    svm_classifier.fit(PWD_0_clutered, labels[labels>-1])
    new_assignments = svm_classifier.predict(PWD_0_noisy).astype('int')

    labels[labels==-1]  = new_assignments

    return labels
    

class FindNodes:
    def __init__(self, data, FIs, eps = None, theta = None, algorithm = 'CNNC', metric=None, R=None, periodic=False, with_noise=True):
        
        """
        ......
        """ 

        nodes                   = np.ones(data.N)

        # index of chi clusters per each node
        index_chi_node          = []
        
        # nodes per each chi cluster
        nodes_for_clusters     = np.empty(FIs.Nintervals, dtype=object)

        # Count the nodes
        Nnodes = 0

        # Loop over the intervals
        for i in tqdm(range(FIs.Nintervals)):

            if metric == 'L2norm':
                
                Xi = data.X0[FIs.chi_intervals == i]
                di    = distance_matrix(Xi)
            
            elif metric == 'mdtraj_rmsd':

                Xi = data.MDtraj[FIs.chi_intervals == i]
                di    = distance_matrix2(Xi)
            
            elif metric == 'coord_rmsd':

                Xi = data.X0[FIs.chi_intervals == i]
                di = coord_rmsd_matrix(Xi)

            elif metric == 'angle_rmsd':

                Xi = data.X0[FIs.chi_intervals == i]
                di = angle_rmsd_matrix(Xi)



            if algorithm == 'CNNC':
                
                clustering = CommonNNClustering(eps=eps[i], min_samples=theta[i], n_jobs=-1, metric='precomputed').fit(di) #
                #print(clustering.labels_)
                #print(np.sum(clustering.labels_==-1))
                #print(len(clustering.labels_))


                if np.sum(clustering.labels_==-1) > 0 and with_noise == True:
                    clustering.labels_[clustering.labels_==-1] = np.max(np.unique(clustering.labels_)) + 1

                """
                #If there are not assigned points and these points are fewer than the total points of the interval
                if np.sum(clustering.labels_==-1) > 0 and np.sum(clustering.labels_==-1)<len(clustering.labels_):
                    if len(np.unique(clustering.labels_))==2:
                        clustering.labels_[clustering.labels_==-1] = 0
                    else:
                        clustering.labels_ = assign_noisy_states(clustering.labels_,di)
                """
            elif algorithm == 'DBSCAN':
                clustering = DBSCAN(eps=eps, min_samples=theta, n_jobs=-1, metric='precomputed').fit(di)
            elif algorithm == 'HDBSCAN':
                clustering = HDBSCAN(min_cluster_size=theta[i], metric='precomputed').fit(di) 
            
            # Labels of nodes in cluster i
            nodes_i               = clustering.labels_
            nodes_i[nodes_i>-1]   = nodes_i[nodes_i>-1] + Nnodes

            # Find the indeces of the states in the chi-cluster
            chi_intervals_i        = np.where(FIs.chi_intervals == i)[0]
            nodes[chi_intervals_i] = nodes_i
            
            # Number of not noisy nodes in cluster i
            unique_nodes_i               = np.unique(nodes_i[nodes_i>-1], return_counts=False)
            unique_nodes_i               = unique_nodes_i + Nnodes
            Nnodes_i                     = len(unique_nodes_i)

            # total number of nodes
            Nnodes = Nnodes + Nnodes_i

            # assign to each node the interval i
            for n in range(Nnodes_i):
                index_chi_node.append(i)
        
        index_chi_node = np.asarray(index_chi_node)

        _, nodes_size = np.unique(nodes[nodes>-1], return_counts=True)

        self.Nnodes          = Nnodes
        self.nodes           = nodes

        self.nodes0          = nodes[0:data.N]
        
        self.index_chi_node  = index_chi_node
        self.nodes_size      = nodes_size






###########################################################################################
class BuildAdjacencyMatrix:
    def __init__(self, data, FNs, FIs, include_same_interval=False, metric ='L2norm', r_neighbors = 1, overlap_threshold=0):
        """
        Select a frame 
        Calculate distance between final point ij and all the starting points n
        Assign the node

        max number of neighbors : k
        """
        
        if metric == 'L2norm':
            di    = distance_matrix(data.X0)
        elif metric == 'mdtraj_rmsd':
            di    = distance_matrix2(data.MDtraj)

        rmsd_nodes = np.zeros((FNs.Nnodes,FNs.Nnodes))
        """
        # Average distance between cluster i and j
        # estimated as average distance between each state
        # in cluster i and j
        for i in range(FNs.Nnodes):

            # Distance between all points of cluster i and all the other points
            di_i = di[FNs.nodes==i]

            for j in range(FNs.Nnodes):
                
                # Submatrix that contains distances between all points of i and all points of j
                di_i_j = di_i[:,FNs.nodes==j]
                di_i_j = di_i_j.flatten()
                di_i_j = di_i_j[di_i_j > 0]
                
                avg_di_i_j = np.sum(di_i_j) / len(di_i_j) / 2
                rmsd_nodes[i,j] = avg_di_i_j
        """
        # Average distance between cluster i and j
        # estimated as distance between average structures of the clusters
        for i in range(FNs.Nnodes):

            as_i = compute_average_structure(data.MDtraj[FNs.nodes==i].xyz)
            as_i = md.Trajectory(as_i[None, :, :], data.MDtraj.top)   

            for j in range(FNs.Nnodes):
                
                as_j = compute_average_structure(data.MDtraj[FNs.nodes==j].xyz)
                as_j = md.Trajectory(as_j[None, :, :], data.MDtraj.top)       
                
                rmsd_nodes[i,j] =  md.rmsd(as_i, as_j)

        nbrs = NearestNeighbors(radius = r_neighbors, metric='precomputed').fit(di)
        neighbors = nbrs.radius_neighbors(di, return_distance=False)

        X0 = data.X0
        C = np.eye(FNs.Nnodes)

        for i in range(FIs.Nintervals-1):

            # Identify clusters in interval i and i+1
            clusters_interval      = np.where(FNs.index_chi_node==i)[0]
            clusters_next_interval = np.where(FNs.index_chi_node==i+1)[0]

            # Count how many clusters there are in the interval i and i + 1
            num_clusters_interval      = len(clusters_interval)
            num_clusters_next_interval = len(clusters_next_interval)
            
            
            for n in range(num_clusters_interval):

                ci = clusters_interval[n]

                # select neighbors of cluster i
                neighbors_i = np.concatenate(neighbors[FNs.nodes == ci]) #.flatten()

                if include_same_interval==True:
                                    
                    for m in range(n+1,num_clusters_interval):

                        cj = clusters_interval[m]

                        # select neighbors of cluster j
                        neighbors_j = np.concatenate(neighbors[FNs.nodes == cj]) #.flatten()

                        # Compute the overlap
                        shared_neighbors = np.intersect1d(neighbors_i, neighbors_j)
                        overlap_fraction = len(shared_neighbors) / len(np.union1d(neighbors_i, neighbors_j))

                        # Add an edge if overlap exceeds the threshold
                        if overlap_fraction >= overlap_threshold:
                            C[ci,cj] = overlap_fraction
                            C[cj,ci] = overlap_fraction
                
                for m in range(num_clusters_next_interval):

                    cj = clusters_next_interval[m]

                    # select neighbors of cluster j
                    neighbors_j = np.concatenate(neighbors[FNs.nodes == cj]) #.flatten()

                    # Compute the overlap
                    shared_neighbors = np.intersect1d(neighbors_i, neighbors_j)
                    overlap_fraction = len(shared_neighbors) / len(np.union1d(neighbors_i, neighbors_j))

                    # Add an edge if overlap exceeds the threshold
                    if overlap_fraction >= overlap_threshold:
                        C[ci,cj] = overlap_fraction
                        C[cj,ci] = overlap_fraction

        A = (C > 0).astype(int)

        # direct adjacency matrix
        Ad = np.copy(A)
        # Upper triangular Adjacency matrix
        Ad[np.triu_indices(Ad.shape[0], 0)] = 0

        # Counting matrix
        self.C = C

        # Adjacency matrix
        self.A  = A

        # Direct adjacency matrix
        self.Ad = Ad

        # Prob transition matrix
        P = np.copy(C)
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1
        P = P / row_sums[:, np.newaxis]
        self.P = P
        
        self.rmsd_nodes = rmsd_nodes
###########################################################################################
class BuildAdjacencyMatrix2:
    def __init__(self, data, FNs, FIs, include_same_interval=False, metric ='L2norm', r_neighbors = 1, overlap_threshold=0):
        """
        Select a frame 
        Calculate distance between final point ij and all the starting points n
        Assign the node

        max number of neighbors : k
        """

        rmsd_nodes = np.zeros((FNs.Nnodes,FNs.Nnodes))

        if metric == 'L2norm':
            di    = distance_matrix(data.X0)
            for i in range(FNs.Nnodes):
                avg_i = np.mean(data.X0[FNs.nodes==i], axis=0)
                for j in range(FNs.Nnodes):
                    avg_j = np.mean(data.X0[FNs.nodes==j], axis=0)
                    rmsd_nodes[i,j] = np.linalg.norm(avg_i - avg_j, axis = None)

        elif metric == 'mdtraj_rmsd':
            # Average distance between cluster i and j
            # estimated as distance between average structures of the clusters
            for i in range(FNs.Nnodes):

                as_i = compute_average_structure(data.MDtraj[FNs.nodes==i].xyz)
                as_i = md.Trajectory(as_i[None, :, :], data.MDtraj.top)   

                for j in range(FNs.Nnodes):
                    
                    as_j = compute_average_structure(data.MDtraj[FNs.nodes==j].xyz)
                    as_j = md.Trajectory(as_j[None, :, :], data.MDtraj.top)       
                    
                    rmsd_nodes[i,j] =  md.rmsd(as_i, as_j)

        nbrs = NearestNeighbors(radius = r_neighbors, metric='precomputed').fit(di)
        neighbors = nbrs.radius_neighbors(di, return_distance=False)

        X0 = data.X0
        C = np.eye(FNs.Nnodes)

        for i in range(FIs.Nintervals-1):

            # Identify clusters in interval i and i+1
            clusters_interval      = np.where(FNs.index_chi_node==i)[0]
            clusters_next_interval = np.where(FNs.index_chi_node==i+1)[0]

            # Count how many clusters there are in the interval i and i + 1
            num_clusters_interval      = len(clusters_interval)
            num_clusters_next_interval = len(clusters_next_interval)
                
            for n in range(num_clusters_interval):

                ci = clusters_interval[n]

                # select neighbors of cluster i
                neighbors_i = np.concatenate(neighbors[FNs.nodes == ci]) #.flatten()

                #
                if include_same_interval==True:
                                    
                    for m in range(n+1,num_clusters_interval):

                        cj = clusters_interval[m]

                        # select neighbors of cluster j
                        neighbors_j = np.concatenate(neighbors[FNs.nodes == cj]) #.flatten()

                        # Compute the overlap
                        shared_neighbors = np.intersect1d(neighbors_i, neighbors_j)
                        overlap_fraction = len(shared_neighbors) / len(np.union1d(neighbors_i, neighbors_j))

                        # Add an edge if overlap exceeds the threshold
                        if overlap_fraction >= overlap_threshold:
                            C[ci,cj] = overlap_fraction
                            C[cj,ci] = overlap_fraction
                #
                
                for m in range(num_clusters_next_interval):

                    cj = clusters_next_interval[m]

                    # select neighbors of cluster j
                    neighbors_j = np.concatenate(neighbors[FNs.nodes == cj]) #.flatten()

                    # Compute the overlap
                    shared_neighbors = np.intersect1d(neighbors_i, neighbors_j)
                    overlap_fraction = len(shared_neighbors) / len(np.union1d(neighbors_i, neighbors_j))

                    # Add an edge if overlap exceeds the threshold
                    if overlap_fraction >= overlap_threshold:
                        C[ci,cj] = overlap_fraction
                        C[cj,ci] = overlap_fraction

        A = (C > 0).astype(int)

        # direct adjacency matrix
        Ad = np.copy(A)
        Ad[np.triu_indices(Ad.shape[0], 0)] = 0

        # Counting matrix
        self.C = C

        # Adjacency matrix
        self.A  = A

        # Direct adjacency matrix
        self.Ad = Ad

        # Prob transition matrix
        P = np.copy(C)
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1
        P = P / row_sums[:, np.newaxis]
        self.P = P
        
        self.rmsd_nodes = rmsd_nodes
        """
        mds = MDS(dissimilarity='precomputed')
        embedding = mds.fit_transform(rmsd_nodes)

        if add_center == True:
            embedding = np.vstack((np.array([0,0]), embedding))
            z = 1

        X0 = data.X0
        C = np.eye(FNs.Nnodes + z)

        tri = Delaunay(embedding)

        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    C[simplex[i], simplex[j]] = 1
                    C[simplex[j], simplex[i]] = 1

        if add_center == True:
            C = C[1:, :]
            C = C[:, 1:]

        A = (C > 0).astype(int)

        # direct adjacency matrix
        Ad = np.copy(A)
        Ad[np.triu_indices(Ad.shape[0], 0)] = 0

        # Counting matrix
        self.C = C

        # Adjacency matrix
        self.A  = A

        # Direct adjacency matrix
        self.Ad = Ad

        # Prob transition matrix
        P = np.copy(C)
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1
        P = P / row_sums[:, np.newaxis]
        self.P = P
        
        self.rmsd_nodes = rmsd_nodes
        """

###########################################################################################
class BuildAdjacencyMatrix3:
    def __init__(self, data, FNs, FIs, include_same_interval=False, metric ='L2norm', r_neighbors = 1, overlap_threshold=0):
        """
        Select a frame 
        Calculate distance between final point ij and all the starting points n
        Assign the node

        max number of neighbors : k
        """

        rmsd_nodes = np.zeros((FNs.Nnodes,FNs.Nnodes))

        if metric == 'L2norm':
            di    = distance_matrix(data.X0)
            for i in range(FNs.Nnodes):
                avg_i = np.mean(data.X0[FNs.nodes==i], axis=0)
                for j in range(FNs.Nnodes):
                    avg_j = np.mean(data.X0[FNs.nodes==j], axis=0)
                    rmsd_nodes[i,j] = np.linalg.norm(avg_i - avg_j, axis = None)

        elif metric == 'mdtraj_rmsd':
            # Average distance between cluster i and j
            # estimated as distance between average structures of the clusters
            for i in range(FNs.Nnodes):

                as_i = compute_average_structure(data.MDtraj[FNs.nodes==i].xyz)
                as_i = md.Trajectory(as_i[None, :, :], data.MDtraj.top)   

                for j in range(FNs.Nnodes):
                    
                    as_j = compute_average_structure(data.MDtraj[FNs.nodes==j].xyz)
                    as_j = md.Trajectory(as_j[None, :, :], data.MDtraj.top)       
                    
                    rmsd_nodes[i,j] =  md.rmsd(as_i, as_j)

        nbrs = NearestNeighbors(radius = r_neighbors, metric='precomputed').fit(di)
        neighbors = nbrs.radius_neighbors(di, return_distance=False)

        X0 = data.X0
        C = np.eye(FNs.Nnodes)

        for i in range(FIs.Nintervals-1):

            # Identify clusters in interval i and i+1
            clusters_interval      = np.where(FNs.index_chi_node==i)[0]
            clusters_next_interval = np.where(FNs.index_chi_node==i+1)[0]

            # Count how many clusters there are in the interval i and i + 1
            num_clusters_interval      = len(clusters_interval)
            num_clusters_next_interval = len(clusters_next_interval)
                
            for n in range(num_clusters_interval):

                ci = clusters_interval[n]

                # select neighbors of cluster i
                neighbors_i = np.concatenate(neighbors[FNs.nodes == ci]) #.flatten()

                #

                for m in range(num_clusters_next_interval):

                    cj = clusters_next_interval[m]

                    # select neighbors of cluster j
                    neighbors_j = np.concatenate(neighbors[FNs.nodes == cj]) #.flatten()

                    # Compute the overlap
                    shared_neighbors = np.intersect1d(neighbors_i, neighbors_j)
                    overlap_fraction = len(shared_neighbors) / len(np.union1d(neighbors_i, neighbors_j))

                    # Add an edge if overlap exceeds the threshold
                    if overlap_fraction >= overlap_threshold:
                        C[ci,cj] = overlap_fraction
                        C[cj,ci] = overlap_fraction

        A = (C > 0).astype(int)

        # direct adjacency matrix
        Ad = np.copy(A)
        #Upper triangle of A (to prevent double counting)
        Ad[np.triu_indices(Ad.shape[0], 0)] = 0

        # Counting matrix
        self.C = C

        # Adjacency matrix
        self.A  = A

        # Direct adjacency matrix
        self.Ad = Ad

        # Prob transition matrix
        P = np.copy(C)
        row_sums = P.sum(axis=1)
        row_sums[row_sums == 0] = 1
        P = P / row_sums[:, np.newaxis]
        self.P = P
        
        self.rmsd_nodes = rmsd_nodes
        
###########################################################################################
def ProjectFunctionOntoNodes(data, f0, FNs, ft=None, periodic = False):

    """
    # check dimensions
    print("Your function should be organize as:")
    print("f0.shape = (Npoints, )")
    #print("ft.shape = (Nframes, Npoints, Nfinpoints)")
    print("or")
    print("f0.shape = (Npoints, Nd)")
    #print("ft.shape = (Nframes, Npoints, Nfinpoints, Nd)")
    print("where Nd is the dimensionality of the function.")
    print("For example, if you are interested in a Ramachandran plot Nd = 2")
    print(" ")
    print("Let's see...")
    print("f0.shape = ", f0.shape)
    #print("ft.shape = ", ft.shape)
    """

    if f0.ndim == 1:
        f0 = f0[:,np.newaxis]
        Nd = 1
    else:
        Nd = f0.shape[1]
        

    f = f0
    
    f_nodes    =    np.zeros((FNs.Nnodes,Nd))
    
    for i in range(FNs.Nnodes):
    
        for d in range(Nd):
            if periodic == True:
                fx = np.cos(f[FNs.nodes==i,d])
                fy = np.sin(f[FNs.nodes==i,d])
                
                # Compute average vector
                fx_mean = np.mean(fx)
                fy_mean = np.mean(fy)
                
                # Compute the average angle from the average vector
                f_nodes[i,d] = np.arctan2(fy_mean, fx_mean)
            else:
                f_nodes[i,d] = np.mean(f[FNs.nodes==i,d])

    return f_nodes


        
###########################################################################################
class BuildGraph:
    def __init__(self, FNs, BAM):

        """
        .....
        """

        Nnodes     = FNs.Nnodes
        C          = BAM.C
        P          = BAM.P
        
        #Direc graph
        Gd  = nx.from_numpy_array(BAM.Ad, create_using=nx.MultiDiGraph())
        
        # Initialize the graph

        # Graph with all edges equal to 1
        G  = nx.Graph()
        # Graph with edges equal to number of counts
        GC = nx.Graph()
        # Graph with normalized esges
        GP = nx.Graph()
        
        G.add_nodes_from(range(Nnodes))
        GC.add_nodes_from(range(Nnodes))
        GP.add_nodes_from(range(Nnodes))
        
        for i in range(Nnodes):
            for j in range(i + 1, Nnodes):
                if C[i, j] != 0:
                    GC.add_edge(i, j,  weight=C[i, j])
                    GP.add_edge(i, j,  weight=P[i, j])
                    G.add_edge(i, j, weight=1)
        
        # Extract edges and their weights
        edges   = GP.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]        

        self.Gd      = Gd
        self.G       = G
        self.GC      = GC
        self.GP      = GP
        self.edges   = edges
        self.weights = weights

###########################################################################################
class FindPaths:
    def __init__(self, FNs, BG, BAM, cutoff=10):

        Nnodes = FNs.Nnodes
        G      = BG.G
        Gd     = BG.Gd
        Aw     = BAM.A
        
        imin = 0
        imax = Nnodes-1
        #
        
        #for p in nx.all_shortest_paths(Gd, source=imax, target=imin):
        
        listPaths = []
        
        k = 0
        for p in nx.all_simple_paths(Gd, source=imax, target=imin, cutoff=cutoff):
            listPaths.append(p)
            k = k +1
        
            if k==100:
                break
        
        Npaths = len(listPaths)
        print("Number of paths:", Npaths)
        paths_weights = np.zeros(Npaths)
        
        for i,p in enumerate(listPaths):
            paths_weights[i] = np.sum([Aw[p[n], p[n+1]] for n in range(len(p)-1)])

        # Sort paths according to the weight
        indeces_paths = np.argsort( - paths_weights )

        sortedListPaths = []
        for p in range(Npaths):
            sortedListPaths.append(listPaths[indeces_paths[p]])

        self.list_paths = sortedListPaths

###########################################################################################
class CalculateEnergy:
    def __init__(self, FNs, beta = 0.40): # kJ^-1

        nodes_size = FNs.nodes_size
        
        tot_size   = np.sum(nodes_size)
        W          = nodes_size / tot_size
        energy     = - 1 / beta * np.log(W)
        energy     = energy - np.min(energy)

        self.energy = energy

class PlotGraph:
    def __init__(self, FNs, BAM):

        """
        .....
        """


def SqRA(FNs, CE, BAM, beta = 0.4, flux = 1):
    Af   = flux * BAM.A

    # Diagonalization
    PI   = np.exp( - beta * CE.energy )
    SQRA = np.sqrt(PI)
    Di   = scipy.sparse.spdiags(SQRA,   0, FNs.Nnodes, FNs.Nnodes) #.toarray()
    D1   = scipy.sparse.spdiags(1/SQRA, 0, FNs.Nnodes, FNs.Nnodes)       
    Q    = D1 * Af * Di

    Q    = Q + scipy.sparse.spdiags( - Q.sum(axis=1).T, 0, FNs.Nnodes, FNs.Nnodes )

    return Q

def effective_rate(pathA, Q):
    total_time = 0.0

    for i in range(len(pathA) - 1):
        from_state = pathA[i]
        to_state = pathA[i + 1]
        rate = Q[from_state, to_state]

        if rate <= 0:
            raise ValueError(f"No valid transition rate from state {from_state} to {to_state} (rate = {rate})")
        
        total_time += 1.0 / rate

    effective_rate = 1.0 / total_time
    return effective_rate
