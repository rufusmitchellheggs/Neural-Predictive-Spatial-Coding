from __init__ import *

def trace_binning(trace_array, bin_size=15):
    """Bin trace time series in to select bin size (takes mean of bin)
    Input:
    ------
    - trace_array = N x R array, where N = Dimensions/number of neurons and R = Continuous trace
    - bin_size = The number of frames per bin
    
    Output:
    ------
    - binned_trace
    """ 
    bin_size = bin_size
    binned_trace = []
    for neuron in trace_array:
        per_neuron=[]
        i=0
        while i <= len(neuron):
            per_neuron.append(np.mean(neuron[i:i+bin_size]))
            i+=bin_size
        binned_trace.append(per_neuron)
    return binned_trace

def behav_vector_binning(behav_vector, bin_size=15):
    """Bin event trace time series in to select bin size (takes mode of bin)
    Input:
    ------
    - behav_vector = 1D behavioural vector
    - bin_size = The number of frames per bin
    
    Output:
    ------
    - binned_behav_vector
    """ 
    bin_size = bin_size
    binned_behav_vector = []
    i=0
    while i <= len(behav_vector)-1:
        binned_behav_vector.append(stats.mode(behav_vector[i:i+bin_size])[0])
        i+=bin_size
    return binned_behav_vector

#Define all functions
def similarity_calc(calcium_trace_vectors, behav_vector):
    """ 
    Calculate the similarity score between calcium trace vectors and behvioral vector 
    
    INPUT:
    -------
    >> calcium_trace_vectors - all calcium traces as np.array
    >> behav_vector - behavioral vector
    
    OUTPUT:
    -------
    >> similarity - the similarity score
    """
    similarity = []
    for i in range(len(calcium_trace_vectors)):
        similarity.append(np.dot((2*behav_vector),calcium_trace_vectors[i])/((np.linalg.norm(behav_vector)**2)+(np.linalg.norm(calcium_trace_vectors[i])**2)))
    return similarity


def behav_vectors(behavioral_vector):
    """ 
    Create a new matrix that replaces sequences of ones 
    with the length of the sequence of ones
    
    INPUT:
    -------
    >> behavioral_vector - the original behavioral vector
    
    OUTPUT:
    -------
    >> behav_vectors - adpated behavioral vector (e.g. 001100111 --> 002003)
    """
    behav_vectors = []
    for i in range(len(behavioral_vector)):
        if i > 0:
            if behavioral_vector[i] > 0 and behavioral_vector[i-1] == 0:
                count = 0
                idx = 0
                while behavioral_vector[i+idx] != 0 in behavioral_vector[i:]:
                    idx +=1
                    count +=1
                behav_vectors.append(count)
            elif behavioral_vector[i] !=1 :
                behav_vectors.append(behavioral_vector[i])
        elif behavioral_vector[i] != 1:
            behav_vectors.append(behavioral_vector[i])
        
    return behav_vectors

def shuffled_vector_scores(behavioral_vector, calcium_traces, num_frames, shuffles = 5000):
    """ 
    Creates n shuffles of the behavioral vector and calculates the new similarity 
    score between behavioral vectors and calcium trace vectors
    INPUT:
    -------
    >> behavioral_vector -  use behav_vectors func
    >> calcium_traces - (raw calcium trace for each file)
    >> shuffles - the number of times to shuffle the data (default = 5000)
    
    OUTPUT:
    -------
    >> similarity_shuffled_all - Similarity score for calcium trace vectors and shuffled behavioral vector
    >> similarity_calcium_traces - Shuffled behavioral vectors (as a list)
    """
    
    similarity_shuffled_all = []
    new_seq_all = []
    # Loop over shuffles = x (default 5000)
    for i in range(shuffles):
        # Shuffle behavo
        np.random.shuffle(behavioral_vector)
        
        indexes = []
        num_ones = []
        behav_vector_all = np.zeros(num_frames)
        for vector in range(len(behavioral_vector)):
            if behavioral_vector[vector] > 0:
                indexes.append(np.arange(vector, vector+behavioral_vector[vector],1))
                num_ones.append(np.ones(int(behavioral_vector[vector])))           

        indexes = [int(i) for sublist in indexes for i in sublist]
        num_ones = [int(i) for sublist in num_ones for i in sublist]
        behav_vector_all[indexes] = num_ones
        new_seq_all.append(behav_vector_all)
        similarity_shuffled_all.append(similarity_calc(calcium_traces, behav_vector_all))  
    
    return similarity_shuffled_all, new_seq_all



def percentile(similarity_shuffled_all, similarity_calcium_traces):
    """ 
    Calculates the percentile of the calcium trace vector similarity score
    INPUT:
    -------
    >> similarity_shuffled_all -  use shuffled_vector_scores function
    >> similarity_calcium_traces - use similarity_calc
    
    OUTPUT:
    -------
    >> percentile - percentile out of shuffled dsimilarity score distribution
    >> similarity_distribution_all - similarity score distribution for all behav_vectors
    """
    similarity_distribution_all = []
    percentile = []
    for i in range(len(similarity_shuffled_all[0])):
        similarity_distribution = []
        for j in range(len(similarity_shuffled_all)):
            similarity_distribution.append(similarity_shuffled_all[j][i])
        percentile.append(stats.percentileofscore(similarity_distribution, similarity_calcium_traces[i]))
        similarity_distribution_all.append(similarity_distribution)
    return [percentile, similarity_distribution_all]


def self_cosine_vectorized(a):
    dots = a.dot(a.T)
    sqrt_sums = np.sqrt((a**2).sum(1))
    cosine_dists = 1 - (dots/sqrt_sums)/sqrt_sums[:,None]
    np.fill_diagonal(cosine_dists,0)
    return cosine_dists

def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
    """
    # Number of points                                                                        
    n = len(D)
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    pos_evals = [eval for eval in evals if eval > 0]
    evecs = evecs[:,idx]
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals>0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
    return Y, pos_evals

def plot_2D(X, Y, colors, b_vec_labels, target_names, x_comp = 0, y_comp=1):
    fig = plt.figure(num=None, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
    for color, i, target_name in zip(colors, b_vec_labels, target_names):
        plt.scatter(X[:,x_comp][Y == i], 
                    X[:,y_comp][Y == i],
                    color=color, label=target_name, s=2)  
#     plt.legend(target_names, markerscale=3)
    plt.xlabel('MDS%d' %int(x_comp+1))
    plt.ylabel('MDS%d' %int(y_comp+1))
    sns.despine()
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.grid(False)
    plt.show()
    return
    
def Plot_3D(X, Y, colors, b_vec_labels, target_names, x_comp=0, y_comp=1, z_comp=2):

    fig = plt.figure(num=None, figsize=(8, 4), dpi=200, facecolor='w', edgecolor='k')
    ax = Axes3D(fig)
#     sns.set_style('whitegrid')

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    
    ax.set_xlabel('MDS%d' %int(x_comp+1), fontsize=10)
    ax.set_ylabel('MDS%d' %int(y_comp+1), fontsize=10)
    ax.set_zlabel('MDS%d' %int(z_comp+1), fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)

    for color, i, target_name in zip(colors,b_vec_labels, target_names):
        ax.scatter(X[:,x_comp][Y == i],
                   X[:,y_comp][Y == i], 
                   X[:,z_comp][Y == i],
                   s=2, c=color)    
        
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.5, 0.5)
    return ax
