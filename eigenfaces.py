import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def svd(X):
    """
    U  : The left singular vectors, shape (m, n).
    E  : The diagonal matrix of singular values, shape (n, n) in descending order.
    VT : The right singular vectors transposed, shape (n, n).
    """
    n = X.shape[1]  
    
    # Calculate the covariance matrix X^T X
    print(f"Calculating the covariance matrix X^T X of shape {n}x{n}...")
    XTX = np.dot(X.T, X)
    
    # Find the eigenvalues and eigenvectors of X^T X
    # These eigenvectors form the columns of V.
    print("Calculating eigenvalues and eigenvectors of X^T X...")
    eigenvalues, eigenvectors = np.linalg.eig(XTX)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]   
    sorted_eigenvalues  = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:,idx]
    
    # Construct the E and V^T matrices
    # For E, singular values are the square root of eigenvalues 
    singular_values = np.sqrt(sorted_eigenvalues)
    
    # Create the diagonal matrix E, also known as Sigma
    E = np.diag(singular_values)
    
    # V is the matrix of sorted eigenvectors
    V = sorted_eigenvectors
    
    # V^T is the transpose of V
    VT = V.T
    
    # Construct the U matrix
    # Using the formula, u_i = (1 / \sigma_i) * X * v_i
    # We can compute all columns of U at once with matrix multiplication.
    # U = X @ V @ E^-1
    E_inv = np.diag(1.0 / (singular_values))
    
    print("Calculating U matrix...\n")
    U = np.dot(X, np.dot(V, E_inv))
    
    return U, E, VT

def plot_eigenfaces(VT, num_faces):
    # Dimension of a single face image (1024 -> 32)
    d = int(np.sqrt(VT.shape[1]))

    # Square grid (36 -> 6x6, 100 -> 10x10)
    rows = int(np.sqrt(num_faces))
    cols = rows

    # Create a blank canvas
    canvas = np.zeros((rows * d, cols * d))

    # Place each eigenface to the canvas
    for i in range(num_faces):
        r = i // cols
        c = i % cols
        eface = VT[i, :].reshape(d, d, order='F')
        canvas[r*d:(r+1)*d, c*d:(c+1)*d] = eface

    plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    data = loadmat('faces.mat')
    X = data['X']
    
    print(f"Shape of original matrix X: {X.shape}\n")
    
    U, E, VT = svd(X)

    print(f"Shape of U: {U.shape}")
    print(f"Shape of E: {E.shape}")
    print(f"Shape of V^T: {VT.shape}\n")

    plot_eigenfaces(VT, num_faces=36)