import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans

def svd(X):
    n = X.shape[1]  
    
    print(f"Calculating X^T X of shape {n}x{n}...")
    XTX = np.dot(X.T, X)
    
    print("Calculating eigenvalues and eigenvectors of X^T X...")
    eigenvalues, eigenvectors = np.linalg.eig(XTX)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]   
    sorted_eigenvalues  = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:,idx]
    
    # Singular values are the square root of eigenvalues 
    singular_values = np.sqrt(sorted_eigenvalues)
    
    # Create the diagonal matrix S
    S = np.diag(singular_values)
    
    # V is the matrix of sorted eigenvectors
    V = sorted_eigenvectors
    
    # V^T is the transpose of V
    VT = V.T
    
    # Construct the U matrix using u_i = (1 / sigma_i) * X * v_i
    # We can compute all columns of U at once with matrix multiplication
    # U = X @ V @ S^-1
    S_inv = np.diag(1.0 / (singular_values))
    
    print("Calculating U matrix...\n")
    U = np.dot(X, np.dot(V, S_inv))
    
    return U, S, VT

def plot_eigenfaces(VT, num_faces=36):
    d = 32 # Image dimension, 32x32
    n = int(np.ceil(np.sqrt(num_faces))) # Grid dimension, 6x6
    
    # Create a blank canvas
    canvas = np.zeros((n * d, n * d))
    
    for i in range(num_faces):
        # Calculate grid position
        r = i // n
        c = i % n
        
        # Reshape the eigenface vector to 32x32
        eface = VT[i, :].reshape(d, d, order='F')
        
        # Normalize the image to improve contrast
        eface = (eface - eface.min()) / (eface.max() - eface.min())
        
        # Place image to the canvas
        canvas[r*d:(r+1)*d, c*d:(c+1)*d] = eface
        
    plt.figure(figsize=(6, 6))
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(top=0.92) 
    plt.show()

def plot_dimensionality_reduction(X, VT):
    k = 2 
    
    print(f"Projecting {X.shape[1]} dimensions to {k} dimensions...")
    
    # Select top 2 eigenfaces
    V_k = VT[:k, :]
    
    # Project data onto the eigenfaces
    Z = np.dot(X, V_k.T)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(Z[:, 0], Z[:, 1], alpha=0.5, s=10, label='Face Samples', color='black')
    plt.title('Dimensionality Reduction, 1024 to 2 dimensions', fontsize=16)
    plt.xlabel('Principal Component 1 (Max Variance)', fontsize=12)
    plt.ylabel('Principal Component 2 (Second Max Variance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    print("Interpretation: Each dot represents a face. Faces that look similar will be close to each other in this 2D space.")

def plot_cumulative_variance(S, thresholds):
    # Extract singular values and calculate cumulative ratios
    singular_values = np.diag(S)
    cumulative_ratios = np.cumsum(singular_values) / np.sum(singular_values)
    
    ks = range(1, len(cumulative_ratios) + 1)

    plt.figure(figsize=(12, 7))
    plt.plot(ks, cumulative_ratios, linestyle='-', linewidth=2, color='black', label='Cumulative Variance')
    
    plt.title('Cumulative Variance Ratio vs. k', fontsize=16)
    plt.xlabel('Number of Eigenfaces, k', fontsize=12)
    plt.ylabel('Cumulative Variance Ratio', fontsize=12)
    
    colors = ['green', 'blue']
    labels = ["PC1 + PC2", "PC1 + PC2 + PC3"]
    optimal_ks = []

    for i, threshold in enumerate(thresholds):
        # Find optimal k for this specific threshold
        k_indices = np.where(cumulative_ratios >= threshold)[0]
        
        if len(k_indices) > 0:
            optimal_k = k_indices[0] + 1 
        else:
            optimal_k = len(singular_values)
        
        optimal_ks.append(optimal_k)
        
        c = colors[i] if i < len(colors) else 'green'
        l = labels[i] if i < len(labels) else f"Threshold {threshold*100:.1f}%"

        plt.axhline(y=threshold, color=c, linestyle='--', alpha=0.6)
        
        plt.axvline(x=optimal_k, color=c, linestyle=':', linewidth=2, 
                    label=f'{l}: k={optimal_k} ({threshold*100:.2f}%)')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
    
    return optimal_ks

def plot_original_vs_reconstructed(X, VT, k_values, num_to_plot=100):    
    d = 32
    rows = int(np.ceil(np.sqrt(num_to_plot)))
    cols = rows 

    canvas_orig = np.zeros((rows * d, cols * d))
    for i in range(num_to_plot):
        r = i // cols
        c = i % cols
        img = X[i, :].reshape(d, d, order='F')
        img = (img - img.min()) / (img.max() - img.min())
        canvas_orig[r*d:(r+1)*d, c*d:(c+1)*d] = img

    plt.figure(figsize=(10, 10))
    plt.imshow(canvas_orig, cmap='gray')
    plt.axis('off')
    plt.title(f'First {num_to_plot} Original Faces', fontsize=16)
    plt.show()
    
    for k in k_values:
        print(f"Reconstructing first {num_to_plot} faces using k={k}...")
        
        V_k = VT[:k, :]
        Z = np.dot(X, V_k.T)
        X_approx = np.dot(Z, V_k)
        
        canvas_recon = np.zeros((rows * d, cols * d))
        for i in range(num_to_plot):
            r = i // cols
            c = i % cols
            img = X_approx[i, :].reshape(d, d, order='F')
            img = (img - img.min()) / (img.max() - img.min())
            canvas_recon[r*d:(r+1)*d, c*d:(c+1)*d] = img

        plt.figure(figsize=(10, 10))
        plt.imshow(canvas_recon, cmap='gray')
        plt.axis('off')
        plt.title(f'Reconstructed Faces (k={k})', fontsize=16)
        plt.show()

def plot_reconstruction_error(X, VT, max_k=100):
    errors = []
    ks = range(1, max_k + 1)
    
    print(f"Calculating average reconstruction error for k=1 to {max_k}...")
    
    for k in ks:
        V_k = VT[:k, :]
        Z = np.dot(X, V_k.T)
        X_approx = np.dot(Z, V_k)
        mse = np.mean((X - X_approx) ** 2)
        errors.append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(ks, errors, linestyle='-', color='black')
    plt.title('Average Reconstruction Error vs k')
    plt.xlabel('Number of Eigenfaces, k')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

def analyze_features(X, VT, eigenface_index):
    # Plot the eigenface
    ef_vector = VT[eigenface_index, :]
    ef_image = ef_vector.reshape(32, 32, order='F')

    projections = np.dot(X, ef_vector.T)

    max_idx = np.argmax(projections)
    min_idx = np.argmin(projections)
    median_idx = np.argsort(projections)[len(projections) // 2]

    max_face = X[max_idx, :].reshape(32, 32, order='F')
    min_face = X[min_idx, :].reshape(32, 32, order='F')
    median_face = X[median_idx, :].reshape(32, 32, order='F')

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plt.subplots_adjust(bottom=0.2)

    # The Eigenface
    axes[0].imshow(ef_image, cmap='gray')
    axes[0].set_title(f'Eigenface {eigenface_index + 1}')
    axes[0].axis('off')

    # Max Projection : Strong presence of the eigenface
    axes[1].imshow(max_face, cmap='gray')
    axes[1].set_title(f'Max Projection\n(Image {max_idx})')
    axes[1].axis('off')

    # Median Projection : Neutral face at the midpoint of this variation
    axes[2].imshow(median_face, cmap='gray')
    axes[2].set_title(f'Median Projection\n(Image {median_idx})')
    axes[2].axis('off')

    # Min Projection : Absence or opposite of the eigenface
    axes[3].imshow(min_face, cmap='gray')
    axes[3].set_title(f'Min Projection\n(Image {min_idx})')
    axes[3].axis('off')

    plt.show()

def analyze_component_variance(S):
    singular_values = np.diag(S)
    variances = singular_values ** 2
    total_variance = np.sum(variances)
    
    # Calculate and plot the percentage of explained variances for each principal component
    explained_var_ratio = (variances / total_variance)
    
    num_components = len(explained_var_ratio)
    x = np.arange(1, num_components + 1)
    
    limit = 50 
    
    plt.figure(figsize=(12, 6))
    plt.bar(x[:limit], explained_var_ratio[:limit] * 100, color='dimgray', label='Individual Variance')
    plt.plot(x[:limit], explained_var_ratio[:limit] * 100, color='black', linestyle='-', linewidth=2, markersize=4, label='Trend Line')
    plt.title('Percentage of Explained Variance by Principal Component', fontsize=16)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Percentage of Explained Variance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.xticks(range(1, limit + 1))
    plt.legend()
    
    r1 = explained_var_ratio[0] * 100
    r2 = explained_var_ratio[1] * 100
    r3 = explained_var_ratio[2] * 100
    
    # Returns ratios for PC1+PC2 and PC1+PC2+PC3.
    ratio_two_pca   = explained_var_ratio[0] + explained_var_ratio[1]
    ratio_three_pca = explained_var_ratio[0] + explained_var_ratio[1] + explained_var_ratio[2]
    
    print(f"\nCapturing variance using principal components...")
    print(f"Variance captured by PC 1: {r1:.4f}%")
    print(f"Variance captured by PC 2: {r2:.4f}%")
    print(f"Variance captured by PC 3: {r3:.4f}%")
    print(f"Combined (PC 1 + PC 2):  {r1 + r2:.4f}%")
    print(f"Combined (PC 1 + PC 2 + PC 3):  {r1 + r2 + r3:.4f}%")
    
    plt.show()
    
    return ratio_two_pca, ratio_three_pca

def plot_pairwise_principal_components(X, VT, n_components=5, n_clusters=4):
    Z = np.dot(X, VT[:n_components, :].T)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Z)
    
    fig, axes = plt.subplots(n_components, n_components, 
                             figsize=(20, 20), 
                             sharex='col', 
                             sharey='row')
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.1, left=0.1, top=0.92)
    
    fig.suptitle(f'Pairwise PCA using Top {n_components} Components', fontsize=20)

    for i in range(n_components):
        for j in range(n_components):
            ax = axes[i, j]
            
            # Diagonal plots (e.g., PC1 vs PC1) will be straight lines
            scatter = ax.scatter(Z[:, j], Z[:, i], c=labels, cmap='bone', vmin=0, vmax=labels.max() + 2, alpha=0.5, s=10)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            if i == n_components - 1:
                ax.set_xlabel(f'PC {j+1}', fontsize=12)
            if j == 0:
                ax.set_ylabel(f'PC {i+1}', fontsize=12)

    plt.show()

if __name__ == "__main__":
    data = loadmat('faces.mat')
    X = data['X']
    print(f"Shape of original matrix X: {X.shape}\n")

    U, S, VT = svd(X)

    print(f"Shape of U: {U.shape}")
    print(f"Shape of S: {S.shape}")
    print(f"Shape of V^T: {VT.shape}\n")

    print("Plotting first 36 eigenfaces...")
    plot_eigenfaces(VT, num_faces=36)

    print("\nAnalyzing what eigenfaces represent...")
    
    for i in range(5):
        analyze_features(X, VT, i)

    print("\nIllustrating dimensionality reduction...")
    plot_dimensionality_reduction(X, VT)

    ratio_two_pca, ratio_three_pca = analyze_component_variance(S)

    print("\nFinding optimal k using explained variance...")
    
    optimal_ks = plot_cumulative_variance(S, thresholds=[ratio_two_pca, ratio_three_pca])
    
    k_two_pca   = optimal_ks[0]
    k_three_pca = optimal_ks[1]
    
    print(f"\nTo retain {ratio_two_pca*100:.2f}% (PC1+PC2), we need k = {k_two_pca}.")
    print(f"To retain {ratio_three_pca*100:.2f}% (PC1+PC2+PC3), we need k = {k_three_pca}.")
    
    plot_original_vs_reconstructed(X, VT, k_values=[k_two_pca, k_three_pca, 1024], num_to_plot=100)

    print("\nPlotting average reconstruction error...")
    plot_reconstruction_error(X, VT, max_k=100)

    print("\nPCA using top components...")
    

    plot_pairwise_principal_components(X, VT, n_components=5, n_clusters=4)
