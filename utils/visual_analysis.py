import os
import torch
from torchvision.utils import save_image

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def visual_analysis(model, test_loader, device, output_dir="reconstructed_images"):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            masked_images = images * (1.0 - masks)

            # Forward pass to reconstruct images
            reconstructed_images, _, _ = model(masked_images)

            # Save original and reconstructed images side by side for comparison
            for i in range(reconstructed_images.size(0)):
                save_image(reconstructed_images[i], os.path.join(output_dir, f"reconstructed_{batch_idx}_{i}.png"))
                save_image(images[i], os.path.join(output_dir, f"original_{batch_idx}_{i}.png"))

    print(f"Reconstructed images saved in {output_dir}")


def plot_pca(points, labels=None, save_path='pca_plot.png'):
    """
    Apply PCA on a given n x d tensor and reduce it to n x 2.
    Optionally save the results as a figure with different colors for each class based on labels.
    
    Args:
        points (torch.Tensor): Input tensor of shape (n, d).
        labels (torch.Tensor or np.ndarray, optional): Labels for each data point, used to color different classes.
        save_path (str): The file path where the plot will be saved.
    """

    # Convert points to numpy if it's a torch tensor
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    # Apply PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(points)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    if labels is not None:
        # Convert labels to numpy if it's a torch tensor
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Get unique labels for different classes
        unique_labels = np.unique(labels)
        
        # Plot each class with a different color
        for label in unique_labels:
            idx = labels == label
            plt.scatter(reduced_data[idx, 0], reduced_data[idx, 1], label=f'Class {label}')
        plt.legend()
    else:
        # Plot without labels (all points in one color)
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1])

    plt.title('PCA Results (n x 2)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    # Save the plot instead of displaying it
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the figure to avoid displaying it in notebooks