import os
import numpy as np
import matplotlib.pyplot as plt
from tda import VietorisRips, PersistentHomology, plot_barcode, plot_persistence_diagram

def main():
    print("=== Topological Data Analysis (TDA) Demonstration ===")
    
    # 1. Generate a noisy circle dataset
    print("\nGenerating noisy circle point cloud...")
    np.random.seed(42)
    n_samples = 30
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    radius = 1.0
    x = radius * np.cos(theta) + np.random.normal(0, 0.08, n_samples)
    y = radius * np.sin(theta) + np.random.normal(0, 0.08, n_samples)
    X = np.column_stack([x, y])

    # Plot point cloud
    os.makedirs("examples", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.scatter(X[:, 0], X[:, 1], color='#4D96FF', edgecolors='k', s=50, alpha=0.8)
    ax.set_facecolor('#F8F9FA')
    ax.grid(True, linestyle='--', color='#E2E8F0')
    ax.set_title("Noisy Circle Point Cloud", fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    # hide spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    
    pointcloud_path = "examples/noisy_circle.png"
    plt.savefig(pointcloud_path, bbox_inches='tight')
    plt.close()
    print(f"Point cloud plot saved to: {pointcloud_path}")

    # 2. Build Vietoris-Rips complex filtration
    print("\nBuilding Vietoris-Rips filtration...")
    max_epsilon = 1.5
    vr = VietorisRips(max_dim=2, max_epsilon=max_epsilon)
    filtration = vr.fit_transform(X)
    print(f"Total simplices constructed: {len(filtration)}")

    # 3. Compute Persistent Homology
    print("\nComputing persistent homology...")
    ph = PersistentHomology()
    intervals = ph.fit_transform(filtration)
    
    # Display summary of persistence intervals
    for dim in sorted(intervals.keys()):
        print(f"\nHomology Dimension H_{dim}:")
        print(f"  Total features: {len(intervals[dim])}")
        # Print top 5 longest-living features
        sorted_feats = sorted(intervals[dim], key=lambda x: (x[1] - x[0] if x[1] != float('inf') else float('inf')), reverse=True)
        for i, (birth, death) in enumerate(sorted_feats[:5]):
            status = "infinite" if death == float('inf') else f"dies at {death:.4f}"
            persistence = "inf" if death == float('inf') else f"{death - birth:.4f}"
            print(f"    Feature {i+1}: born at {birth:.4f}, {status} (persistence: {persistence})")

    # 4. Visualization and Plotting
    print("\nGenerating persistence barcode and diagram...")
    
    # Barcode
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    plot_barcode(intervals, max_limit=max_epsilon, ax=ax, show=False)
    barcode_path = "examples/barcode.png"
    plt.savefig(barcode_path, bbox_inches='tight')
    plt.close()
    print(f"Barcode plot saved to: {barcode_path}")

    # Persistence Diagram
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    plot_persistence_diagram(intervals, max_limit=max_epsilon, ax=ax, show=False)
    diagram_path = "examples/persistence_diagram.png"
    plt.savefig(diagram_path, bbox_inches='tight')
    plt.close()
    print(f"Persistence diagram plot saved to: {diagram_path}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
