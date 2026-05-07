"""Nested spaces visualization example.

Creates 4 nested spaces with the same TRS transformation applied at each level,
showing how transformations accumulate through the hierarchy.
"""

from coordinatus import Space2D, Point, create_space
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("=== Nested Spaces Visualization ===\n")
    
    # Create root space
    root = Space2D()
    
    # Create 4 nested spaces, each with the same TRS transform
    # trs2D(tx=1, ty=0, angle_rad=pi/4, sx=0.75, sy=0.75)
    spaces = [root]
    
    print("Creating nested spaces with transform: trs2D(1, 0, π/4, 0.75, 0.75)")
    for i in range(10):
        space = create_space(
            parent=spaces[-1],
            tx=1,
            ty=0,
            angle_rad=np.pi/4,
            sx=1/2**0.5,
            sy=1/2**0.5
        )
        spaces.append(space)
        print(f"  Space {i+1} created")
    
    # Define 4 local points for each space (unit triangle corners)
    local_points = [
        (0, 0),  # Origin
        (1, 0),  # Right
        (0, 1),  # Top
    ]
    
    print(f"\nLocal points in each space: {local_points}")
    print("\nConverting all points to absolute coordinates...\n")
    
    # Collect all absolute points, organized by space
    all_absolute_points = []
    space_labels = []
    
    for space_idx, space in enumerate(spaces):
        space_points = []
        for x, y in local_points:
            point = Point(np.array([x, y]), space=space)
            absolute = point.to_absolute()
            space_points.append((absolute.coords[0], absolute.coords[1]))
            all_absolute_points.append((absolute.coords[0], absolute.coords[1]))
            space_labels.append(space_idx)
        
        print(f"Space {space_idx} absolute points:")
        for i, (ax, ay) in enumerate(space_points):
            print(f"  Local {local_points[i]} → Absolute ({ax:.3f}, {ay:.3f})")
    
    # Prepare data for plotting
    x_coords = [p[0] for p in all_absolute_points]
    y_coords = [p[1] for p in all_absolute_points]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
   
    # Plot each space's points with a different color
    for space_idx in range(len(spaces)):
        # Get points for this space
        space_mask = [i for i, label in enumerate(space_labels) if label == space_idx]
        space_x = [x_coords[i] for i in space_mask]
        space_y = [y_coords[i] for i in space_mask]
        
        # Plot the triangle for this space
        # Close the triangle by adding the first point at the end
        triangle_x = space_x + [space_x[0]]
        triangle_y = space_y + [space_y[0]]
        
        ax.plot(triangle_x, triangle_y, '.-', 
                linewidth=2, 
                markersize=8,
                label=f'Space {space_idx}',
                alpha=0.7)
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X (absolute)', fontsize=12)
    ax.set_ylabel('Y (absolute)', fontsize=12)
    ax.set_title('Nested Spaces with Accumulated TRS Transformations\n' +
                 'Each space: translate(1,0), rotate(π/4), scale(√2/2)',
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add origin marker
    ax.plot(0, 0, 'k*', markersize=15, label='World Origin', zorder=10)
    
    plt.tight_layout()
    plt.savefig('examples/nested_spaces.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: examples/nested_spaces.png")
    plt.show()


if __name__ == "__main__":
    main()
