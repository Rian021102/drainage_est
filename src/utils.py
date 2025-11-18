import pandas as pd
import matplotlib.pyplot as plt


def load_data(path):
    '''
    Load data from excel file
    '''
    data=pd.read_excel(path)
    return data


def draw_drainage(data):
    '''
    Draw drainage plot with actual circles based on Re (Oil) radius
    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw circles with Re (Oil) as actual radius
    for i, row in data.iterrows():
        circle = plt.Circle((row['X'], row['Y']), 
                           radius=row['Rad (m)'], 
                           alpha=0.3, 
                           fill=True,
                           edgecolor='blue',
                           facecolor='lightblue',
                           linewidth=2)
        ax.add_patch(circle)
        
        # Add well point at center
        ax.scatter(row['X'], row['Y'], 
                  c='red', s=100, marker='o', zorder=5)
        
        # Add well name label
        ax.text(row['X'], row['Y'], 
               row['Well name'], fontsize=10, 
               ha='center', va='bottom')
    
    # Set equal aspect ratio so circles appear circular
    ax.set_aspect('equal', adjustable='datalim')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Drainage Plot for {data["Surface"].values[0]}')
    
    # Auto-adjust limits to show all circles
    ax.autoscale()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.show()