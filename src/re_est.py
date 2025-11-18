import numpy as np
import pandas as pd

def fill_re_average_neighbors(data, metric='manhattan', n_closest=4):
    """
    Fill missing Re (Oil) by averaging the radii of closest neighbors
    """
    data_filled = data.copy()
    missing_re = data_filled['Rad (m)'].isna()
    
    if not missing_re.any():
        return data_filled
    
    print(f"Found {missing_re.sum()} wells with missing Re (Oil)\n")
    
    coords = data_filled[['X', 'Y']].values
    well_names = data_filled['Well name'].values
    re_values = data_filled['Rad (m)'].values
    
    missing_indices = data_filled[missing_re].index.tolist()
    
    for idx in missing_indices:
        well_name = data_filled.loc[idx, 'Well name']
        i = list(data_filled.index).index(idx)
        
        # Find neighbors with valid Re values
        neighbor_re = []
        
        for j in range(len(coords)):
            if i != j and not np.isnan(re_values[j]):
                if metric == 'manhattan':
                    dist = np.abs(coords[i][0] - coords[j][0]) + np.abs(coords[i][1] - coords[j][1])
                else:
                    dist = np.linalg.norm(coords[i] - coords[j])
                
                neighbor_re.append((dist, re_values[j], well_names[j]))
        
        # Sort by distance and take n_closest
        neighbor_re.sort(key=lambda x: x[0])
        closest = neighbor_re[:n_closest]
        
        # Calculate average
        if closest:
            avg_radius = np.mean([r for _, r, _ in closest])
            print(f"{well_name}: Average of {len(closest)} neighbors = {avg_radius:.1f}m")
            print(f"  Neighbors: {', '.join([f'{name}({r:.0f}m)' for _, r, name in closest])}")
        else:
            avg_radius = 200
            print(f"{well_name}: No neighbors, using default {avg_radius:.1f}m")
        
        data_filled.loc[idx, 'Rad (m)'] = avg_radius
    
    return data_filled


def distance_to_outer_circle(data, metric='manhattan', n_closest=4):
    """
    Return DataFrame with distance to outer edge of neighboring circles
    """
    coords = data[['X', 'Y']].values
    well_names = data['Well name'].values
    re_values = data['Rad (m)'].values
    
    # Calculate all distances first
    all_distances = {}
    
    for i in range(len(coords)):
        distances_list = []
        
        for j in range(len(coords)):
            if i != j:
                if metric == 'manhattan':
                    center_dist = np.abs(coords[i][0] - coords[j][0]) + np.abs(coords[i][1] - coords[j][1])
                else:
                    center_dist = np.linalg.norm(coords[i] - coords[j])
                
                # Distance to outer edge = center distance - neighbor's radius
                if not np.isnan(re_values[j]):
                    edge_dist = center_dist - re_values[j]
                else:
                    edge_dist = center_dist  # If no radius, use center distance
                
                distances_list.append({
                    'neighbor': well_names[j],
                    'center_distance': center_dist,
                    'neighbor_radius': re_values[j],
                    'edge_distance': edge_dist
                })
        
        # Sort by edge distance and keep top n
        distances_list.sort(key=lambda x: x['edge_distance'])
        all_distances[well_names[i]] = distances_list[:n_closest]
    
    # Create a formatted DataFrame
    rows = []
    for well, neighbors in all_distances.items():
        row = {'Well': well}
        for idx, neighbor_info in enumerate(neighbors, 1):
            row[f'Neighbor_{idx}'] = neighbor_info['neighbor']
            row[f'Center_Dist_{idx}'] = neighbor_info['center_distance']
            row[f'Neighbor_Re_{idx}'] = neighbor_info['neighbor_radius']
            row[f'Edge_Dist_{idx}'] = neighbor_info['edge_distance']
        rows.append(row)
    
    return pd.DataFrame(rows)


def fill_re_from_distances(data, metric='manhattan', safety_buffer=0, default_radius=200, min_radius=50):
    """
    Fill missing Re (Oil) values based on distances to neighboring wells.
    Calculate maximum possible radius without overlapping neighbors.
    
    Parameters:
    - safety_buffer: Gap between circles (0 = just touch)
    - default_radius: Used when no neighbors exist
    - min_radius: Minimum allowable radius
    """
    data_filled = data.copy()
    missing_re = data_filled['Rad (m)'].isna()
    
    if not missing_re.any():
        print("No missing Rad (m) values found.")
        return data_filled
    
    print(f"Found {missing_re.sum()} wells with missing Rad (m) values\n")
    
    # Get distance information for all wells
    well_distances = distance_to_outer_circle(data_filled, metric=metric, n_closest=4)
    
    # Process each well with missing Re
    missing_indices = data_filled[missing_re].index.tolist()
    
    for idx in missing_indices:
        well_name = data_filled.loc[idx, 'Well name']
        
        # Get this well's distance info from the DataFrame
        well_row = well_distances[well_distances['Well'] == well_name]
        
        if well_row.empty:
            assigned_radius = default_radius
            print(f"{well_name}: No neighbors found, assigned default {assigned_radius:.1f}m")
        else:
            # Find minimum edge distance (closest neighbor's outer circle)
            min_edge_dist = float('inf')
            limiting_neighbor = None
            
            for i in range(1, 5):  # Check up to 4 neighbors
                edge_col = f'Edge_Dist_{i}'
                neighbor_col = f'Neighbor_{i}'
                
                if edge_col in well_row.columns and not pd.isna(well_row[edge_col].values[0]):
                    edge_dist = well_row[edge_col].values[0]
                    if edge_dist < min_edge_dist:
                        min_edge_dist = edge_dist
                        limiting_neighbor = well_row[neighbor_col].values[0]
            
            # Calculate radius
            if min_edge_dist == float('inf'):
                assigned_radius = default_radius
                print(f"{well_name}: No valid neighbors, assigned default {assigned_radius:.1f}m")
            else:
                # Maximum radius = distance to closest edge - safety buffer
                max_radius = min_edge_dist - safety_buffer
                assigned_radius = max(max_radius, min_radius)
                
                if max_radius < min_radius:
                    print(f"{well_name}: Limited space ({max_radius:.1f}m), using min {min_radius:.1f}m (may overlap)")
                else:
                    print(f"{well_name}: Limited by {limiting_neighbor}, edge_dist={min_edge_dist:.1f}m, assigned {assigned_radius:.1f}m")
        
        # Assign the radius
        data_filled.loc[idx, 'Rad (m)'] = assigned_radius
    
    return data_filled


#Approach 3: Inverse Distance Weighted Average of Nearest Neighbors
def fill_re_weighted_average(data, metric='manhattan', n_closest=4):
    """
    Fill missing Re (Oil) using inverse distance weighted average
    Closer neighbors have more influence on the predicted radius
    """
    data_filled = data.copy()
    missing_re = data_filled['Rad (m)'].isna()
    if not missing_re.any():
        return data_filled
    print(f"Found {missing_re.sum()} wells with missing Rad (m\n")
    coords = data_filled[['X', 'Y']].values
    well_names = data_filled['Well name'].values
    re_values = data_filled['Rad (m)'].values
    missing_indices = data_filled[missing_re].index.tolist()
    for idx in missing_indices:
        well_name = data_filled.loc[idx, 'Well name']
        i = list(data_filled.index).index(idx)
        # Find neighbors with valid Re values
        neighbor_data = []
        for j in range(len(coords)):
            if i != j and not np.isnan(re_values[j]):
                if metric == 'manhattan':
                    dist = np.abs(coords[i][0] - coords[j][0]) + np.abs(coords[i][1] - coords[j][1])
                else:
                    dist = np.linalg.norm(coords[i] - coords[j])
                neighbor_data.append((dist, re_values[j], well_names[j]))
        # Sort by distance and take n_closest
        neighbor_data.sort(key=lambda x: x[0])
        closest = neighbor_data[:n_closest]
        if closest:
            # Inverse distance weighting: weight = 1/distance
            weights = []
            radii = []
            for dist, radius, name in closest:
                weight = 1.0 / (dist + 1)  # +1 to avoid division by zero
                weights.append(weight)
                radii.append(radius)
            # Weighted average
            weighted_radius = np.average(radii, weights=weights)
            print(f"{well_name}: Weighted average = {weighted_radius:.1f}m")
            for (dist, radius, name), weight in zip(closest, weights):
                print(f"  {name}: Re={radius:.0f}m, dist={dist:.1f}m, weight={weight:.3f}")
        else:
            weighted_radius = 200
            print(f"{well_name}: No neighbors, using default {weighted_radius:.1f}m")
        data_filled.loc[idx, 'Rad (m)'] = weighted_radius
    return data_filled