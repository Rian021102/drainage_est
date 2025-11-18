from utils import load_data, draw_drainage
from re_est import fill_re_average_neighbors




def main():
    # Load data
    path = 'P:\project\pythonpro\myvenv\Drainage_Radius\data\input_wells.xlsx'
    data=load_data(path)
    # draw drainage before filling missing Re
    for reservoir in data['Res_Name'].unique():
     for tank in data['Res_Number/Tank'].unique():
         draw_drainage(data[(data['Res_Name'] == reservoir) & (data['Res_Number/Tank'] == tank)])
    
    # Fill missing Re (Oil) values
    if data['Re (Oil)'].isna().any():
        for reservoir in data['Res_Name'].unique():
            for tank in data['Res_Number/Tank'].unique():
                well_data = data[(data['Res_Name'] == reservoir) & (data['Res_Number/Tank'] == tank)]
                
                if well_data.empty:
                    continue
                
                print("\n" + "="*60)
                print(f"Processing: {reservoir} - Tank {tank}")
                print("="*60)
                well_data_filled = fill_re_average_neighbors(well_data, metric='manhattan', n_closest=4)
                # Draw
                draw_drainage(well_data_filled)
                
                # Update main dataframe
                for idx in well_data_filled.index:
                    data.loc[idx, 'Re (Oil)'] = well_data_filled.loc[idx, 'Re (Oil)']

if __name__ == "__main__":
    main()
