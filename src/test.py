import pandas as pd
from utils import draw_drainage
from re_est import fill_re_average_neighbors,fill_re_from_distances,fill_re_weighted_average
def main():
    # Load data
    path = '/Users/rianrachmanto/pypro/project/drainage/data/Drainage Radius SJD_02.xlsx'
    data=pd.read_excel(path,skiprows=1)
    # for reservoir in data['Surface'].unique():
    #     draw_drainage(data[data['Surface'] == reservoir])

    #fill for each Surface that has Rad (m) missing values:
    for reservoir in data['Surface'].unique():
        well_data=data[(data['Surface'] == reservoir)]
        if well_data['Rad (m)'].isna().any():
            filled_data = fill_re_weighted_average(well_data,metric='euclidian', n_closest=4)
            draw_drainage(filled_data)



if __name__ == "__main__":
    main()