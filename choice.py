# # Importing Packages
import os
import copy
import config
import dota2api
import numpy as np
import pandas as pd
from sklearn import preprocessing

# # applying roulette wheel
def roulette_wheel(id_hero, all_win_data_only, heroes_dict):
    hero_id = id_hero
    grouped1 = all_win_data_only.groupby(all_win_data_only[hero_id]).size()
    
    data_bucket = []
    for i in range(len(grouped1)):
        count = grouped1.iloc[i]
        for j in range(count):
            data_bucket.append(count)
    
    # # Shuffling
    np.random.shuffle(data_bucket)
    
    # # Choosing 
    choice = np.random.choice(data_bucket)
    
    # # Hero ID
    hero_id = grouped1.loc[grouped1.values == choice].index[0]

    # # Hero Name
    hero_name = heroes_dict[hero_id]

    return hero_name
