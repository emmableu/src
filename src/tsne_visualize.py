#!/usr/local/bin/python3
from ast import literal_eval

import pandas as pd
import sys
import os
import pickle
from time import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold
from sklearn.utils import check_random_state


# for filename in os.listdir(directory):
def save_obj(obj, name, dir, sub_dir):
    if sub_dir:
        csv_dir = dir +"/"+ sub_dir
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        csv_dir = dir
        pickle_dir = dir + '/pickle_files'
    atom_mkdir(csv_dir)
    save_csv_or_txt(obj, csv_dir + '/' + name)


    atom_mkdir(pickle_dir)
    with open(pickle_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, dir, sub_dir):
    if sub_dir:
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        pickle_dir = dir + '/pickle_files'

    with open(pickle_dir+ "/"+ name + '.pkl', 'rb') as f:
        pickle_load = pickle.load(f)
        return pickle_load


def save_csv_or_txt(obj, dir_plus_name):
    try:
        obj.to_csv(dir_plus_name + '.csv')
    except:
        with open(dir_plus_name + '.txt', 'w') as f:
            for item in obj:
                f.write("%s\n" % item)

def atom_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

pd.set_option('display.max_columns', None)
cwd = os.getcwd()

def get_current_file_path(file_name):
    return ("/Users/wwang33/Documents/IJAIED20/src/workspace/data/game_labels_415/" + file_name)

def get_head(file_name):
    file_path = get_current_file_path(file_name)
    print(file_path)
    data = pd.read_csv(file_path)
    # data = pd.read_csv(file_path,header=None,names=list(range(10)))
    print(data.head())
    # print(type(data['good'].unique()[1]))
    print("columns: \n", data.columns)
    print("length: \n", len(data.index))
    return data



if __name__ == "__main__":
    file_name = str(sys.argv[1])
    data = get_head(file_name)
    content = [literal_eval(data["Abstract"][index]) for index in range(len(data))]
    input_data = np.array(content)
    random_state = check_random_state(0)
    n_neighbors = 30

    # p = random_state.rand(186) * (2 * np.pi - 0.55)
    # t = random_state.rand(186) * np.pi

    colors = [int(data["label"][index] == 'yes') for index in range(len(data))]
    fig = plt.figure(figsize=(15, 8))
    clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                                          method='modified')
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    trans_data = clf.fit_transform(input_data).T
    # trans_data = tsne.fit_transform(input_data).T
    ax = fig.add_subplot(1, 2, 1)
    plt.scatter(trans_data[0], trans_data[1], c = [color*100/255.0 for color in colors], cmap=plt.cm.rainbow)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    plt.show()
