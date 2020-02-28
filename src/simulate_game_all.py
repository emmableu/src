from __future__ import print_function, division
import sys, os
root = os.getcwd().split("src")[0] + "src/src/util"
sys.path.append(root)
from game import Game
import pickle
import os


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

def save_pickle(obj, name, dir, sub_dir):
    if sub_dir:
        csv_dir = dir +"/"+ sub_dir
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        csv_dir = dir
        pickle_dir = dir + '/pickle_files'
    atom_mkdir(csv_dir)
    atom_mkdir(pickle_dir)
    with open(pickle_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def simulate_30_times_to_get_all(label_name,total, thres = 0):
    data_path='game_labels_'+ str(total) + '/' + label_name + '.csv'
    # target_recall = 0.7
    # thres = 10
    all_simulation = []
    for i in range(30):
        read = Game()
        read = read.create(data_path)
        read.enable_est = False
        all_simulation.append(read)
        count = 0
        for j in range(total):
            pos, neg, total_real = read.get_numbers()
            if total_real != total:
                print("wrong! total_real != total")
                break
            if pos + neg < total:
                count += 1
            if pos < 1:
                for id in read.start_as_1_pos():
                    read.code(id, read.body["label"][id])
            else:
                uncertain, uncertain_proba, certain, certain_proba, _ = read.train(weighting=True, pne=True)
                if pos <= thres:
                    for id in uncertain:
                        read.code(id, read.body["label"][id])
                else:
                    for id in certain:
                        read.code(id, read.body["label"][id])
        read.count = count
        save_pickle(all_simulation, 'all_simulation_' + label_name,
                    '/Users/wwang33/Documents/IJAIED20/src/workspace/data/game_labels_'+str(total),
                    'simulation_'+ str(thres) +"_"+ "all")


if __name__ == "__main__":
    total = 186
    thres = 0
    label_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse','moveanimate']
    for label_name in label_name_s:
        simulate_30_times_to_get_all(label_name,total, thres)



    # all_simulation_wrap = load_obj('all_simulation_'+behavior,'/Users/wwang33/Documents/IJAIED20/src/workspace/data/game_labels_'+str(total), 'simulation_'+ str(thres) +"_"+ str(target_recall))
    # count_s = []
    # for simulation in all_simulation_wrap:
    #     count_s.append(simulation.count)
    # median_index = np.argsort(count_s)[len(count_s) // 2]
    # all_simulation_wrap[median_index].plot()
    # print(all_simulation_wrap[median_index].count)




