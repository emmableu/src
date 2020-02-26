from simulate_game import *
from scipy import stats
total = 186
target_recall = 0.7
thres = 10
label_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse','moveanimate']
count_s_all = {}
def get_summary(label_name, total, thres, target_recall):
    all_simulation = load_obj('all_simulation_'+label_name,'/Users/wwang33/Documents/IJAIED20/src/workspace/data/game_labels_'+str(total), 'simulation_'+ str(thres) +"_"+ str(target_recall))
    count_s = []
    for simulation in all_simulation:
        count_s.append(simulation.count)
    median_index = np.argsort(count_s)[len(count_s) // 2]
    all_simulation[median_index].plot("/Users/wwang33/Documents/IJAIED20/src/workspace/data/game_labels_186/simulation_10_0.7/plots/", show = False)
    # plot_real(all_simulation[median_index],"/Users/wwang33/Documents/IJAIED20/src/workspace/data/game_labels_186/simulation_10_0.7/plots/")
    print(all_simulation[median_index].count)
    count_s_all[label_name] = (count_s)

for label_name in label_name_s:
    get_summary(label_name, total, thres, target_recall)
    print(count_s_all)
