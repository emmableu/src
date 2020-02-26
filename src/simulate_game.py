from __future__ import print_function, division
import sys, os
root = os.getcwd().split("src")[0] + "src/src/util"
sys.path.append(root)
from game import Game
from pdb import set_trace

if __name__ == "__main__":
    data_path='game_labels_95/jump.csv'
    target_recall = 0.95
    thres = 10
    read = Game()
    read = read.create(data_path)
    read.enable_est = False
    count = 0
    while True:
        count += 1
        print(count)

        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" % (pos, pos + neg, read.est_num))
        except:
            print("%d, %d" % (pos, pos + neg))

        if pos + neg >= total:
            print("pos + neg >= total:")
            break

        if pos < 1:
                for id in read.start_as_1_pos():
                    read.code(id, read.body["label"][id])
        else:
            uncertain, uncertain_proba, certain, certain_proba, _ = read.train(weighting=True, pne=True)
            if target_recall * read.est_num <= pos:
                print("target_recall * read.est_num <= pos:")
                break
            if pos <= thres:
                for id in uncertain:
                    read.code(id, read.body["label"][id])
            else:
                for id in certain:
                    read.code(id, read.body["label"][id])

    read.plot()
