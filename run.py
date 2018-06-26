from predictor3 import NeighborInitialization
import cProfile
import re

def bar():
    ni = NeighborInitialization("data_train.csv", 0.0)
    ni.do_work()
bar()
#cProfile.run('bar()')

