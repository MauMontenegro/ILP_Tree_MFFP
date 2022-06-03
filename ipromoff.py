from utils import utils
import sys
import os

if __name__ == '__main__':
    only_graph= False
    args = utils.argParser(sys.argv[:])
    config = utils.getExpConfig(args.config)
    if only_graph:
        utils.generateGraph(config)
    else:
        solver = utils.createSolver(args.solver)
        solver_ = solver(args.mode, args.load, os.getcwd(),config)
        solver_.solve()
        solver_.saveSolution()
        print(solver_.getTimes())
        print(solver_.getSaved())

