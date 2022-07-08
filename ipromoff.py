from utils import utils
import sys
import os

if __name__ == '__main__':
    only_graph = True
    args = utils.argParser(sys.argv[:])
    config = utils.getExpConfig(args.config)
    if only_graph:
        utils.generateGraph(config)
        #utils.generateGraphSeeds(config)
    else:
        solver = utils.createSolver(args.solver)
        solver_ = solver(args.mode, args.load, os.getcwd(), config)
        solver_.solve_pulp()
        solver_.saveSolution()
        print('Times')
        print(solver_.getTimes())
        print('Saved')
        print(solver_.getSaved())
        #print(solver_.getSolution())
        print('Restrictions')
        print(solver_.getVariables_Restrictions())
        print('Degrees')
        print(solver_.getDegrees())

