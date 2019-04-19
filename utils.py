'''
 Version: 0419
 Various Util Functions
'''
import networkx as nx
import numpy as np
import random,copy, time, json, os, argparse
from gurobipy import *
from UMLP_solver import *


# INPUT description:
# G <- a DAG object representing n knowledge points' dependencies
# B <- a number describing total budget
# C <- a row vector of length n describing cost of learning Ki
# U <- a row vector of length n describing the value of learning Ki
# type <- type of cost function
small_n = 10e-5

parser = argparse.ArgumentParser(description = 'UMLP simulator')
parser.add_argument('--n', default = '[30,30,10]',type=str, help='Specify the range of nodes in [start,end,step] format.')
parser.add_argument('--density', default = '[0.2,0.2,0.1]', type=str, 
    help='Specify the range of edge density in [start,end,step] format.')
parser.add_argument('--nsim', default = 30, type = int, help = 'Specify number of simulations.')
parser.add_argument('--verbose',default = False, type=bool, help="Print progress?")
parser.add_argument('--solver',default = '[bf]', type=str, 
    help="Specify solver types in [x,y,...] form (bf: Brute Force; gd: Greedy; ilp: Integer Linear Program)")
parser.add_argument('--maxlearnP',default = '[0.166,0.166,0.1]', type=str, 
    help="Specify the range of maximum fraction of knowledge points that user can learn in [1/start,1/end,step] format.")
parser.add_argument('--costType',default = '[add]', type=str, 
    help="Specify cost type in [x,y,...] form (add: additive; mono: monotone; sub: submodular)")

def process_args(p):
	def splitAndStrip(s):
		l = s.split(",")
		l = list(map(lambda x: x.replace("[","").replace("]",""),l))
		return l

	nsim = p['nsim']
	verbose = p['verbose']
	costType = p['costType']
	p.pop('nsim')
	p.pop('verbose')

	
	arg_vals = list(map(splitAndStrip, list(p.values())))
	Ns, densities, solvers, budgets, costType = arg_vals
	try:
		assert(len(Ns) == 3 and len(densities) == 3 and len(budgets)==3)
	except:
		raise AssertionError('Input form of Ns, densities, solvers, or budgets invalid. Please check you arguments (Hint: put them in list form).')

	try:
	    Ns = np.arange(int(Ns[0]),int(Ns[1])+int(Ns[2]),step=int(Ns[2]))
	    densities = np.arange(float(densities[0]),float(densities[1])+small_n,
	                          step=float(densities[2]))
	    budgets = np.arange(float(budgets[0]),float(budgets[1])+small_n,
	                        step=float(budgets[2]))
	except:
		raise Exception("Number conversion error! Please check your arguments.")

	for s in solvers:
		if s not in ['bf','gd','ilp']:
			raise AssertionError('Unrecognized solver type!')

	for t in costType:
		if t not in ['add','mono','sub']:
			raise AssertionError('Unrecognized cost function type!')


	return [Ns, densities, solvers, budgets, nsim, costType, verbose]

def generate_result_dict(N, density, budget, cost, solvers, sols, times):
	sols_means = sols.mean(axis=0)
	sols_sds = sols.std(axis=0)
	times_means = times.mean(axis=0)
	times_sds = times.std(axis=0)
	result = []
	for solver_idx in range(len(solvers)):
		d = {"N":N, "Density":density, 'Solver':solvers[solver_idx],"Budget":budget, "Cost":cost,
			'Time_avg':times_means[solver_idx],'Time_sd':times_sds[solver_idx],'Sol_avg':sols_means[solver_idx],
			'Sol_sd':sols_sds[solver_idx]}
		result.append(d)
	return result


