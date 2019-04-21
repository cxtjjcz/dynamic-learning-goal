'''
 Version: 0419
 Utility Maximizing Learning Plan Simulations
    # Brute force, Greedy, ILP Solver
    # Support Additive cost function
'''
import networkx as nx
import numpy as np
import random,copy, time, json, os, argparse, csv, datetime
from gurobipy import *
from UMLP_solver import *
import utils

# INPUT description:
# G <- a DAG object representing n knowledge points' dependencies
# B <- a number describing total budget
# C <- a row vector of length n describing cost of learning Ki
# U <- a row vector of length n describing the value of learning Ki
# type <- type of cost function

def generate_random_dag(nodes, density):
    #Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges.
    G = nx.DiGraph()
    edges = density * nodes * (nodes - 1)
    for i in range(nodes):
        G.add_node(i)
    for i in range(nodes**2):
        a = random.randint(0,nodes-1)
        b = a
        while b==a:
            b = random.randint(0,nodes-1)
        
        if G.has_edge(a,b): 
            G.remove_edge(a,b)
        else:
            G.add_edge(a,b)
            current_degree = sum(dict(G.degree()).values())
            if not (nx.is_directed_acyclic_graph(G) and current_degree <= edges):
                G.remove_edge(a,b)
    return G


def generate_cost(G):
    N = G.order()
    return np.random.uniform(1,10, N)


def generate_utility(G):
    N = G.order()
    return np.random.uniform(1,10, N)


def simulate():
    args = utils.process_args(vars(utils.parser.parse_args()))
    print(args)
    Ns, densities, solvers, budgets, nsim, costType, verbose, loadPrev = args
    result_dict = []
    result_colnums_names = ['N','Density','Solver','Budget','Cost',
                            'Time_avg','Time_sd','Sol_avg','Sol_sd']
    total_simulations = utils.getTotalSimulation([Ns, densities, budgets, costType])
    total_simulations *= nsim
    progress = 0

    for N in Ns:
        for density in densities:
            for budget in budgets:
                for cost in costType:
                    sols = np.zeros((nsim,len(solvers)))
                    times = np.zeros((nsim,len(solvers)))
                    if loadPrev:
                        try:
                            print ("Loading previously saved test instances...")
                            sims = utils.load_saved_instance(N,density,budget,cost)
                        except:
                            print ("Failed to load... Creating new instances...")
                            sims = []
                            loadPrev = False
                    else:
                        print ("Creating new instances...")
                        sims = []

                    for sim in range(nsim):
                        if loadPrev and sim < len(sims):
                            changed_instance = False
                            G,B,U,C = sims[sim]
                        else:
                            changed_instance = True
                            G = generate_random_dag(N, density)
                            B = 5 * N * budget
                            U = generate_utility(G)
                            C = generate_cost(G)
                            sims.append((G,B,U,C))
                        for solver_index in range(len(solvers)):
                            solver = solvers[solver_index]
                            if solver == "ilp":
                                s_time, s_sol = ilp_time(G,C,B,U)
                            elif solver == "bf":
                                s_time, s_sol = brute_force_time(G,C,B,U,cost)
                            elif solver == "gd":
                                s_time, s_sol = greedy_time(G,C,B,U,cost)
                            sols[sim,solver_index] = s_sol
                            times[sim,solver_index] = s_time
                        progress += 1
                        if verbose: utils.update_progress(progress/total_simulations)
                    if changed_instance:
                        print ("\nTest instances saved for future use.")
                        utils.save_instance(sims,N,density,budget,cost)

                    result_dict.extend(utils.generate_result_dict(N, density, budget, 
                                                                  cost, solvers, sols, times))
                        
                    
    now = datetime.datetime.now()
    csv_file = "simulation/simulation_" + now.strftime("%Y-%m-%d-%H-%M") + ".csv"
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=result_colnums_names)
            writer.writeheader()
            for data in result_dict:
                writer.writerow(data)
    except IOError:
        print("I/O error") 


if __name__ == '__main__':
    simulate()

