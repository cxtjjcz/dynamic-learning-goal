# Dyanmic Learning Goal
## Utility Maximizing Learning Plan Solver

If you want to do simluations with the solver, plese first create a folder under root directory called "simulation".


Then, in your Terminal, run
```
~ python3 UMLP_sim.py
```
The parameters that you can specify are in the following table:

| flag | Description | Example
| ----------- | ----------- | ----------- |
| `n` | The range of nodes in randomly generated graphs in the form of `[start,end,step]`. | `[30,40,10]` | 
| `density` | The range of the edge density of the randomly generated graphs in the form of `[start,end,step]` |`[0.2,0.3,0.1]` |
| `nsim` | The number of simulations for each set of parameters | `30` |
| `solver` | The type(s) of solver used in the form of `[x,y,...]` (brute force: bf; integer linear program: ilp; greedy: gd) | `[bf, ilp ,gd]`|
|`maxlearnP`| The range of the maximum portion of knowledge points that user can learn in the form of `[start,end,step]`  | `[0.166,0.166,0.1]`| 
|`costType`| The cost function type used for simulation in the form of `[x,y,...]` (add: additive; mono: monotone; sub: submodular) |`[add, mono, sub]` |
