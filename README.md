## Community Detection With Genetic Algorithm

From time to time, social network analysis requires us to classify the number groups made up a given network. This simple module is a python implementation of a locus based genetic algorithm for community detection by Clara Pizzuti.[1] This algorithm is particularly good at detecting communities in which we don't know their number beforehand. This algorithm finds communities by its structure: it classify densely connected nodes into a group. Although it runs slow, this algorithm will help us in preliminary analysis of a social network. 

#### Example use:

```
import gacomm as gc

nodes = [0,1,2,3,4,5,6,7,8,9,10]
edges = [(0,1),(0,4),(1,2),(2,3),(1,3),(3,0),(0,2),(4,5),(5,6),(6,7),(10,8),(10,9),(8,9),(8,7),(9,7),(7,10)]

gc.community(N,E)

# Possible output
# [[0,1,2,3],[4,5,6],[7,8,9,10]]

```

Reference:

[1. **Pizzuti**, C. (2008). Ga-net: A genetic algorithm for commu-nity detection in social networks. In *Inter conf on par-allel problem solving from nature*, pages 1081–1090.Springer.](https://www.researchgate.net/publication/220701568_GA-Net_A_Genetic_Algorithm_for_Community_Detection_in_Social_Networks)