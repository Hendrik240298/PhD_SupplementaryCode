# MORe DWR implementation in FEniCS
## Idea 

We first show Hesthaven's illustration of the *POD-Greedy Algorithm* in Algorithm 1. We use $\mathbb{P}_h \subset \mathbb{P}$ as a surrogate of the parameter space.


In our case, we take advantage of the MORe DWR nature and modify the enrichment process. We propose two different approaches to decide on which parameter-timestep pair to enrich. The first algorithm is nearer to the original approach, whereas the second approach can be regarded as more unconventional. The performance of both will be discussed in the Results section.