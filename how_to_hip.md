# How to make a MLIP HIP

We need the following steps:
1. Modify the message passing function to return the messages before they are aggregated
2. Turn the l=0,1,2 components of the messages and node features into 3x3 Hessian subblocks
3. Compute indices which messages belong to which off-diagonal Hessian entry from edge_index
4. Compute indices which nodes belong to which diagonal Hessian entry 
5. Add the message features to the off-diagonal Hessian entries
6. Add the node features to the diagonal Hessian entries

Optional but recommended: use a larger graph for the Hessian messages
7. Modify the graph generation function to generate a second graph for the Hessian messages
8. Modify the model to use the second graph for the Hessian messages

## Graph and index computation
Why do we need indices?
Naively we could loop over the messages/nodes, lookup how the messages/nodes are ordered relative to the edges/atoms in the Hessian, and then add the messages/nodes into the Hessian entries.
The problem is that using loops is slow. We found that torch's `_index_add` is ~10^4 times faster than looping.
Note that we store the Hessian as a 1D array to allow for batching, which adds a bit of complexity to the index computation. 
We always compute the graph and the indices on the fly during the forward pass or in the dataloader (less code and same speed)

