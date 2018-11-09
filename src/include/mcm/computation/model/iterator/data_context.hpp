#ifndef DATA_CONTEXT_HPP_
#define DATA_CONTEXT_HPP_

#include <map>
#include <string>
#include "include/mcm/computation/model/iterator/model_iterator.hpp"
#include "include/mcm/tensor/tensor.hpp"

namespace mv
{  

    namespace Data
    {

        using OpListIterator = IteratorDetail::OpIterator<computation_graph::first_graph, computation_graph::first_graph::node_list_iterator, ComputationOp, DataFlow>;
        using OpReverseListIterator = IteratorDetail::OpIterator<computation_graph::first_graph, computation_graph::first_graph::node_reverse_list_iterator, ComputationOp, DataFlow>;
        using OpDFSIterator = IteratorDetail::OpIterator<computation_graph::first_graph, computation_graph::first_graph::node_dfs_iterator, ComputationOp, DataFlow>;
        using OpBFSIterator = IteratorDetail::OpIterator<computation_graph::first_graph, computation_graph::first_graph::node_bfs_iterator, ComputationOp, DataFlow>;
        using OpChildIterator = IteratorDetail::OpIterator<computation_graph::first_graph, computation_graph::first_graph::node_child_iterator, ComputationOp, DataFlow>;
        using OpParentIterator = IteratorDetail::OpIterator<computation_graph::first_graph, computation_graph::first_graph::node_child_iterator, ComputationOp, DataFlow>;
        using OpSiblingIterator = IteratorDetail::OpIterator<computation_graph::first_graph, computation_graph::first_graph::node_sibling_iterator, ComputationOp, DataFlow>;
        
        using FlowListIterator = IteratorDetail::FlowIterator<computation_graph::first_graph, computation_graph::first_graph::edge_list_iterator, DataFlow, ComputationOp>;
        using FlowReverseListIterator = IteratorDetail::FlowIterator<computation_graph::first_graph, computation_graph::first_graph::edge_reverse_list_iterator, DataFlow, ComputationOp>;
        using FlowDFSIterator = IteratorDetail::FlowIterator<computation_graph::first_graph, computation_graph::first_graph::edge_dfs_iterator, DataFlow, ComputationOp>;
        using FlowBFSIterator = IteratorDetail::FlowIterator<computation_graph::first_graph, computation_graph::first_graph::edge_bfs_iterator, DataFlow, ComputationOp>;
        using FlowChildIterator = IteratorDetail::FlowIterator<computation_graph::first_graph, computation_graph::first_graph::edge_child_iterator, DataFlow, ComputationOp>;
        using FlowParentIterator = IteratorDetail::FlowIterator<computation_graph::first_graph, computation_graph::first_graph::edge_child_iterator, DataFlow, ComputationOp>;
        using FlowSiblingIterator = IteratorDetail::FlowIterator<computation_graph::first_graph, computation_graph::first_graph::edge_sibling_iterator, DataFlow, ComputationOp>;

        using TensorIterator = IteratorDetail::ModelValueIterator<std::map<std::string, std::shared_ptr<Tensor>>::iterator, Tensor>;

    }

}

#endif // DATA_CONTEXT_HPP_