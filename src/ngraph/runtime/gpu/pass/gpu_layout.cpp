//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>

#include "gpu_layout.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/runtime/gpu/gpu_op_annotations.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace pass
            {
                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::ReplaceSlice)
                {
                    auto rep_slice = static_cast<ngraph::op::ReplaceSlice*>(node.get());

                    auto op_annotations = rep_slice->get_op_annotations();
                    if (op_annotations)
                    {
                        // pass-through
                        op_annotations->add_in_place_oi_pair({0, 0, true});
                    }
                    else
                    {
                        op_annotations = std::make_shared<ngraph::runtime::gpu::GPUOpAnnotations>();
                        // pass-through
                        op_annotations->add_in_place_oi_pair({0, 0, true});
                        rep_slice->set_op_annotations(op_annotations);
                    }
                }
                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::Reshape)
                {
                    auto reshape = static_cast<ngraph::op::Reshape*>(node.get());
                    if (reshape->get_is_transpose())
                    {
                        return;
                    }
                    // Shape change only, tensor in native layout can be
                    // forwarded to output
                    auto op_annotations = reshape->get_op_annotations();
                    if (op_annotations)
                    {
                        // pass-through
                        op_annotations->add_in_place_oi_pair({0, 0, false});
                    }
                    else
                    {
                        op_annotations = std::make_shared<ngraph::runtime::gpu::GPUOpAnnotations>();
                        // pass-through
                        op_annotations->add_in_place_oi_pair({0, 0, false});
                        reshape->set_op_annotations(op_annotations);
                    }
                }

                template <>
                void GPULayout::LAYOUT_DECL(ngraph::op::TopK)
                {
                    auto topk = static_cast<ngraph::op::TopK*>(node.get());
                    auto topk_axis = topk->get_top_k_axis();
                    auto topk_k = topk->get_k();
                    auto parent_node = topk->get_argument(0);

                    auto in_shape = topk->get_input_shape(0);
                    auto out_shape = in_shape;
                    out_shape[topk_axis] = topk_k;
                    size_t ndim = static_cast<size_t>(in_shape.size());
                    AxisVector reshape_axis_order(ndim);
                    iota(reshape_axis_order.begin(), reshape_axis_order.end(), 0);
                    reshape_axis_order.erase(reshape_axis_order.begin() + topk_axis);
                    reshape_axis_order.push_back(topk_axis);

                    Shape pre_reshape_out;
                    for (size_t j = 0; j < ndim; j++)
                    {
                        pre_reshape_out.push_back(in_shape[reshape_axis_order[j]]);
                    }

                    Shape pre_2d_reshape_out(2);
                    pre_2d_reshape_out[1] = pre_reshape_out[ndim - 1];
                    pre_2d_reshape_out[0] = static_cast<size_t>(
                        ngraph::shape_size(pre_reshape_out) / pre_2d_reshape_out[1]);

                    auto pre_reshape = make_shared<ngraph::op::Reshape>(
                        parent_node, reshape_axis_order, pre_reshape_out);

                    AxisVector axis_order(ndim);
                    iota(axis_order.begin(), axis_order.end(), 0);
                    auto pre_2d_reshape = make_shared<ngraph::op::Reshape>(
                        pre_reshape, axis_order, pre_2d_reshape_out);

                    insert_new_node_between(parent_node, node, pre_reshape);
                    parent_node = topk->get_argument(0);
                    insert_new_node_between(parent_node, node, pre_2d_reshape);

                    NodeVector goes = op::get_output_elements(node);

                    auto new_topk = make_shared<ngraph::op::TopK>(topk->get_argument(0),
                                                                  1,
                                                                  topk->get_index_element_type(),
                                                                  topk->get_k(),
                                                                  topk->get_compute_max());

                    ngraph::replace_node(node, new_topk);
                    NodeVector new_goes;
                    for (auto& goe : goes)
                    {
                        auto out_idx =
                            std::dynamic_pointer_cast<op::GetOutputElement>(goe)->get_n();
                        auto new_goe = std::make_shared<op::GetOutputElement>(new_topk, out_idx);
                        ngraph::replace_node(goe, new_goe);
                        new_goes.push_back(new_goe);
                    }

                    Shape reordered_out_shape;
                    for (size_t j = 0; j < ndim; j++)
                    {
                        reordered_out_shape.push_back(out_shape[reshape_axis_order[j]]);
                    }

                    NodeVector post_2d_reshapes;
                    for (auto& new_goe : new_goes)
                    {

                        for (auto node : new_goe->get_users())
                        {

                            for (size_t i = 0; i < node->get_input_size(); i++)
                            {
                                if (node->get_argument(i) == new_goe)
                                {
                                    auto post_2d_reshape = make_shared<ngraph::op::Reshape>(
                                        new_goe, AxisVector{0, 1}, reordered_out_shape);

                                    node->get_inputs().at(i).replace_output(
                                        post_2d_reshape->get_outputs().at(0));
                                    post_2d_reshapes.push_back(post_2d_reshape);
                                }
                            }
                        }
                    }


                    axis_order.pop_back();
                    axis_order.insert(axis_order.begin()+ topk_axis, 1, ndim-1);

                    for (auto& new_reshape : post_2d_reshapes)
                    {

                        for (auto node : new_reshape->get_users())
                        {
                            for (size_t i = 0; i < node->get_input_size(); i++)
                            {
                                if (node->get_argument(i) == new_reshape)
                                {
                                    auto post_reshape = make_shared<ngraph::op::Reshape>(
                                        new_reshape, axis_order, out_shape);
 
                                    node->get_inputs().at(i).replace_output(
                                        post_reshape->get_outputs().at(0));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#define TI(x) type_index(typeid(x))

static const runtime::gpu::pass::LayoutOpMap s_dispatcher{
    {TI(ngraph::op::ReplaceSlice),
     &runtime::gpu::pass::GPULayout::layout<ngraph::op::ReplaceSlice>},
    {TI(ngraph::op::Reshape), &runtime::gpu::pass::GPULayout::layout<ngraph::op::Reshape>},
    {TI(ngraph::op::TopK), &runtime::gpu::pass::GPULayout::layout<ngraph::op::TopK>},
};

bool runtime::gpu::pass::GPULayout::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const auto& node : nodes)
    {
        auto& n = *node;
        auto handler = s_dispatcher.find(TI(n));
        if (handler != s_dispatcher.end())
        {
            handler->second(m_external_function, node);
        }
    }

    return false;
}
