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

#include "ngraph/runtime/myriadx/myriadx_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

runtime::myriadx::MyriadXLayout::MyriadXLayout(const descriptor::Tensor& tv)
    : TensorLayout(tv)
{
}

size_t runtime::myriadx::MyriadXLayout::get_index_offset(const vector<size_t>& indices)
{
    if (indices.size() != strides.size())
    {
        throw ngraph_error("Indices have incorrect rank");
    }

    return true;
}

bool runtime::myriadx::MyriadXLayout::
    operator==(const descriptor::layout::TensorLayout& other) const
{
    const MyriadXLayout* p_other = dynamic_cast<const MyriadXLayout*>(&other);
    if (!p_other)
    {
        return false;
    }

    return true;
}

