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

#include <memory>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/runtime/myriadx/myriadx_layout.hpp"
#include "ngraph/runtime/myriadx/myriadx_tensor_view.hpp"

using namespace ngraph;
using namespace std;

runtime::myriadx::MyriadXTensorView::MyriadXTensorView(const element::Type& element_type,
                                                          const Shape& shape,
                                                          void* memory_pointer)
    : runtime::Tensor(make_shared<descriptor::Tensor>(element_type, shape, "external"))
{

}

void runtime::myriadx::MyriadXTensorView::write(const void* source,
                                                  size_t tensor_offset,
                                                  size_t n)
{
}

void runtime::myriadx::MyriadXTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
}
