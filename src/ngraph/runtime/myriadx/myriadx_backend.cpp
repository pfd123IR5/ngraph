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

#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/runtime/myriadx/myriadx_backend.hpp"
#include "ngraph/runtime/myriadx/myriadx_layout.hpp"
#include "ngraph/runtime/myriadx/myriadx_op_convolution.hpp"
#include "ngraph/runtime/myriadx/myriadx_tensor_view.hpp"
#include "ngraph/runtime/myriadx/compilation_unit.hpp"
#include "ngraph/runtime/myriadx/compositional_model_recorder.hpp"
#include "ngraph/runtime/myriadx/logger.hpp"

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter_vector.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

#include <math.h>
#include <iostream>

using namespace std;
using namespace ngraph;

using myriadx_space = runtime::myriadx::MyriadXLayout;

#define USE_MYRIADX_CUSTOM_KERNELS 0

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/op_tbl.hpp"
};
#undef NGRAPH_OP

static OP_TYPEID get_typeid(const string& s)
{
// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// {"Abs", OP_TYPEID::Abs},
// {"Acos", OP_TYPEID::Acos},
// ...
#define NGRAPH_OP(a, b) {#a, OP_TYPEID::a},
    static const unordered_map<string, OP_TYPEID> typeid_map{
#include "ngraph/op/op_tbl.hpp"
    };
#undef NGRAPH_OP
    auto it = typeid_map.find(s);
    if (it == typeid_map.end())
    {
        throw unsupported_op("Unsupported op '" + s + "'");
    }
    return it->second;
}

static void arguments_check(const shared_ptr<Node>& op, size_t input, size_t output)
{
    if (op->get_input_size() != input || op->get_output_size() != output)
    {
        ostringstream os;
        os << "Operation \"" << op->description() << "\" input and output sizes mismatch."
           << " Expected input size=" << input << ", provided=" << op->get_input_size()
           << ". Expected output size=" << output << ", provided=" << op->get_output_size();
        throw invalid_argument(os.str());
    }
}

static const string& get_input_name(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_tensor().get_name();
}

static const string& get_output_name(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_tensor().get_name();
}

static const Shape& get_input_shape(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_shape();
}

static const Shape& get_output_shape(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_shape();
}

static const element::Type& get_input_type(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_tensor().get_element_type();
}

static const element::Type& get_output_type(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_tensor().get_element_type();
}

static void do_unary_operation(
                               const shared_ptr<Node>& op)
{
    arguments_check(op, 1, 1);

}

static void do_pooling_operation(
                                 const shared_ptr<Node>& op,
                                 const Shape& pool_shape,
                                 const Strides& pool_strides,
                                 const Shape& pad_below)
{
    arguments_check(op, 1, 1);

}

static void do_logical_operation(
                                 const shared_ptr<Node>& op,
                                 const string& operation)
{
    arguments_check(op, 2, 1);

}

// This function needed to only change the name of the data in topology
// No real data copy needed
static void do_equal_propagation(
                                 const string& input_name,
                                 const string& output_name)
{
}

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
std::cout << "calling  myriadx BE constructor" << std::endl;
    return new runtime::myriadx::MyriadXBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

runtime::myriadx::MyriadXBackend::MyriadXBackend()
{
std::cout << "in myriadx BE constructor" << std::endl;
}

shared_ptr<runtime::Tensor>
    runtime::myriadx::MyriadXBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    return make_shared<runtime::myriadx::MyriadXTensorView>(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::myriadx::MyriadXBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::myriadx::MyriadXTensorView>(
        element_type, shape, memory_pointer);
}

bool runtime::myriadx::MyriadXBackend::compile(shared_ptr<Function> func)
{
std::cout << "in MB compile" << std::endl;

    ngraph::pass::Manager pass_manager;

    pass_manager.register_pass<ngraph::pass::NopElimination>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
    pass_manager.register_pass<ngraph::pass::ReshapeElimination>();

    // GetOutputElementElimination must be after CommonSubexpressionElimination
    pass_manager.register_pass<ngraph::pass::GetOutputElementElimination>();

    pass_manager.run_passes(func);

    for (shared_ptr<Node> op : func->get_ops())
    {
// We want to check that every OP_TYPEID enumeration is included in the list.
// These GCC flags enable compile-time checking so that if an enumeration
// is not in the list an error is generated.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
        switch (get_typeid(op->description()))
        {
        case OP_TYPEID::Parameter:
        {
            break;
        }
        case OP_TYPEID::Result:
        {
            break;
        }
        case OP_TYPEID::GetOutputElement:
        {
            break;
        }
        case OP_TYPEID::Slice:
        {
            break;
        }
        case OP_TYPEID::Select:
        {
            break;
        }
        case OP_TYPEID::Reverse:
        {
            break;
        }
        case OP_TYPEID::Convert:
        {
            break;
        }
        case OP_TYPEID::Concat:
        {
            break;
        }
        case OP_TYPEID::Softmax:
        {
            break;
        }
        case OP_TYPEID::Add:
        {
            break;
        }
        case OP_TYPEID::Multiply:
        {
            break;
        }
        case OP_TYPEID::Divide:
        {
            break;
        }
        case OP_TYPEID::Maximum:
        {
            break;
        }
        case OP_TYPEID::Minimum:
        {
            break;
        }
        case OP_TYPEID::Constant:
        {
            break;
        }
        case OP_TYPEID::Dot:
        {
            break;
        }
        case OP_TYPEID::MaxPool:
        {
            break;
        }
        case OP_TYPEID::MaxPoolBackprop:
        {
            arguments_check(op, 2, 1);

            break;
        }
        case OP_TYPEID::AvgPool:
        {
            break;
        }
        case OP_TYPEID::AvgPoolBackprop:
        {
            break;
        }
        case OP_TYPEID::Broadcast:
        {
            break;
        }
        case OP_TYPEID::Sum:
        {
            break;
        }
        case OP_TYPEID::Product:
        {
            break;
        }
        case OP_TYPEID::Reshape:
        {
            break;
        }
        case OP_TYPEID::Negative:
        {
            break;
        }
        case OP_TYPEID::Relu:
        {
            break;
        }
        case OP_TYPEID::ReluBackprop:
        {
            break;
        }
        case OP_TYPEID::Reduce:
        {
            break;
        }
        case OP_TYPEID::Abs:
        {
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            break;
        }
        case OP_TYPEID::Tanh:
        {
            break;
        }
        case OP_TYPEID::Sin:
        {
            break;
        }
        case OP_TYPEID::Asin:
        {
            break;
        }
        case OP_TYPEID::Sinh:
        {
            break;
        }
        case OP_TYPEID::Cos:
        {
            break;
        }
        case OP_TYPEID::Acos:
        {
            break;
        }
        case OP_TYPEID::Cosh:
        {
            break;
        }
        case OP_TYPEID::Log:
        {
            break;
        }
        case OP_TYPEID::Exp:
        {
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            break;
        }
        case OP_TYPEID::SigmoidBackprop:
        {
            break;
        }
        case OP_TYPEID::Not:
        {
            break;
        }
        case OP_TYPEID::Greater:
        {
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            break;
        }
        case OP_TYPEID::Equal:
        {
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            break;
        }
        case OP_TYPEID::Less:
        {
            break;
        }
        case OP_TYPEID::LessEq:
        {
            break;
        }
        case OP_TYPEID::And:
        {
            break;
        }
        case OP_TYPEID::Or:
        {
            break;
        }
        case OP_TYPEID::Subtract:
        {
            break;
        }
        case OP_TYPEID::Power:
        {
            break;
        }
        case OP_TYPEID::Atan:
        {
            break;
        }
        case OP_TYPEID::Ceiling:
        {
            break;
        }
        case OP_TYPEID::Floor:
        {
            break;
        }
        case OP_TYPEID::Sign:
        {
            break;
        }
        case OP_TYPEID::Tan:
        {
            break;
        }
        case OP_TYPEID::Pad:
        {
            break;
        }
        case OP_TYPEID::Convolution:
        {
std::cout << "in case do convolution " << std::endl;

/*

            arguments_check(op, 2, 1);

            const shared_ptr<op::Convolution> conv_op = static_pointer_cast<op::Convolution>(op);
            const Strides& win_stride = conv_op->get_window_movement_strides();
            const Strides& win_dilation = conv_op->get_window_dilation_strides();
            const Strides& data_dilation = conv_op->get_data_dilation_strides();
            const CoordinateDiff& pad_below = conv_op->get_padding_below();
            const CoordinateDiff& pad_above = conv_op->get_padding_above();

            // clDNN has quite limited support for Convolution operation
            // following are the checks to go with workaround
            if ((win_stride.size() > 2) || (pad_below.size() > 2 || pad_above.size() > 2) ||
                (pad_below.at(0) != pad_above.at(0) || pad_below.at(1) != pad_above.at(1)) ||
                (win_dilation.size() > 2) ||
                (data_dilation.size() > 2 || data_dilation.at(0) != 1 || data_dilation.at(1) != 1))
            {
                do_convolution_operation(
                                         get_input_name(op, 0),
                                         get_input_shape(op, 0),
                                         get_input_name(op, 1),
                                         get_input_shape(op, 1),
                                         get_output_name(op),
                                         get_output_shape(op),
                                         get_output_type(op),
                                         conv_op->get_padding_below(),
                                         conv_op->get_window_movement_strides(),
                                         conv_op->get_window_dilation_strides(),
                                         conv_op->get_data_dilation_strides(),
                                         0,
                                         1,
                                         1,
                                         "input[batch][input_channel]",
                                         "filter[output_channel][input_channel]",
                                         "output[batch][output_channel]",
                                         false);
*/
        auto unit = new mv::CompilationUnit("test_backend");

        mv::Logger::setVerboseLevel(mv::Logger::VerboseLevel::VerboseDebug);
        unit->loadTargetDescriptor(mv::Target::ma2480);

    mv::CompositionalModel& test_cm = unit->model();

    // Compose minimal functional computation model - one computation operation of type conv2D
    auto input1 = test_cm.input({32, 32, 1}, mv::DTypeType::Float16, mv::Order("WHC"));
    std::vector<double> weights1Data({ 0.1111, 0.1121, 0.1131, 0.1141, 0.1151, 0.1161, 0.1171, 0.1181, 0.1191});
    auto weights1 = test_cm.constant(weights1Data, {3, 3, 1, 1}, mv::DTypeType::Float16, mv::Order("NCHW"));
    auto conv1 = test_cm.conv2D(input1, weights1, {4, 4}, {0, 0, 0, 0});
    auto output1 = test_cm.output(conv1);

std::cout << "in compile conv, setting comilation descriptors" << std::endl;
   std::string blobName = "./test_conv.blob";
    unit->compilationDescriptor()["GenerateBlob"]["fileName"] = blobName;
    unit->compilationDescriptor()["GenerateBlob"]["enableFileOutput"] = true;
    unit->compilationDescriptor()["GenerateBlob"]["enableRAMOutput"] = false;
    unit->compilationDescriptor()["GenerateDot"]["output"] = std::string("blob_output_conv_01.dot");
    unit->compilationDescriptor()["GenerateDot"]["scope"] = std::string("OpControlModel");
    unit->compilationDescriptor()["GenerateDot"]["content"] = std::string("full");
    unit->compilationDescriptor()["GenerateDot"]["html"] = true;
    unit->compilationDescriptor()["MarkHardwareOperations"]["disableHardware"] = true;
    unit->loadTargetDescriptor("/home/patd/MCM/mcmCompiler/config/target/ma2480.json");
    unit->initialize();
    //unit.passManager().disablePass(mv::PassGenre::Validation);
    unit->passManager().disablePass(mv::PassGenre::Serialization);
    unit->passManager().enablePass(mv::PassGenre::Serialization, "GenerateBlob");
    unit->initialize();
std::cout << "in compile conv, running compile" << std::endl;
    auto compOutput = unit->run();


            break;
        }
        case OP_TYPEID::ConvolutionBackpropFilters:
        {
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData:
        {
            break;
        }
        case OP_TYPEID::Min:
        {
            break;
        }
        case OP_TYPEID::Max:
        {
            break;
        }
        case OP_TYPEID::OneHot:
        {
            break;
        }
        case OP_TYPEID::ArgMax:
        {
            break;
        }
        case OP_TYPEID::ArgMin:
        {
            break;
        }
        case OP_TYPEID::LRN:
        {
            break;
        }
        case OP_TYPEID::AllReduce:
        case OP_TYPEID::FunctionCall:
        case OP_TYPEID::Dequantize:
        case OP_TYPEID::Quantize:
        case OP_TYPEID::ReduceWindow:
        case OP_TYPEID::ReplaceSlice:
        case OP_TYPEID::ReverseSequence:
        case OP_TYPEID::SelectAndScatter:
        case OP_TYPEID::StopGradient:
        case OP_TYPEID::TopK:
        case OP_TYPEID::BatchNormInference:
        case OP_TYPEID::BatchNormTraining:
        case OP_TYPEID::BatchNormTrainingBackprop:
        {
            throw unsupported_op("Unsupported op '" + op->description() +
                                 "' in MyriadX back end.");
        }
#pragma GCC diagnostic pop
        }
    }

    return true;
}

bool runtime::myriadx::MyriadXBackend::call(shared_ptr<Function> func,
                                              const vector<shared_ptr<runtime::Tensor>>& outputs,
                                              const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    validate_call(func, outputs, inputs);

    // Process input parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_parameters and inputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < inputs.size(); i++)
    {
        shared_ptr<runtime::myriadx::MyriadXTensorView> tv =
            static_pointer_cast<runtime::myriadx::MyriadXTensorView>(inputs[i]);
        const op::ParameterVector& input_params = func->get_parameters();
        const string& tensor_name = input_params[i]->get_output_tensor().get_name();
    }

    // Execute network
std::cout << "in executing network in MBE runtime call" << std::endl;


    // Process output parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_results and outputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < func->get_output_size(); i++)
    {
std::cout << "in executing network for output " << i <<std::endl;
        shared_ptr<runtime::myriadx::MyriadXTensorView> ngraph_res =
            static_pointer_cast<runtime::myriadx::MyriadXTensorView>(outputs[i]);
        const string& tensor_name = func->get_output_op(i)->get_output_tensor().get_name();

    }
std::cout << "returning from MBE call" << std::endl;

    return true;
}
