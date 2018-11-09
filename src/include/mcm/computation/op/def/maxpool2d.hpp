#ifndef MAXPOOL2D_HPP_
#define MAXPOOL2D_HPP_

#include "include/mcm/computation/op/pool2d_op.hpp"
#include "include/mcm/deployer/blob_serialization/myriadX_hardware_descriptors.hpp"

namespace mv
{

    namespace op
    {

        /// \todo Add assertions (dimensions)
        class MaxPool2D : public Pool2DOp
        {

        public:

            MaxPool2D(std::array<unsigned short, 2> kernelSize, std::array<unsigned short, 2> stride, std::array<unsigned short, 4> padding, const std::string &name);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);
            void gatherSerialFields() override;
        };

    }

}

#endif // MAXPOOL2D_HPP_
