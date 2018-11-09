#ifndef CONVERSION_H
#define CONVERSION_H

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"

namespace mv
{

    namespace op
    {

        class Conversion : public SinkOp, public SourceOp
        {

        public:

            Conversion(const std::string &name, Order targetOrder);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);
            void gatherSerialFields() override;
        };

    }

}

#endif // CONVERSION_H
