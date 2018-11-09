#ifndef BATCH_NORM_HPP_
#define BATCH_NORM_HPP_

#include "include/mcm/computation/op/source_op.hpp"
#include "include/mcm/computation/op/sink_op.hpp"


namespace mv
{
    namespace op
    {

        class BatchNorm : public SourceOp, public SinkOp
        {

        public:

            BatchNorm(double varianceEps, const std::string &name);
            Tensor getOutputDef(std::size_t idx);
            bool isHardwarizeable(mv::json::Object& targetDescriptor);

        };

    }

}

#endif // BATCH_NORM_HPP_
