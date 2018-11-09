#ifndef PASS_MANAGER_HPP_
#define PASS_MANAGER_HPP_

#include <vector>
#include <algorithm>
#include "include/mcm/pass/pass_registry.hpp"
#include "include/mcm/computation/model/computation_model.hpp"
#include "include/mcm/target/target_descriptor.hpp"
#include "include/mcm/base/json/json.hpp"
#include "include/mcm/base/exception/runtime_error.hpp"
#include "include/mcm/logger/log_sender.hpp"

namespace mv
{

    class PassManager : public LogSender
    {

        bool ready_;
        bool completed_;
        bool running_;

        TargetDescriptor targetDescriptor_;
        json::Object compDescriptor_;
        ComputationModel *model_;

        std::vector<std::string> adaptPassQueue_;
        std::vector<std::string> optPassQueue_;
        std::vector<std::string> finalPassQueue_;
        std::vector<std::string> serialPassQueue_;
        std::vector<std::string> validPassQueue_;

        std::string buffer_;

        const std::vector<std::pair<PassGenre, std::vector<std::string>*>> passFlow_ =
        {
            {PassGenre::Validation, &validPassQueue_},
            {PassGenre::Adaptation, &adaptPassQueue_},
            {PassGenre::Validation, &validPassQueue_},
            {PassGenre::Optimization, &optPassQueue_},
            {PassGenre::Validation, &validPassQueue_},
            {PassGenre::Finalization, &finalPassQueue_},
            {PassGenre::Validation, &validPassQueue_},
            {PassGenre::Serialization, &serialPassQueue_}
        };

        std::vector<std::pair<PassGenre, std::vector<std::string>*>>::const_iterator currentStage_;
        std::vector<std::string>::iterator currentPass_;
        json::Object compOutput_;

        static std::string toString(PassGenre passGenre);

    protected:

        

    public:

        PassManager();
        bool initialize(ComputationModel &model, const TargetDescriptor& targetDescriptor, const mv::json::Object& compDescriptor);
        bool enablePass(PassGenre stage, const std::string& pass, int pos = -1);
        bool disablePass(PassGenre stage, const std::string& pass);
        bool disablePass(PassGenre stage);
        bool disablePass();
        std::size_t scheduledPassesCount(PassGenre stage) const;
        const std::vector<std::string>& scheduledPasses(PassGenre stage) const;
        void reset();
        bool validDescriptors() const;
        bool ready() const;
        bool completed() const;
        json::Object& step();
        std::string getLogID() const override;
        
    };

}

#endif // PASS_MANAGER_HPP_
