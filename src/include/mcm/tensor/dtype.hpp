#ifndef MV_TENSOR_DTYPE_HPP_
#define MV_TENSOR_DTYPE_HPP_

#include <string>
#include <unordered_map>
#include <functional>
#include "include/mcm/base/exception/dtype_error.hpp"

namespace mv
{

    enum class DTypeType
    {
        Float16
    };

    struct DTypeTypeHash
    {
        template <typename T>
        std::size_t operator()(T t) const
        {
            return static_cast<std::size_t>(t);
        }
    };

    class DType : public LogSender
    {

    private:

        static const std::unordered_map<DTypeType, std::string, DTypeTypeHash> dTypeStrings_;
        DTypeType dType_;

    public:

        DType();
        DType(DTypeType value);
        DType(const DType& other);
        DType(const std::string& value);

        std::string toString() const;

        DType& operator=(const DType& other);
        DType& operator=(const DTypeType& other);
        bool operator==(const DType& other) const;
        bool operator==(const DTypeType& other) const;
        bool operator!=(const DType& other) const;
        bool operator!=(const DTypeType& other) const;
        operator DTypeType() const;

        std::string getLogID() const override;

    };

}

#endif // MV_TENSOR_DTYPE_HPP_
