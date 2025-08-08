#pragma once

#include "matx.h"
#include "test_types.h"
#include "gtest/gtest.h"

namespace matx {
namespace test {

template <typename TensorType>
class OperatorTestsComplex : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloat : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumeric : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericNonComplex : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplex : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonHalf : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplexNonHalf : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsIntegral : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsBoolean : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatHalf : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericNoHalf : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsAll : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericNonComplexAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplexNonHalfAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericNoHalfAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsComplexNonHalfTypesAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsComplexTypesAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplexAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplexSingleThreadedHostAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsIntegralAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsBooleanAllExecs : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsCastToFloatAllExecs : public ::testing::Test {};

TYPED_TEST_SUITE(OperatorTestsFloatNonHalf,
  MatXFloatNonHalfTypesAllExecs);  
TYPED_TEST_SUITE(OperatorTestsNumericNonComplexAllExecs,
                 MatXNumericNonComplexTypesAllExecs);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexNonHalfAllExecs,
                 MatXFloatNonComplexNonHalfTypesAllExecs);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexAllExecs,
                 MatXTypesFloatNonComplexAllExecs);
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexSingleThreadedHostAllExecs,
                 MatXTypesFloatNonComplexSingleThreadedHostAllExecs);
TYPED_TEST_SUITE(OperatorTestsNumericAllExecs,
                 MatXTypesNumericAllExecs);                                
TYPED_TEST_SUITE(OperatorTestsNumericNoHalfAllExecs, MatXNumericNoHalfTypesAllExecs);          
TYPED_TEST_SUITE(OperatorTestsComplexNonHalfTypesAllExecs, MatXComplexNonHalfTypesAllExecs);
TYPED_TEST_SUITE(OperatorTestsComplexTypesAllExecs, MatXComplexTypesAllExecs);
TYPED_TEST_SUITE(OperatorTestsAllExecs, MatXAllTypesAllExecs);
TYPED_TEST_SUITE(OperatorTestsFloatAllExecs, MatXTypesFloatAllExecs);
TYPED_TEST_SUITE(OperatorTestsIntegralAllExecs, MatXTypesIntegralAllExecs);
TYPED_TEST_SUITE(OperatorTestsBooleanAllExecs, MatXTypesBooleanAllExecs);
TYPED_TEST_SUITE(OperatorTestsCastToFloatAllExecs, MatXTypesCastToFloatAllExecs);


} // namespace test
} // namespace matx 