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

// Operator-specific type aliases using ExecutorTypesAllWithJIT instead of ExecutorTypesAll
using MatXFloatNonHalfTypesAllExecsWithJIT           = TupleToTypes<TypedCartesianProduct<MatXFloatNonHalfTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXNumericNonComplexTypesAllExecsWithJIT      = TupleToTypes<TypedCartesianProduct<MatXNumericNonComplexTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXFloatNonComplexNonHalfTypesAllExecsWithJIT = TupleToTypes<TypedCartesianProduct<MatXFloatNonComplexNonHalfTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXTypesFloatNonComplexAllExecsWithJIT        = TupleToTypes<TypedCartesianProduct<MatXFloatNonComplexTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXTypesNumericAllExecsWithJIT                = TupleToTypes<TypedCartesianProduct<MatXNumericTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXNumericNoHalfTypesAllExecsWithJIT          = TupleToTypes<TypedCartesianProduct<MatXNumericNonHalfTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXComplexNonHalfTypesAllExecsWithJIT         = TupleToTypes<TypedCartesianProduct<MatXComplexNonHalfTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXComplexTypesAllExecsWithJIT                = TupleToTypes<TypedCartesianProduct<MatXComplexTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXAllTypesAllExecsWithJIT                    = TupleToTypes<TypedCartesianProduct<MatXAllTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXTypesFloatAllExecsWithJIT                  = TupleToTypes<TypedCartesianProduct<MatXFloatTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXTypesIntegralAllExecsWithJIT               = TupleToTypes<TypedCartesianProduct<MatXIntegralTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXTypesBooleanAllExecsWithJIT                = TupleToTypes<TypedCartesianProduct<MatXIntegralTuple, ExecutorTypesAllWithJIT>::type>::type;
using MatXTypesCastToFloatAllExecsWithJIT            = TupleToTypes<TypedCartesianProduct<MatXCastToFloatTuple, ExecutorTypesAllWithJIT>::type>::type;

TYPED_TEST_SUITE(OperatorTestsFloatNonHalf,
  MatXFloatNonHalfTypesAllExecsWithJIT);  
TYPED_TEST_SUITE(OperatorTestsNumericNonComplexAllExecs,
                 MatXNumericNonComplexTypesAllExecsWithJIT);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexNonHalfAllExecs,
                 MatXFloatNonComplexNonHalfTypesAllExecsWithJIT);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexAllExecs,
                 MatXTypesFloatNonComplexAllExecsWithJIT);
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexSingleThreadedHostAllExecs,
                 MatXTypesFloatNonComplexSingleThreadedHostAllExecs);
TYPED_TEST_SUITE(OperatorTestsNumericAllExecs,
                 MatXTypesNumericAllExecsWithJIT);                                
TYPED_TEST_SUITE(OperatorTestsNumericNoHalfAllExecs, MatXNumericNoHalfTypesAllExecsWithJIT);          
TYPED_TEST_SUITE(OperatorTestsComplexNonHalfTypesAllExecs, MatXComplexNonHalfTypesAllExecsWithJIT);
TYPED_TEST_SUITE(OperatorTestsComplexTypesAllExecs, MatXComplexTypesAllExecsWithJIT);
TYPED_TEST_SUITE(OperatorTestsAllExecs, MatXAllTypesAllExecsWithJIT);
TYPED_TEST_SUITE(OperatorTestsFloatAllExecs, MatXTypesFloatAllExecsWithJIT);
TYPED_TEST_SUITE(OperatorTestsIntegralAllExecs, MatXTypesIntegralAllExecsWithJIT);
TYPED_TEST_SUITE(OperatorTestsBooleanAllExecs, MatXTypesBooleanAllExecsWithJIT);
TYPED_TEST_SUITE(OperatorTestsCastToFloatAllExecs, MatXTypesCastToFloatAllExecsWithJIT);

// Operator-specific type aliases using ExecutorTypesAllWithoutJIT instead of ExecutorTypesAll
using MatXFloatNonHalfTypesAllExecsWithoutJIT           = TupleToTypes<TypedCartesianProduct<MatXFloatNonHalfTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXNumericNonComplexTypesAllExecsWithoutJIT      = TupleToTypes<TypedCartesianProduct<MatXNumericNonComplexTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXFloatNonComplexNonHalfTypesAllExecsWithoutJIT = TupleToTypes<TypedCartesianProduct<MatXFloatNonComplexNonHalfTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXTypesFloatNonComplexAllExecsWithoutJIT        = TupleToTypes<TypedCartesianProduct<MatXFloatNonComplexTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXTypesNumericAllExecsWithoutJIT                = TupleToTypes<TypedCartesianProduct<MatXNumericTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXNumericNoHalfTypesAllExecsWithoutJIT          = TupleToTypes<TypedCartesianProduct<MatXNumericNonHalfTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXComplexNonHalfTypesAllExecsWithoutJIT         = TupleToTypes<TypedCartesianProduct<MatXComplexNonHalfTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXComplexTypesAllExecsWithoutJIT                = TupleToTypes<TypedCartesianProduct<MatXComplexTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXAllTypesAllExecsWithoutJIT                    = TupleToTypes<TypedCartesianProduct<MatXAllTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXTypesFloatAllExecsWithoutJIT                  = TupleToTypes<TypedCartesianProduct<MatXFloatTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXTypesIntegralAllExecsWithoutJIT               = TupleToTypes<TypedCartesianProduct<MatXIntegralTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXTypesBooleanAllExecsWithoutJIT                = TupleToTypes<TypedCartesianProduct<MatXIntegralTuple, ExecutorTypesAllWithoutJIT>::type>::type;
using MatXTypesCastToFloatAllExecsWithoutJIT            = TupleToTypes<TypedCartesianProduct<MatXCastToFloatTuple, ExecutorTypesAllWithoutJIT>::type>::type;

TYPED_TEST_SUITE(OperatorTestsFloatNonHalfWithoutJIT,
  MatXFloatNonHalfTypesAllExecsWithoutJIT);  
TYPED_TEST_SUITE(OperatorTestsNumericNonComplexAllExecsWithoutJIT,
                 MatXNumericNonComplexTypesAllExecsWithoutJIT);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexNonHalfAllExecsWithoutJIT,
                 MatXFloatNonComplexNonHalfTypesAllExecsWithoutJIT);  
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexAllExecsWithoutJIT,
                 MatXTypesFloatNonComplexAllExecsWithoutJIT);
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexSingleThreadedHostAllExecsWithoutJIT,
                 MatXTypesFloatNonComplexSingleThreadedHostAllExecs);
TYPED_TEST_SUITE(OperatorTestsNumericAllExecsWithoutJIT,
                 MatXTypesNumericAllExecsWithoutJIT);                                
TYPED_TEST_SUITE(OperatorTestsNumericNoHalfAllExecsWithoutJIT, MatXNumericNoHalfTypesAllExecsWithoutJIT);          
TYPED_TEST_SUITE(OperatorTestsComplexNonHalfTypesAllExecsWithoutJIT, MatXComplexNonHalfTypesAllExecsWithoutJIT);
TYPED_TEST_SUITE(OperatorTestsComplexTypesAllExecsWithoutJIT, MatXComplexTypesAllExecsWithoutJIT);
TYPED_TEST_SUITE(OperatorTestsAllExecsWithoutJIT, MatXAllTypesAllExecsWithoutJIT);
TYPED_TEST_SUITE(OperatorTestsFloatAllExecsWithoutJIT, MatXTypesFloatAllExecsWithoutJIT);
TYPED_TEST_SUITE(OperatorTestsIntegralAllExecsWithoutJIT, MatXTypesIntegralAllExecsWithoutJIT);
TYPED_TEST_SUITE(OperatorTestsBooleanAllExecsWithoutJIT, MatXTypesBooleanAllExecsWithoutJIT);
TYPED_TEST_SUITE(OperatorTestsCastToFloatAllExecsWithoutJIT, MatXTypesCastToFloatAllExecsWithoutJIT);

// Template class declarations for WithoutJIT types
template <typename TensorType>
class OperatorTestsFloatNonHalfWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericNonComplexAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplexNonHalfAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplexAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatNonComplexSingleThreadedHostAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsNumericNoHalfAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsComplexNonHalfTypesAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsComplexTypesAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsFloatAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsIntegralAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsBooleanAllExecsWithoutJIT : public ::testing::Test {};

template <typename TensorType>
class OperatorTestsCastToFloatAllExecsWithoutJIT : public ::testing::Test {};

} // namespace test
} // namespace matx 