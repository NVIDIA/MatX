////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;

class NvtxIOTests : public ::testing::Test 
{
protected:
  
  tensor_t<float, 1> data {{10}};
  
  void SetUp() override
  {
    data.SetVals({0,1,2,3,4,5,6,7,8,9});
  }

};

//
// Test that we get a unique ID for subsequent calls for unique RangeIDs
//
TEST_F(NvtxIOTests, testIDGeneration)
{
  MATX_ENTER_HANDLER();
  
  int rangeID1 = getNVTX_Range_ID( );
  int rangeID2 = getNVTX_Range_ID( );
  
  ASSERT_NE(rangeID1, rangeID2);
  
  MATX_EXIT_HANDLER();
}

//
// Test that we can set the globalLogLevel
//
TEST_F(NvtxIOTests, testSetNVTXLogLevel)
{
  MATX_ENTER_HANDLER();
  
  setNVTXLogLevel(matx_nvxtLogLevels::MATX_NVTX_LOG_ALL);
  
  ASSERT_EQ(MATX_NVTX_LOG_ALL, globalNvtxLevel);
  
  MATX_EXIT_HANDLER();
}


//
// Test that we can register a new event *compilation test only*
//
TEST_F(NvtxIOTests, testRegisterEvent)
{
  MATX_ENTER_HANDLER();
  
  int registerId  = 0;
  nvtxRangeId_t eventId = nvtxRangeStartA("Test range");
  
  registerEvent(registerId, eventId);
  
  endEvent( 0 ); // calls nvtxRangeEnd for us
  
  MATX_EXIT_HANDLER();
}


//
// Test Full NvtxEvent
//
TEST_F(NvtxIOTests, testNvtxEventValid)
{
  MATX_ENTER_HANDLER();
  
  // test fully qualified
  NvtxEvent myEventFull("testNvtxEventValid", "TestEvent", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 0 );
  // test auto-enumerated
  NvtxEvent myEventAuto("testNvtxEventValid", "TestEvent2" );
  
  MATX_EXIT_HANDLER();
}


//
// Test Emtpy NvtxEvent
//
TEST_F(NvtxIOTests, testNvtxEventEmpty)
{
  MATX_ENTER_HANDLER();
  
  // test unqualified
  NvtxEvent myEventFull(1);
  
  MATX_EXIT_HANDLER();
}


//
// Test NvtxEvent AutoCreate Function
//
TEST_F(NvtxIOTests, testNvtxEventAuto)
{
  MATX_ENTER_HANDLER();
  
  // test unqualified
  int myEvent = -1;
  myEvent = autoCreateNvtxEvent("AutoCreaeeFunction", "MyAutoRange");
  
  ASSERT_NE(myEvent, -1);
  
  MATX_EXIT_HANDLER();
}


//
// Test NvtxEvent Macros
//
TEST_F(NvtxIOTests, testNvtxMacros)
{
  MATX_ENTER_HANDLER();
  
  MATX_NVTX_SET_LOG_LEVEL( matx_nvxtLogLevels::MATX_NVTX_LOG_USER)
  
  MATX_NVTX_START("Range1")
  MATX_NVTX_START("Range2", matx_nvxtLogLevels::MATX_NVTX_LOG_USER)
  int range3Id = MATX_NVTX_START_RANGE("Range3" )
  int range4Id = MATX_NVTX_START_RANGE("Range4", matx_nvxtLogLevels::MATX_NVTX_LOG_USER )
  MATX_NVTX_START_RANGE("Range5", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 117 )
  
  MATX_NVTX_END_RANGE(range3Id)
  MATX_NVTX_END_RANGE(range4Id)
  MATX_NVTX_END_RANGE(117)

#ifdef MATX_NVTX_FLAGS  
  ASSERT_NE(range3Id, 0);
  ASSERT_NE(range4Id, 0);
#else
  ASSERT_EQ(range3Id, 0);
  ASSERT_EQ(range4Id, 0);  
#endif  
  
  MATX_EXIT_HANDLER();
}

