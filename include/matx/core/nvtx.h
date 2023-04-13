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
#pragma once
#include<functional>
#include<map>
#include <mutex>
#include<string>
#include<utility>
#include <nvToolsExt.h>

namespace matx
{

/**
 * @brief levels of NVTX Logging. Lower level is more selective (prints less)
 *
 */
enum matx_nvxtLogLevels
{
  MATX_NVTX_LOG_NONE = 0,
  MATX_NVTX_LOG_USER = 1,
  MATX_NVTX_LOG_API = 2,
  MATX_NVTX_LOG_INTERNAL = 3,
  MATX_NVTX_LOG_ALL = 0xFFFFFF
};


const int32_t NVTX_BLACK  = 0x000000;
const int32_t NVTX_RED    = 0xFF0000;
const int32_t NVTX_GREEN  = 0x00FF00;
const int32_t NVTX_BLUE   = 0x0000FF;
const int32_t NVTX_ORANGE = 0xFFA500;
const int32_t NVTX_PURPLE = 0x800080;
const int32_t NVTX_YELLOW = 0xFFFF00;
const int32_t NVTX_TEAL   = 0x008080;
const int32_t NVTX_PINK   = 0xFFC0CB;
const int32_t NVTX_WHITE  = 0xFFFFFF;

static const int32_t nunNvtxColors = 10;

/**
 * @brief automatic NVTX Colors
 *
 */
static const int32_t nvtxColors[nunNvtxColors] = {
                                             NVTX_BLACK,
                                             NVTX_RED,
                                             NVTX_GREEN,
                                             NVTX_BLUE,
                                             NVTX_ORANGE,
                                             NVTX_PURPLE,
                                             NVTX_YELLOW,
                                             NVTX_TEAL,
                                             NVTX_PINK,
                                             NVTX_WHITE
                                             };

inline uint64_t                      curColorIdx;     ///< counter for rotation of colors for sequential ranges
inline std::map< int, nvtxRangeId_t> nvtx_eventMap;   ///< map of currently active NVTX ranges
inline std::mutex                    nvtx_memory_mtx; ///< Mutex protecting updates from map

inline matx_nvxtLogLevels globalNvtxLevel = matx_nvxtLogLevels::MATX_NVTX_LOG_API;

//////  macros to ensure custom variable names for every call  ////////
#define MATX_CONCAT(a, b) MATX_CONCAT_INNER(a, b)
#define MATX_CONCAT_INNER(a, b) a ## b

#define MATX_UNIQUE_NAME(base) MATX_CONCAT(base, __COUNTER__)
///////////////////////////////////////////////////////////////////////


////////////             Enable NVTX Macros          /////////////////

#ifdef MATX_NVTX_FLAGS

  ///\todo update to use C++20 runtime fucntion for actual call location
  /// https://en.cppreference.com/w/cpp/utility/source_location
  #define MATX_NVTX_1( message ) matx::NvtxEvent MATX_UNIQUE_NAME(nvtxFlag_)( __FUNCTION__, message );
  #define MATX_NVTX_2( message, nvtxLevel ) matx::NvtxEvent MATX_UNIQUE_NAME(nvtxFlag_)( __FUNCTION__, message, nvtxLevel );

  #define MATX_NVTX_X(x,A,B,FUNC, ...)  FUNC

  // The macro that the programmer uses
  #define MATX_NVTX_START(...)  MATX_NVTX_X(,##__VA_ARGS__,\
                                MATX_NVTX_2(__VA_ARGS__),\
                                MATX_NVTX_1(__VA_ARGS__)\
                                )

  #define MATX_NVTX_START_RANGE( message, nvtxLevel, id ) matx::NvtxEvent MATX_UNIQUE_NAME(nvtxFlag_)( __FUNCTION__, message, nvtxLevel, id );

  #define MATX_NVTX_END_RANGE( id ) matx::endEvent( id );

  #define MATX_NVTX_SET_LOG_LEVEL( nvtxLevel ) matx::setNVTXLogLevel( nvtxLevel );

////////////             Disable NVTX Macros          /////////////////
#else

  #define MATX_NVTX_1( message );
  #define MATX_NVTX_2( message, customId );
  #define MATX_NVTX_START_RANGE( message, nvtxLevel, id );

  #define MATX_NVTX_X(x,A,B,FUNC, ...)  FUNC

  // The macro that the programmer uses
  #define MATX_NVTX_START(...)    MATX_NVTX_X(,##__VA_ARGS__,\
                                  MATX_NVTX_2(__VA_ARGS__),\
                                  MATX_NVTX_1(__VA_ARGS__)\
                                  )

  #define MATX_NVTX_END_RANGE( id );

  #define MATX_NVTX_SET_LOG_LEVEL( nvtxLevel );

#endif
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
///
///\brief Utility Function to set Global Log Level. should be called through the
///       MATX_NVTX_SET_LOG_LEVEL macro with the same parameters
///
////////////////////////////////////////////////////////////////////////////////
[[maybe_unused]] static void setNVTXLogLevel( matx_nvxtLogLevels newNVTXLevel)
{
  globalNvtxLevel = newNVTXLevel;
}

////////////////////////////////////////////////////////////////////////////////
///
///\brief fucntion wrapping NVTX management for automatic creation/deletion
///       MATX_NVTX_START or MATX_NVTX_START_RANGE macro with the same parameters
///
////////////////////////////////////////////////////////////////////////////////
[[maybe_unused]] static void registerEvent( int registerId, nvtxRangeId_t eventId )
{
  std::unique_lock lck(nvtx_memory_mtx);

  std::pair< int, nvtxRangeId_t > newPair( registerId, eventId );
  nvtx_eventMap.insert(newPair);
}


////////////////////////////////////////////////////////////////////////////////
///
///\brief fucntion wrapping NVTX management for automatic creation/deletion
///       MATX_NVTX_END_RANGE macro with the same parameters
///
////////////////////////////////////////////////////////////////////////////////
static void endEvent( int id )
{
  std::unique_lock lck(nvtx_memory_mtx);

  auto foundIter = nvtx_eventMap.find( id );

  if( foundIter != nvtx_eventMap.end())
  {
    nvtxRangeEnd(foundIter->second);
    nvtx_eventMap.erase( foundIter );
  }
}

////////////////////////////////////////////////////////////////////////////////
///
///\brief Class wrapping NVTX management for automatic creation/deletion
///
////////////////////////////////////////////////////////////////////////////////
class NvtxEvent
{
  public:

  ////////////////////////////////////////////////////////////////////////////////
  ///
  ///\brief ctor
  ///
  ///\param functionName name of calling function, only used if no message is provided;
  ///                    is auto-populated from macro
  ///\param message      custom message/name for range defaults to ""
  ///                    which uses function name instead
  ///\param nvtxLevel    level of NVTX events to use higher number reduces scope
  ///\param registerId   customID (integer) used to reference ranges you wish to manually end

  ///
  ////////////////////////////////////////////////////////////////////////////////
  NvtxEvent( std::string functionName, std::string message="",  matx_nvxtLogLevels nvtxLevel = matx_nvxtLogLevels::MATX_NVTX_LOG_INTERNAL, int registerId = -1 )
  {

    if( nvtxLevel > globalNvtxLevel )
    {
      userHandle_ = -1;
      return;
    }

    int32_t curColor = nvtxColors[ curColorIdx % nunNvtxColors];
    curColorIdx++;

    nvtxEventAttributes_t eventAttrib;

    // default event info
    eventAttrib.version     = NVTX_VERSION;
    eventAttrib.size        = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType   = NVTX_COLOR_ARGB;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;

    // set custom color
    eventAttrib.color = curColor;

    // set message, if no message provided use the calling funciton as name
    if( message != "" )
    {
      eventAttrib.message.ascii = message.c_str();
    }
    else
    {
      eventAttrib.message.ascii =  functionName.c_str();
    }

    // save the id
    rangeId_ = nvtxRangeStartEx(&eventAttrib);
    userHandle_ = registerId;
    // if register with global map
    if( registerId >= 0 )
    {
      registerEvent( registerId, rangeId_ );
    }

  }


  ////////////////////////////////////////////////////////////////////////////////
  ///
  ///\brief dtor
  ///
  ////////////////////////////////////////////////////////////////////////////////
  ~NvtxEvent( )
  {
    if(userHandle_ != -1)
    {
      endEvent( userHandle_ );
    }

    nvtxRangeEnd(rangeId_);
  }


  nvtxRangeId_t  rangeId_;
  int            userHandle_;
};

} // end matx namespace
