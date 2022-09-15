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
#include<string>
#include<utility>
#include <nvToolsExt.h>

namespace matx
{

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

const int32_t nunNvtxColors = 10;

const int32_t  colors[nunNvtxColors] = {NVTX_BLACK, NVTX_RED, NVTX_GREEN, NVTX_BLUE, NVTX_ORANGE, NVTX_PURPLE, NVTX_YELLOW, NVTX_TEAL, NVTX_PINK, NVTX_WHITE};

uint64_t curColorIdx;

std::map< int, nvtxRangeId_t> eventMap;

////////////  macros to ensure custom variable names for every call   ////////////
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a ## b

#define UNIQUE_NAME(base) CONCAT(base, __COUNTER__)
////////////////////////////////////////////////////////////////////////////////


////////////             Enable or Disable NVTX Macros          /////////////////
#define NVTX_FLAGS 1

#ifdef NVTX_FLAGS

  #define NVTX_1( message ) NvtxEvent UNIQUE_NAME(nvtxFlag_)( __FUNCTION__, message );
  #define NVTX_2( message, customId ) NvtxEvent UNIQUE_NAME(nvtxFlag_)( __FUNCTION__, message, customId );
  #define NVTX_3( message, customId, nvtxLevel ) NvtxEvent UNIQUE_NAME(nvtxFlag_)( __FUNCTION__, message, customId, nvtxLevel );

  #define NVTX_X(x,A,B,C,FUNC, ...)  FUNC

  // The macro that the programmer uses
  #define NVTX_START(...)   NVTX_X(,##__VA_ARGS__,\
                                  NVTX_3(__VA_ARGS__),\
                                  NVTX_2(__VA_ARGS__),\
                                  NVTX_1(__VA_ARGS__)\
                                  )

  #define NVTX_END( id ) endEvent( id );

#else

  #define NVTX_START( message )
  #define NVTX_START_SCOPED( message )

  #define NVTX_END( message )

#endif
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
///
///\brief Class wrapping NVTX management for automatic creation/deletion
///
////////////////////////////////////////////////////////////////////////////////
void registerEvent( size_t registerId, nvtxRangeId_t eventId )
{

  std::pair< int, nvtxRangeId_t > newPair( registerId, eventId );
  std::cout << "Registering Event: "  << registerId << std::endl;
  eventMap.insert(newPair);
}


////////////////////////////////////////////////////////////////////////////////
///
///\brief Class wrapping NVTX management for automatic creation/deletion
///
////////////////////////////////////////////////////////////////////////////////
void endEvent( size_t id )
{


  auto foundIter = eventMap.find( id );

  if( foundIter != eventMap.end())
  {
    nvtxRangeEnd(foundIter->second);
    eventMap.erase( foundIter );
  }
  else
  {
    std::cout << "!!! Warning, failed to end NVTX range: " << id << ", ranges may be incorrect !!!" << std::endl;
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
  ///\param message
  ///
  ////////////////////////////////////////////////////////////////////////////////
  NvtxEvent( std::string functionName, std::string message="", int registerId = -1, size_t nvtxLevel=0 )
  {
    size_t userLevel = 0;
    if( nvtxLevel < userLevel )
    {
      return;
    }

    int32_t curColor = colors[ curColorIdx % nunNvtxColors];
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
      ///\todo get the name of the calling function instead https://en.cppreference.com/w/cpp/utility/source_location
      eventAttrib.message.ascii =  functionName.c_str();
    }

    // save the id
    rangeId_ = nvtxRangeStartEx(&eventAttrib);

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
    nvtxRangeEnd(rangeId_);
  }


  nvtxRangeId_t  rangeId_;
};

} // end matx namespace