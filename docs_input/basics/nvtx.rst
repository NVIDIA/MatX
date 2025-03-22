.. _nvtx-profiling:

NVTX Profiling
##############

Overview
--------
MatX provides an NVTX API to enable native compile-in profiling capabilities. The MatX NVTX API enable a user to 
easily profile all MatX calls using built-in NVTX ranges, while also providing a convenient API for the user to insert 
custom ranges in their own code. This API provides many convenience features such as:

- A convenient compile-in/compile-out MACRO based API 
- verbosity levels allowing varying levels of profiling detail
- Built-in color rotation
- Automatic scope management and range naming 
- Overloaded API for manual range specification

The MatX NVTX API is implemented as a set of C++ Macros, allowing the user to compile all calls out of the project for 
maximum performance when profiling is not needed. 

Enabling NVTX API and Setting Log Level
---------------------------------------
To enable the NVTX Profiling API, simple compile with the ``MATX_NVTX_FLAGS=ON`` enabled in the cmake command.
Once the flags are enabled at compile time, the project defaults to logging at the API level, which will provide NVTX
ranges for all MatX API calls. If another logging level is desired, this can be changed using the ``matx::setNVTXLogLevel()`` call. 
Possible log levels are defined in ``matx_nvxtLogLevels``.

Using the NVTX API
------------------
The MatX NVTX API consists of two modes: auto managed, and manual range. The auto-managed API will automatically match the NVTX range to 
the scope in which it is declared, establishing the NVTX range from the call’s instantiation to the end of its parent scope. Only a single 
call is needed, with optional inputs defined below. If no message is provided, the call defaults to using the calling function’s name as 
the NVTX range’s message.

The Manual Range NVTX API requires the user to make a call to the NVTX API at both the beginning and end of the desired range. The Manual 
Range API uses a user defined handle (int) to reference the NVTX range. A Manual NVTX Range can be fully qualified, or the user can allow the API to auto-enumerate the range. 
If the user chooses to allow the API to define the handle, an int is returned from the call for the user.

.. note::
  If a user chooses to use a manual range, they must manually call ``MATX_NVTX_END_RANGE(RANGE_HANDLE)``, or the range will not terminate until the end of execution.

NVTX Examples
-------------

.. list-table::
  :widths: 60 40
  :header-rows: 1
  
  * - Command 
    - Result
  * - MATX_NVTX_START("")
    - NVTX range scoped to this function, named the same as function with log level of User 
  * - MATX_NVTX_START("MY_MESSAGE")
    - NVTX range scoped to this function, named “MY_MESSAGE” with log level of User
  * - MATX_NVTX_START("MY_MESSAGE", matx::MATX_NVTX_LOG_API )
    - NVTX range scoped to this function, named “MY_MESSAGE” with log level of API
  * - MATX_NVTX_START_RANGE( "MY_MESSAGE" )
    - NVTX range with manual scope, named “MY_MESSAGE”, log level of User, and an auto-enumerated handle
  * - MATX_NVTX_START_RANGE( "MY_MESSAGE", matx_nvxtLogLevels::MATX_NVTX_LOG_INTERNAL )
    - NVTX range with manual scope, named “MY_MESSAGE”, log level of Internal, and an auto-enumerated handle    
  * - MATX_NVTX_START_RANGE( "MY_MESSAGE", matx_nvxtLogLevels::MATX_NVTX_LOG_USER, 1 )
    - NVTX range with manual scope, named “MY_MESSAGE”, log level of USER, and handle ID of 1
  * - MATX_NVTX_END_RANGE(1)
    - Ends the NVTX range of range with a handle of 1 used in NVTX_START_RANGE        
    
Code examples are provided in the ``simple_radar_pipeline`` code to show user utilization of the MatX NVTX API. 

MatX NVTX API 
-------------
.. doxygenfunction:: matx::setNVTXLogLevel
.. doxygenfunction:: matx::registerEvent
.. doxygenfunction:: matx::endEvent

MatX NVTX Logging Levels
------------------------
.. doxygenenum:: matx::matx_nvxtLogLevels

MatX NVTX Auto Range Colors
---------------------------
.. doxygenvariable:: matx::nvtxColors    
