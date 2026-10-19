execute_process(COMMAND "${READELF}" -d "${LIBRARY}"
  RESULT_VARIABLE result OUTPUT_VARIABLE dependencies ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect library dependencies: ${error}")
endif()
if(dependencies MATCHES "Shared library: \\[libcuda\\.so")
  message(FATAL_ERROR "MatX consumer has a direct dependency on libcuda: ${dependencies}")
endif()
