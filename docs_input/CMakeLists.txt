set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

# Find all the public headers
file(GLOB_RECURSE MATX_PUBLIC_HEADERS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../include/matx/*.h)

# Find all the rst documentation
file(GLOB_RECURSE MATX_RST_DOCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.rst)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYGEN_EXAMPLE_DIR ${PROJECT_SOURCE_DIR}/examples)
set(DOXYGEN_EXCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/*)
set(DOXYGEN_EXCLUDE_DIR2 ${PROJECT_BINARY_DIR}/*)
set(DOXYGEN_EXCLUDE_DIR3 ${PROJECT_SOURCE_DIR}/build*)
set(DOXYGEN_EXCLUDE_DIR4 ${PROJECT_SOURCE_DIR}/examples/cmake_sample_project/build*)
set(DOXYXML_DIR ${PROJECT_BINARY_DIR}/doxygen/xml/)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(SPHINX_CFG_IN ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in)
set(SPHINX_CFG_OUT ${CMAKE_CURRENT_BINARY_DIR}/conf.py)

# Cache used by Sphinx
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_static)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/_templates)
configure_file(${PROJECT_SOURCE_DIR}/docs_input/_templates/layout.html ${CMAKE_CURRENT_BINARY_DIR}/_templates/ COPYONLY)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)
configure_file(${SPHINX_CFG_IN} ${SPHINX_CFG_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html)  # Doxygen won't create this for us
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}/xml)   # Doxygen won't create this for us
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}/latex) # Doxygen won't create this for us


add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${MATX_PUBLIC_HEADERS}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating documentation with Doxygen")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND
                     ${SPHINX_EXECUTABLE} -b html -W
                     -c ${CMAKE_CURRENT_BINARY_DIR}
                     # Tell Breathe where to find the Doxygen output
                     -Dbreathe_projects.MatX=${DOXYGEN_OUTPUT_DIR}/xml
                   ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
                    # Other docs files you want to track should go here (or in some variable)
                    ${MATX_RST_DOCS}
                    ${DOXYGEN_INDEX_FILE}
                   MAIN_DEPENDENCY ${SPHINX_CFG_OUT}
                   COMMENT "Generating documentation with Sphinx")

# Nice named target so we can run the job easily
add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})
