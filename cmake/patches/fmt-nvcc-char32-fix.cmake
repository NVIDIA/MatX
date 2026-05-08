# Patches include/fmt/format.h to replace char32_t U"..." string literals
# whose encoded values exceed U+10FFFF with a constexpr uint32_t array.
#
# Background: nvcc's CUDA frontend rejects char32_t code unit values outside
# [0, U+10FFFF] with "character value is out of range". The function
# fractional_part_rounding_thresholds() in fmt 11.1.4 through 12.1.0 returns
# values such as 0x9999999a by indexing into a U"..." string literal. These
# code units exceed U+10FFFF so nvcc rejects the file at parse time. gcc/clang
# accept arbitrary char32_t bit patterns and are unaffected.
#
# Replacement: a constexpr uint32_t array that is semantically identical,
# valid from C++14 onwards, and accepted by nvcc regardless of value.
#
# This script is idempotent: it is a no-op if the patch has already been
# applied or if the source does not contain the problematic literal.
#
# Invoked by CPMAddPackage as a PATCH_COMMAND:
#   PATCH_COMMAND ${CMAKE_COMMAND}
#       -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/patches/fmt-nvcc-char32-fix.cmake"
# ExternalProject/FetchContent runs PATCH_COMMAND with CWD set to the fmt
# source root, so the relative path "include/fmt/format.h" resolves correctly.

cmake_minimum_required(VERSION 3.12)

set(fmt_format_h "include/fmt/format.h")
if(NOT EXISTS "${fmt_format_h}")
  message(FATAL_ERROR "fmt-nvcc-char32-fix: ${fmt_format_h} not found. "
    "This script must be run from the fmt source root directory.")
endif()

file(READ "${fmt_format_h}" src)

# Quick check: bail early if the problematic char32_t literal is not present.
string(FIND "${src}" "return U\"\\x9999999a" needs_patch)
if(needs_patch EQUAL -1)
  message(STATUS "fmt-nvcc-char32-fix: not needed (already patched or different version)")
  return()
endif()

# Normalize CRLF -> LF so string(REPLACE) works regardless of git line-ending
# settings on the host.
string(REPLACE "\r\n" "\n" src "${src}")

# Replace the two-line char32_t return with a constexpr uint32_t array.
#
# Original (fmt 11.1.4 through 12.1.0):
#   return U"\x9999999a\x828f5c29\x80418938\x80068db9\x8000a7c6\x800010c7"
#          U"\x800001ae\x8000002b"[index];
#
# Replacement (semantically identical, nvcc-compatible):
#   constexpr uint32_t thresholds[] = {
#     0x9999999au, 0x828f5c29u, 0x80418938u, 0x80068db9u,
#     0x8000a7c6u, 0x800010c7u, 0x800001aeu, 0x8000002bu
#   };
#   return thresholds[index];
string(REPLACE
  "  return U\"\\x9999999a\\x828f5c29\\x80418938\\x80068db9\\x8000a7c6\\x800010c7\"\n         U\"\\x800001ae\\x8000002b\"[index];"
  "  constexpr uint32_t thresholds[] = {\n    0x9999999au, 0x828f5c29u, 0x80418938u, 0x80068db9u,\n    0x8000a7c6u, 0x800010c7u, 0x800001aeu, 0x8000002bu\n  };\n  return thresholds[index];"
  src "${src}")

# Verify the patch was applied.
string(FIND "${src}" "return U\"\\x9999999a" still_present)
if(NOT still_present EQUAL -1)
  message(FATAL_ERROR
    "fmt-nvcc-char32-fix: string replacement failed in ${fmt_format_h}. "
    "The fmt source layout may have changed from the expected fmt 11.x format "
    "(check indentation and continuation-line spacing). "
    "Update cmake/patches/fmt-nvcc-char32-fix.cmake or file a MatX issue.")
endif()

file(WRITE "${fmt_format_h}" "${src}")
message(STATUS "fmt-nvcc-char32-fix: successfully patched ${fmt_format_h}")
