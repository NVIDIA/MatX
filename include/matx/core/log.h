////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#ifndef MATX_CORE_LOG_H
#define MATX_CORE_LOG_H

// Check for <format> header availability (C++20 feature not supported by all compilers). Since there 
// are so many compilers that support C++20 but not <format>, we want to make this an optional feature. 
// If the <format> header is not available, we will disable all logging.
#if __has_include(<format>)
#define MATX_HAS_STD_FORMAT 1
#else
#define MATX_HAS_STD_FORMAT 0
#endif

#if MATX_HAS_STD_FORMAT

#include <format>
#include <source_location>
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <mutex>

// Include MatX type traits and complex types for formatting support
#include "matx/core/half.h"
#include "matx/core/half_complex.h"
#include <complex>
#include <cuda/std/complex>

// Helper for formatting complex types
namespace matx {
namespace detail {
  // Generic helper to format any complex-like type with real() and imag() methods
  template<typename ComplexType>
  inline std::string format_complex(const ComplexType& c) {
    return std::format("({:g}{:+g}j)", 
                      static_cast<double>(c.real()), 
                      static_cast<double>(c.imag()));
  }
}
}

// Formatter specializations for all types supported by MatX
namespace std {
  // Formatter for std::complex<T>
  template<typename T>
  struct formatter<std::complex<T>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    
    template<typename FormatContext>
    auto format(const std::complex<T>& c, FormatContext& ctx) const {
      return format_to(ctx.out(), "{}", matx::detail::format_complex(c));
    }
  };
  
  // Formatter for cuda::std::complex<T>
  template<typename T>
  struct formatter<cuda::std::complex<T>> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    
    template<typename FormatContext>
    auto format(const cuda::std::complex<T>& c, FormatContext& ctx) const {
      return format_to(ctx.out(), "{}", matx::detail::format_complex(c));
    }
  };
  
  // Formatter for matxHalfComplex (fp16/bf16 complex) - moved to half_complex.h
  // Formatter for matxFp16 (half-precision float) - moved to half.h
  // Formatter for matxBf16 (bfloat16) - moved to half.h
  
  // Note: The formatters for matxHalfComplex, matxFp16, and matxBf16 are now defined
  // in their respective header files (half_complex.h and half.h) with proper guards.
}

namespace matx {
namespace detail {

/**
 * @brief Log severity levels
 */
enum class LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  FATAL = 5,
  OFF = 6
};

/**
 * @brief Convert log level to string
 */
constexpr std::string_view log_level_to_string(LogLevel level) {
  switch (level) {
    case LogLevel::TRACE: return "TRACE";
    case LogLevel::DEBUG: return "DEBUG";
    case LogLevel::INFO:  return "INFO";
    case LogLevel::WARN:  return "WARN";
    case LogLevel::ERROR: return "ERROR";
    case LogLevel::FATAL: return "FATAL";
    case LogLevel::OFF:   return "OFF";
    default:              return "UNKNOWN";
  }
}

/**
 * @brief Parse log level from string
 */
inline LogLevel parse_log_level(const char* level_str) {
  if (!level_str) return LogLevel::OFF;
  
  std::string level(level_str);
  if (level == "TRACE") return LogLevel::TRACE;
  if (level == "DEBUG") return LogLevel::DEBUG;
  if (level == "INFO")  return LogLevel::INFO;
  if (level == "WARN")  return LogLevel::WARN;
  if (level == "ERROR") return LogLevel::ERROR;
  if (level == "FATAL") return LogLevel::FATAL;
  if (level == "OFF")   return LogLevel::OFF;
  
  // Try numeric levels
  try {
    int num_level = std::stoi(level);
    if (num_level >= 0 && num_level <= 6) {
      return static_cast<LogLevel>(num_level);
    }
  } catch (...) {
    // Invalid level, return OFF
  }
  
  return LogLevel::OFF;
}

/**
 * @brief Format timestamp in ISO 8601 format
 */
inline std::string format_timestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;
  
  std::tm tm_buf;
  #ifdef _WIN32
    localtime_s(&tm_buf, &time_t_now);
  #else
    localtime_r(&time_t_now, &tm_buf);
  #endif
  
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &tm_buf);
  
  return std::format("{}.{:03d}", buffer, ms.count());
}

/**
 * @brief Logger class that manages log output
 */
class Logger {
private:
  LogLevel min_level_;
  std::unique_ptr<std::ostream> file_stream_;
  std::ostream* output_stream_;
  std::mutex mutex_;
  bool show_function_;
  
  Logger() : min_level_(LogLevel::ERROR), output_stream_(&std::cout), show_function_(false) {
    // Read log level from environment
    const char* level_env = std::getenv("MATX_LOG_LEVEL");
    if (level_env) {
      min_level_ = parse_log_level(level_env);
    }
    
    // Read whether to show function names
    const char* func_env = std::getenv("MATX_LOG_FUNC");
    if (func_env) {
      std::string func_str(func_env);
      show_function_ = (func_str == "1" || func_str == "true" || func_str == "TRUE" || func_str == "on" || func_str == "ON");
    }
    
    // Read log destination from environment
    const char* dest_env = std::getenv("MATX_LOG_DEST");
    if (dest_env) {
      std::string dest(dest_env);
      if (dest == "stdout") {
        output_stream_ = &std::cout;
      } else if (dest == "stderr") {
        output_stream_ = &std::cerr;
      } else {
        // Open file for logging
        file_stream_ = std::make_unique<std::ofstream>(dest, std::ios::app);
        if (file_stream_->good()) {
          output_stream_ = file_stream_.get();
        } else {
          // Fall back to stdout if file can't be opened
          std::cerr << "Failed to open log file: " << dest << ", falling back to stdout" << std::endl;
          output_stream_ = &std::cout;
        }
      }
    }
  }
  
public:
  // Singleton instance
  static Logger& instance() {
    static Logger logger;
    return logger;
  }
  
  // Delete copy/move constructors
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;
  Logger(Logger&&) = delete;
  Logger& operator=(Logger&&) = delete;
  
  /**
   * @brief Check if a log level is enabled
   */
  bool is_enabled(LogLevel level) const {
    return level >= min_level_ && min_level_ != LogLevel::OFF;
  }
  
  /**
   * @brief Log a message
   */
  template<typename... Args>
  void log(LogLevel level, 
           const std::source_location& location,
           std::format_string<Args...> fmt,
           Args&&... args) {
    if (!is_enabled(level)) {
      return;
    }
    
    // Format the user message
    std::string message = std::format(fmt, std::forward<Args>(args)...);
    
    // Extract just the filename from the full path
    std::string_view file_path = location.file_name();
    auto last_slash = file_path.find_last_of("/\\");
    std::string_view filename = (last_slash != std::string_view::npos) 
                                  ? file_path.substr(last_slash + 1) 
                                  : file_path;
    
    // Format the complete log message - conditionally include function name
    std::string log_line;
    if (show_function_) {
      log_line = std::format("{} [{}] {}:{} ({}) - {}\n",
                             format_timestamp(),
                             log_level_to_string(level),
                             filename,
                             location.line(),
                             location.function_name(),
                             message);
    } else {
      log_line = std::format("{} [{}] {}:{} - {}\n",
                             format_timestamp(),
                             log_level_to_string(level),
                             filename,
                             location.line(),
                             message);
    }
    
    // Thread-safe output
    std::lock_guard<std::mutex> lock(mutex_);
    *output_stream_ << log_line;
    output_stream_->flush();
  }
  
  /**
   * @brief Get the current minimum log level
   */
  LogLevel get_min_level() const {
    return min_level_;
  }
  
  /**
   * @brief Set the minimum log level (for testing or runtime changes)
   */
  void set_min_level(LogLevel level) {
    min_level_ = level;
  }
  
  /**
   * @brief Reinitialize the logger to pick up new environment settings
   * This is useful for testing when environment variables change
   */
  void reinitialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Re-read log level
    const char* level_env = std::getenv("MATX_LOG_LEVEL");
    if (level_env) {
      min_level_ = parse_log_level(level_env);
    } else {
      min_level_ = LogLevel::OFF;
    }
    
    // Re-read whether to show function names
    const char* func_env = std::getenv("MATX_LOG_FUNC");
    if (func_env) {
      std::string func_str(func_env);
      show_function_ = (func_str == "1" || func_str == "true" || func_str == "TRUE" || func_str == "on" || func_str == "ON");
    } else {
      show_function_ = false;
    }
    
    // Re-read log destination
    const char* dest_env = std::getenv("MATX_LOG_DEST");
    file_stream_.reset();  // Close any open file
    
    if (dest_env) {
      std::string dest(dest_env);
      if (dest == "stdout") {
        output_stream_ = &std::cout;
      } else if (dest == "stderr") {
        output_stream_ = &std::cerr;
      } else {
        // Open file for logging
        file_stream_ = std::make_unique<std::ofstream>(dest, std::ios::app);
        if (file_stream_->good()) {
          output_stream_ = file_stream_.get();
        } else {
          // Fall back to stdout if file can't be opened
          output_stream_ = &std::cout;
        }
      }
    } else {
      output_stream_ = &std::cout;
    }
  }
};

} // namespace detail
} // namespace matx

/**
 * @brief Main logging macro with minimal overhead when disabled
 * 
 * Usage: MATX_LOG(matx::detail::LogLevel::INFO, "Message: {}", value);
 */
#define MATX_LOG(level, ...) \
  do { \
    if (::matx::detail::Logger::instance().is_enabled(level)) { \
      ::matx::detail::Logger::instance().log(level, std::source_location::current(), __VA_ARGS__); \
    } \
  } while(0)

/**
 * @brief Convenience macros for each log level
 */
#define MATX_LOG_TRACE(...) MATX_LOG(::matx::detail::LogLevel::TRACE, __VA_ARGS__)
#define MATX_LOG_DEBUG(...) MATX_LOG(::matx::detail::LogLevel::DEBUG, __VA_ARGS__)
#define MATX_LOG_INFO(...)  MATX_LOG(::matx::detail::LogLevel::INFO, __VA_ARGS__)
#define MATX_LOG_WARN(...)  MATX_LOG(::matx::detail::LogLevel::WARN, __VA_ARGS__)
#define MATX_LOG_ERROR(...) MATX_LOG(::matx::detail::LogLevel::ERROR, __VA_ARGS__)
#define MATX_LOG_FATAL(...) MATX_LOG(::matx::detail::LogLevel::FATAL, __VA_ARGS__)

#else // !MATX_HAS_STD_FORMAT

// <format> header not available - disable all logging
#define MATX_LOG(level, ...) do {} while(0)
#define MATX_LOG_TRACE(...) do {} while(0)
#define MATX_LOG_DEBUG(...) do {} while(0)
#define MATX_LOG_INFO(...)  do {} while(0)
#define MATX_LOG_WARN(...)  do {} while(0)
#define MATX_LOG_ERROR(...) do {} while(0)
#define MATX_LOG_FATAL(...) do {} while(0)

#endif // MATX_HAS_STD_FORMAT

#endif // MATX_CORE_LOG_H
