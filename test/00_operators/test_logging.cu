////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
// All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "matx.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace matx;

class LoggingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Save original environment
    original_level_ = std::getenv("MATX_LOG_LEVEL");
    original_dest_ = std::getenv("MATX_LOG_DEST");
  }

  void TearDown() override {
    // Restore original environment
    if (original_level_) {
      setenv("MATX_LOG_LEVEL", original_level_, 1);
    } else {
      unsetenv("MATX_LOG_LEVEL");
    }
    
    if (original_dest_) {
      setenv("MATX_LOG_DEST", original_dest_, 1);
    } else {
      unsetenv("MATX_LOG_DEST");
    }
    
    // Reset logger to pick up restored environment
    detail::Logger::instance().set_min_level(detail::LogLevel::OFF);
  }

  const char* original_level_;
  const char* original_dest_;
};

TEST_F(LoggingTest, LogLevelParsing) {
  EXPECT_EQ(detail::parse_log_level("TRACE"), detail::LogLevel::TRACE);
  EXPECT_EQ(detail::parse_log_level("DEBUG"), detail::LogLevel::DEBUG);
  EXPECT_EQ(detail::parse_log_level("INFO"), detail::LogLevel::INFO);
  EXPECT_EQ(detail::parse_log_level("WARN"), detail::LogLevel::WARN);
  EXPECT_EQ(detail::parse_log_level("ERROR"), detail::LogLevel::ERROR);
  EXPECT_EQ(detail::parse_log_level("FATAL"), detail::LogLevel::FATAL);
  EXPECT_EQ(detail::parse_log_level("OFF"), detail::LogLevel::OFF);
  EXPECT_EQ(detail::parse_log_level("INVALID"), detail::LogLevel::OFF);
  EXPECT_EQ(detail::parse_log_level(nullptr), detail::LogLevel::OFF);
}

TEST_F(LoggingTest, LogLevelToString) {
  EXPECT_EQ(detail::log_level_to_string(detail::LogLevel::TRACE), "TRACE");
  EXPECT_EQ(detail::log_level_to_string(detail::LogLevel::DEBUG), "DEBUG");
  EXPECT_EQ(detail::log_level_to_string(detail::LogLevel::INFO), "INFO");
  EXPECT_EQ(detail::log_level_to_string(detail::LogLevel::WARN), "WARN");
  EXPECT_EQ(detail::log_level_to_string(detail::LogLevel::ERROR), "ERROR");
  EXPECT_EQ(detail::log_level_to_string(detail::LogLevel::FATAL), "FATAL");
  EXPECT_EQ(detail::log_level_to_string(detail::LogLevel::OFF), "OFF");
}

TEST_F(LoggingTest, DefaultLoggingDisabled) {
  unsetenv("MATX_LOG_LEVEL");
  detail::Logger::instance().set_min_level(detail::LogLevel::OFF);
  
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::TRACE));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::DEBUG));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::INFO));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::WARN));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::ERROR));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::FATAL));
}

TEST_F(LoggingTest, LogLevelHierarchy) {
  detail::Logger::instance().set_min_level(detail::LogLevel::INFO);
  
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::TRACE));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::DEBUG));
  EXPECT_TRUE(detail::Logger::instance().is_enabled(detail::LogLevel::INFO));
  EXPECT_TRUE(detail::Logger::instance().is_enabled(detail::LogLevel::WARN));
  EXPECT_TRUE(detail::Logger::instance().is_enabled(detail::LogLevel::ERROR));
  EXPECT_TRUE(detail::Logger::instance().is_enabled(detail::LogLevel::FATAL));
}

TEST_F(LoggingTest, BasicLogging) {
  detail::Logger::instance().set_min_level(detail::LogLevel::DEBUG);
  
  // These should not crash
  MATX_LOG_DEBUG("Test message");
  MATX_LOG_INFO("Test message with value: {}", 42);
  MATX_LOG_WARN("Test warning");
  MATX_LOG_ERROR("Test error");
}

TEST_F(LoggingTest, LoggingWithDifferentTypes) {
  detail::Logger::instance().set_min_level(detail::LogLevel::DEBUG);
  
  // Test various types
  MATX_LOG_DEBUG("Integer: {}", 42);
  MATX_LOG_DEBUG("Float: {:.2f}", 3.14159);
  MATX_LOG_DEBUG("String: {}", "hello");
  MATX_LOG_DEBUG("Multiple: {} {} {}", 1, 2.5, "test");
  MATX_LOG_DEBUG("Hex: {:x}", 255);
}

TEST_F(LoggingTest, DisabledLoggingNoOverhead) {
  detail::Logger::instance().set_min_level(detail::LogLevel::OFF);
  
  // This should have minimal overhead (just a boolean check)
  for (int i = 0; i < 1000; i++) {
    MATX_LOG_TRACE("Iteration {}", i);
  }
  
  // No assertions needed - this test verifies it compiles and runs
  SUCCEED();
}

TEST_F(LoggingTest, ConvenienceMacros) {
  detail::Logger::instance().set_min_level(detail::LogLevel::TRACE);
  
  // Test all convenience macros
  MATX_LOG_TRACE("Trace message");
  MATX_LOG_DEBUG("Debug message");
  MATX_LOG_INFO("Info message");
  MATX_LOG_WARN("Warn message");
  MATX_LOG_ERROR("Error message");
  MATX_LOG_FATAL("Fatal message");
  
  SUCCEED();
}

TEST_F(LoggingTest, TimestampFormat) {
  std::string timestamp = detail::format_timestamp();
  
  // Verify it has the expected format (ISO 8601 with milliseconds)
  // Format: YYYY-MM-DDTHH:MM:SS.mmm
  EXPECT_GE(timestamp.length(), 23u);
  EXPECT_NE(timestamp.find('T'), std::string::npos);
  EXPECT_NE(timestamp.find('.'), std::string::npos);
}

TEST_F(LoggingTest, SourceLocation) {
  // Verify that source location captures function name
  // This test just ensures the logging with source location compiles and runs
  detail::Logger::instance().set_min_level(detail::LogLevel::DEBUG);
  
  MATX_LOG_DEBUG("Testing source location capture");
  
  // The log message should include this function name "TestBody" or similar
  // We can't easily verify the output without capturing stdout, but we can
  // verify it doesn't crash
  SUCCEED();
}

// Note: File output tests are harder to implement in a unit test
// because they depend on file system access and permissions.
// Manual testing is recommended for file output functionality.

