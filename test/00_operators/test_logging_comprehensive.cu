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
#include <regex>
#include <thread>
#include <chrono>

using namespace matx;

class LoggingComprehensiveTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Save original environment
    original_level_ = std::getenv("MATX_LOG_LEVEL");
    original_dest_ = std::getenv("MATX_LOG_DEST");
    
    // Generate a unique test file name
    test_log_file_ = std::string("/tmp/matx_test_log_") + 
                     std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + 
                     ".log";
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
    detail::Logger::instance().reinitialize();
    
    // Clean up test log file
    if (std::filesystem::exists(test_log_file_)) {
      std::filesystem::remove(test_log_file_);
    }
  }

  std::string ReadFileContents(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
      return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
  }

  bool ContainsPattern(const std::string& text, const std::string& pattern) {
    return text.find(pattern) != std::string::npos;
  }

  bool MatchesRegex(const std::string& text, const std::string& pattern) {
    std::regex regex_pattern(pattern);
    return std::regex_search(text, regex_pattern);
  }

  const char* original_level_;
  const char* original_dest_;
  std::string test_log_file_;
};

// Test basic file output
TEST_F(LoggingComprehensiveTest, FileOutput) {
  // Configure to log to file
  setenv("MATX_LOG_LEVEL", "DEBUG", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  
  // Reinitialize logger to pick up environment settings
  detail::Logger::instance().reinitialize();
  
  // Write some log messages
  MATX_LOG_DEBUG("Test message 1");
  MATX_LOG_INFO("Test message 2");
  MATX_LOG_WARN("Test message 3");
  MATX_LOG_ERROR("Test message 4");
  
  // Give a moment for writes to flush
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  // Read the log file
  std::string contents = ReadFileContents(test_log_file_);
  
  // Verify all messages are present
  EXPECT_TRUE(ContainsPattern(contents, "Test message 1"));
  EXPECT_TRUE(ContainsPattern(contents, "Test message 2"));
  EXPECT_TRUE(ContainsPattern(contents, "Test message 3"));
  EXPECT_TRUE(ContainsPattern(contents, "Test message 4"));
  
  // Verify severity levels are present
  EXPECT_TRUE(ContainsPattern(contents, "[DEBUG]"));
  EXPECT_TRUE(ContainsPattern(contents, "[INFO]"));
  EXPECT_TRUE(ContainsPattern(contents, "[WARN]"));
  EXPECT_TRUE(ContainsPattern(contents, "[ERROR]"));
}

// Test log format with timestamps
TEST_F(LoggingComprehensiveTest, LogFormat) {
  setenv("MATX_LOG_LEVEL", "INFO", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  setenv("MATX_LOG_FUNC", "1", 1);
  detail::Logger::instance().reinitialize();
  
  MATX_LOG_INFO("Format test message");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents = ReadFileContents(test_log_file_);
  
  // Verify ISO 8601 timestamp format: YYYY-MM-DDTHH:MM:SS.mmm
  EXPECT_TRUE(MatchesRegex(contents, R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})"));
  
  // Verify severity level format: [LEVEL]
  EXPECT_TRUE(MatchesRegex(contents, R"(\[INFO\])"));
  
  // Verify file name is present (test_logging_comprehensive.cu)
  EXPECT_TRUE(ContainsPattern(contents, "test_logging_comprehensive.cu"));
  
  // Verify line number is present (format: filename:line)
  EXPECT_TRUE(MatchesRegex(contents, R"(test_logging_comprehensive\.cu:\d+)"));
  
  // Verify function name is present (in parentheses, contains TestBody)
  EXPECT_TRUE(MatchesRegex(contents, R"(\(.*TestBody.*\))"));
  
  // Verify message is present
  EXPECT_TRUE(ContainsPattern(contents, "Format test message"));
}

// Test log level filtering
TEST_F(LoggingComprehensiveTest, LogLevelFiltering) {
  setenv("MATX_LOG_LEVEL", "WARN", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  detail::Logger::instance().reinitialize();
  
  MATX_LOG_TRACE("TRACE level message");
  MATX_LOG_DEBUG("DEBUG level message");
  MATX_LOG_INFO("INFO level message");
  MATX_LOG_WARN("WARN level message");
  MATX_LOG_ERROR("ERROR level message");
  
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents = ReadFileContents(test_log_file_);
  
  // Verify lower levels don't appear
  EXPECT_FALSE(ContainsPattern(contents, "TRACE level message"));
  EXPECT_FALSE(ContainsPattern(contents, "DEBUG level message"));
  EXPECT_FALSE(ContainsPattern(contents, "INFO level message"));
  
  // Verify WARN and above appear
  EXPECT_TRUE(ContainsPattern(contents, "WARN level message"));
  EXPECT_TRUE(ContainsPattern(contents, "ERROR level message"));
  EXPECT_TRUE(ContainsPattern(contents, "[WARN]"));
  EXPECT_TRUE(ContainsPattern(contents, "[ERROR]"));
}

// Test format string with variables
TEST_F(LoggingComprehensiveTest, FormatStrings) {
  setenv("MATX_LOG_LEVEL", "DEBUG", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  detail::Logger::instance().reinitialize();
  
  int int_val = 42;
  double double_val = 3.14159;
  std::string str_val = "hello";
  
  MATX_LOG_DEBUG("Integer: {}", int_val);
  MATX_LOG_DEBUG("Double: {:.2f}", double_val);
  MATX_LOG_DEBUG("String: {}", str_val);
  MATX_LOG_DEBUG("Multiple: {} {} {}", int_val, double_val, str_val);
  MATX_LOG_DEBUG("Hex: 0x{:08x}", 0xDEADBEEF);
  
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents = ReadFileContents(test_log_file_);
  
  EXPECT_TRUE(ContainsPattern(contents, "Integer: 42"));
  EXPECT_TRUE(ContainsPattern(contents, "Double: 3.14"));
  EXPECT_TRUE(ContainsPattern(contents, "String: hello"));
  EXPECT_TRUE(ContainsPattern(contents, "42 3.14159 hello"));
  EXPECT_TRUE(ContainsPattern(contents, "0xdeadbeef"));
}

// Test multiple log messages
TEST_F(LoggingComprehensiveTest, MultipleMessages) {
  setenv("MATX_LOG_LEVEL", "DEBUG", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  detail::Logger::instance().reinitialize();
  
  for (int i = 0; i < 10; i++) {
    MATX_LOG_DEBUG("Message number {}", i);
  }
  
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents = ReadFileContents(test_log_file_);
  
  // Verify all messages are present
  for (int i = 0; i < 10; i++) {
    std::string expected = std::string("Message number ") + std::to_string(i);
    EXPECT_TRUE(ContainsPattern(contents, expected));
  }
  
  // Count the number of log lines (each message should be on its own line)
  auto line_count = std::count(contents.begin(), contents.end(), '\n');
  EXPECT_EQ(line_count, 10);
}

// Test stderr output
TEST_F(LoggingComprehensiveTest, StderrOutput) {
  // Note: Testing stderr output is tricky without redirecting stderr
  // This test just verifies that setting stderr doesn't crash
  setenv("MATX_LOG_LEVEL", "ERROR", 1);
  setenv("MATX_LOG_DEST", "stderr", 1);
  detail::Logger::instance().reinitialize();
  
  // This should not crash
  MATX_LOG_ERROR("Test stderr message");
  
  SUCCEED();
}

// Test stdout output
TEST_F(LoggingComprehensiveTest, StdoutOutput) {
  // Note: Testing stdout output is tricky without redirecting stdout
  // This test just verifies that setting stdout doesn't crash
  setenv("MATX_LOG_LEVEL", "INFO", 1);
  setenv("MATX_LOG_DEST", "stdout", 1);
  detail::Logger::instance().reinitialize();
  
  // This should not crash
  MATX_LOG_INFO("Test stdout message");
  
  SUCCEED();
}

// Test all convenience macros
TEST_F(LoggingComprehensiveTest, ConvenienceMacros) {
  setenv("MATX_LOG_LEVEL", "TRACE", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  detail::Logger::instance().reinitialize();
  
  MATX_LOG_TRACE("Trace message");
  MATX_LOG_DEBUG("Debug message");
  MATX_LOG_INFO("Info message");
  MATX_LOG_WARN("Warn message");
  MATX_LOG_ERROR("Error message");
  MATX_LOG_FATAL("Fatal message");
  
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents = ReadFileContents(test_log_file_);
  
  EXPECT_TRUE(ContainsPattern(contents, "[TRACE]"));
  EXPECT_TRUE(ContainsPattern(contents, "Trace message"));
  EXPECT_TRUE(ContainsPattern(contents, "[DEBUG]"));
  EXPECT_TRUE(ContainsPattern(contents, "Debug message"));
  EXPECT_TRUE(ContainsPattern(contents, "[INFO]"));
  EXPECT_TRUE(ContainsPattern(contents, "Info message"));
  EXPECT_TRUE(ContainsPattern(contents, "[WARN]"));
  EXPECT_TRUE(ContainsPattern(contents, "Warn message"));
  EXPECT_TRUE(ContainsPattern(contents, "[ERROR]"));
  EXPECT_TRUE(ContainsPattern(contents, "Error message"));
  EXPECT_TRUE(ContainsPattern(contents, "[FATAL]"));
  EXPECT_TRUE(ContainsPattern(contents, "Fatal message"));
}

// Test numeric log levels
TEST_F(LoggingComprehensiveTest, NumericLogLevels) {
  // Test numeric level 0 (TRACE)
  setenv("MATX_LOG_LEVEL", "0", 1);
  detail::Logger::instance().set_min_level(detail::parse_log_level("0"));
  EXPECT_TRUE(detail::Logger::instance().is_enabled(detail::LogLevel::TRACE));
  
  // Test numeric level 2 (INFO)
  setenv("MATX_LOG_LEVEL", "2", 1);
  detail::Logger::instance().set_min_level(detail::parse_log_level("2"));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::DEBUG));
  EXPECT_TRUE(detail::Logger::instance().is_enabled(detail::LogLevel::INFO));
  
  // Test numeric level 6 (OFF)
  setenv("MATX_LOG_LEVEL", "6", 1);
  detail::Logger::instance().set_min_level(detail::parse_log_level("6"));
  EXPECT_FALSE(detail::Logger::instance().is_enabled(detail::LogLevel::ERROR));
}

// Test large log messages
TEST_F(LoggingComprehensiveTest, LargeMessages) {
  setenv("MATX_LOG_LEVEL", "INFO", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  detail::Logger::instance().reinitialize();
  
  // Create a large message
  std::string large_message(1000, 'A');
  MATX_LOG_INFO("Large message: {}", large_message);
  
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents = ReadFileContents(test_log_file_);
  EXPECT_TRUE(ContainsPattern(contents, large_message));
}

// Test special characters in messages
TEST_F(LoggingComprehensiveTest, SpecialCharacters) {
  setenv("MATX_LOG_LEVEL", "DEBUG", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  detail::Logger::instance().reinitialize();
  
  MATX_LOG_DEBUG("Message with newline\\n and tab\\t");
  MATX_LOG_DEBUG("Message with quotes: \"hello\"");
  MATX_LOG_DEBUG("Message with backslash: \\");
  
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents = ReadFileContents(test_log_file_);
  // Just verify they don't crash
  EXPECT_GT(contents.length(), 0u);
}

// Test file creation and appending
TEST_F(LoggingComprehensiveTest, FileAppending) {
  setenv("MATX_LOG_LEVEL", "INFO", 1);
  setenv("MATX_LOG_DEST", test_log_file_.c_str(), 1);
  detail::Logger::instance().reinitialize();
  
  MATX_LOG_INFO("First message");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents1 = ReadFileContents(test_log_file_);
  EXPECT_TRUE(ContainsPattern(contents1, "First message"));
  
  // Log another message
  MATX_LOG_INFO("Second message");
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  std::string contents2 = ReadFileContents(test_log_file_);
  EXPECT_TRUE(ContainsPattern(contents2, "First message"));
  EXPECT_TRUE(ContainsPattern(contents2, "Second message"));
}

// Test that disabled logging has minimal overhead
TEST_F(LoggingComprehensiveTest, DisabledLoggingPerformance) {
  unsetenv("MATX_LOG_LEVEL");
  detail::Logger::instance().set_min_level(detail::LogLevel::OFF);
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // This should be very fast (just boolean checks)
  for (int i = 0; i < 100000; i++) {
    MATX_LOG_DEBUG("This won't be logged");
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  
  // 100k disabled log calls should take less than 100ms (very generous)
  EXPECT_LT(duration, 100);
}

// Test invalid log destination fallback
TEST_F(LoggingComprehensiveTest, InvalidDestinationFallback) {
  setenv("MATX_LOG_LEVEL", "INFO", 1);
  setenv("MATX_LOG_DEST", "/invalid/path/that/does/not/exist/log.txt", 1);
  
  // This should fallback to stdout and not crash
  detail::Logger::instance().reinitialize();
  MATX_LOG_INFO("Fallback test");
  
  SUCCEED();
}

