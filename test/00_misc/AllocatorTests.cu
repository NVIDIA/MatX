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
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace matx;

// Example 1: Simple custom allocator - NO INHERITANCE REQUIRED!
class TrackedAllocator {
private:
    static inline std::unordered_map<void*, size_t> allocations_;
    static inline size_t total_allocated_ = 0;
    
public:
    // Duck-typed interface - just implement these methods!
    void* allocate(size_t bytes) {
        void* ptr = std::malloc(bytes);
        allocations_[ptr] = bytes;
        total_allocated_ += bytes;
        std::cout << "TrackedAllocator::allocate(" << bytes << ") -> " << ptr << std::endl;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_allocated_ -= it->second;
            allocations_.erase(it);
        }
        std::cout << "TrackedAllocator::deallocate(" << ptr << ", " << bytes << ")" << std::endl;
        std::free(ptr);
    }
    
    // Utility methods for testing
    static size_t getTotalAllocated() { return total_allocated_; }
    static size_t getActiveAllocations() { return allocations_.size(); }
    static void reset() { 
        allocations_.clear(); 
        total_allocated_ = 0; 
    }
};

// Example 2: Pool allocator - also NO INHERITANCE!
class PoolAllocator {
private:
    std::vector<char> pool_;
    size_t offset_ = 0;
    
public:
    PoolAllocator(size_t pool_size) : pool_(pool_size) {
        std::cout << "PoolAllocator created with " << pool_size << " bytes" << std::endl;
    }
    
    // Duck-typed interface
    void* allocate(size_t bytes) {
        if (offset_ + bytes > pool_.size()) {
            throw std::bad_alloc();
        }
        void* ptr = pool_.data() + offset_;
        offset_ += bytes;
        std::cout << "PoolAllocator::allocate(" << bytes << ") -> " << ptr 
                  << " (offset now " << offset_ << ")" << std::endl;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        // Simple pool - just log the deallocation
        std::cout << "PoolAllocator::deallocate(" << ptr << ", " << bytes << ")" << std::endl;
    }
    
    size_t getBytesUsed() const { return offset_; }
};

// Example 3: Debug allocator with alignment
struct AlignedDebugAllocator {
    size_t alignment_;
    
    AlignedDebugAllocator(size_t alignment = 64) : alignment_(alignment) {}
    
    // Duck-typed interface
    void* allocate(size_t bytes) {
        size_t aligned_bytes = (bytes + alignment_ - 1) & ~(alignment_ - 1);
        void* ptr = std::aligned_alloc(alignment_, aligned_bytes);
        std::cout << "AlignedDebugAllocator::allocate(" << bytes 
                  << ") aligned to " << aligned_bytes << " -> " << ptr << std::endl;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        std::cout << "AlignedDebugAllocator::deallocate(" << ptr << ", " << bytes << ")" << std::endl;
        std::free(ptr);
    }
};

class DuckTypingAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        TrackedAllocator::reset();
    }
    
    void TearDown() override {
        TrackedAllocator::reset();
    }
};

TEST_F(DuckTypingAllocatorTest, TrackedAllocatorObject) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing TrackedAllocator with Storage (object) ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Use allocator object directly - no inheritance required!
        Storage<float> storage(1000, alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 1000);
        EXPECT_GT(TrackedAllocator::getTotalAllocated(), 0);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Test the storage actually works
        storage.data()[0] = 42.0f;
        storage.data()[999] = 99.0f;
        EXPECT_EQ(storage.data()[0], 42.0f);
        EXPECT_EQ(storage.data()[999], 99.0f);
    }
    
    // After destruction, memory should be freed
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, TrackedAllocatorPointer) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing TrackedAllocator with Storage (pointer) ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Use allocator pointer - also works!
        Storage<double> storage(500, &alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 500);
        EXPECT_GT(TrackedAllocator::getTotalAllocated(), 0);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, FactoryFunction) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing TrackedAllocator with make_owning_storage ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Use with factory function
        auto storage = make_owning_storage<int>(2000, alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 2000);
        EXPECT_GT(TrackedAllocator::getTotalAllocated(), 0);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, PoolAllocator) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing PoolAllocator ===" << std::endl;
    
    PoolAllocator pool(10000);  // 10KB pool
    
    {
        Storage<float> storage1(100, pool);  // 400 bytes
        Storage<int> storage2(200, pool);    // 800 bytes
        
        EXPECT_NE(storage1.data(), nullptr);
        EXPECT_NE(storage2.data(), nullptr);
        EXPECT_EQ(storage1.size(), 100);
        EXPECT_EQ(storage2.size(), 200);
        
        // Pool should have allocated bytes
        EXPECT_GT(pool.getBytesUsed(), 0);
        
        // Test the storages work
        storage1.data()[0] = 3.14f;
        storage2.data()[0] = 42;
        EXPECT_EQ(storage1.data()[0], 3.14f);
        EXPECT_EQ(storage2.data()[0], 42);
    }
    
    std::cout << "Pool used " << pool.getBytesUsed() << " bytes total" << std::endl;
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, AlignedAllocator) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing AlignedDebugAllocator ===" << std::endl;
    
    AlignedDebugAllocator aligned_alloc(128);  // 128-byte alignment
    
    {
        Storage<double> storage(64, aligned_alloc);
        
        EXPECT_NE(storage.data(), nullptr);
        EXPECT_EQ(storage.size(), 64);
        
        // Check alignment
        uintptr_t addr = reinterpret_cast<uintptr_t>(storage.data());
        EXPECT_EQ(addr % 128, 0);  // Should be 128-byte aligned
        
        std::cout << "Storage pointer: " << storage.data() 
                  << " (aligned to 128 bytes)" << std::endl;
    }
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, SharedOwnership) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing Shared Ownership ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        Storage<int> storage1(100, alloc);
        EXPECT_EQ(storage1.use_count(), 1);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        {
            Storage<int> storage2 = storage1;  // Copy constructor
            EXPECT_EQ(storage1.use_count(), 2);
            EXPECT_EQ(storage2.use_count(), 2);
            EXPECT_EQ(storage1.data(), storage2.data());  // Same data
            EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);  // Still one allocation
        }
        
        // After storage2 is destroyed, storage1 should still work
        EXPECT_EQ(storage1.use_count(), 1);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        storage1.data()[0] = 42;
        EXPECT_EQ(storage1.data()[0], 42);
    }
    
    // Now memory should be freed
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

TEST_F(DuckTypingAllocatorTest, CompareWithDefault) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Comparing Custom vs Default Allocator ===" << std::endl;
    
    TrackedAllocator custom_alloc;
    
    {
        Storage<float> default_storage(1000);           // Default allocator
        Storage<float> custom_storage(1000, custom_alloc);  // Custom allocator
        
        EXPECT_NE(default_storage.data(), nullptr);
        EXPECT_NE(custom_storage.data(), nullptr);
        EXPECT_EQ(default_storage.size(), 1000);
        EXPECT_EQ(custom_storage.size(), 1000);
        
        // Only custom allocator should be tracked
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Both should work the same way
        default_storage.data()[0] = 1.0f;
        custom_storage.data()[0] = 2.0f;
        EXPECT_EQ(default_storage.data()[0], 1.0f);
        EXPECT_EQ(custom_storage.data()[0], 2.0f);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}

// Demonstrate that it works with tensor_t too!
TEST_F(DuckTypingAllocatorTest, WithTensorAPI) {
    MATX_ENTER_HANDLER();
    std::cout << "\n=== Testing with Tensor API ===" << std::endl;
    
    TrackedAllocator alloc;
    
    {
        // Create storage with custom allocator
        auto storage = make_owning_storage<float>(1000, alloc);
        
        // Use with tensor_t (this shows the full integration)
        DefaultDescriptor<1> desc({1000});
        tensor_t<float, 1> tensor(std::move(storage), std::move(desc));
        
        EXPECT_NE(tensor.Data(), nullptr);
        EXPECT_EQ(tensor.Size(0), 1000);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Test tensor operations
        tensor(0) = 3.14f;
        tensor(999) = 2.71f;
        EXPECT_EQ(tensor(0), 3.14f);
        EXPECT_EQ(tensor(999), 2.71f);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
    
    MATX_EXIT_HANDLER();
}
