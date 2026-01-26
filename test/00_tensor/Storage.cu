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
#include <memory>
#include <atomic>
#include <cstdlib>
#include <cstring>

using namespace matx;

// Test Custom Allocators - NO INHERITANCE REQUIRED!

/**
 * @brief TrackedAllocator - tracks allocation/deallocation calls
 */
class TrackedAllocator {
private:
    static std::atomic<size_t> allocations_;
    static std::atomic<size_t> total_bytes_;
    std::string name_;
    
public:
    TrackedAllocator(const std::string& name = "TrackedAlloc") : name_(name) {}
    
    // Duck-typed interface - just implement these methods!
    void* allocate(size_t bytes) {
        void* ptr = std::malloc(bytes);
        allocations_++;
        total_bytes_ += bytes;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        allocations_--;
        total_bytes_ -= bytes;
        std::free(ptr);
    }
    
    static size_t getActiveAllocations() { return allocations_.load(); }
    static size_t getTotalBytes() { return total_bytes_.load(); }
    static void reset() { 
        allocations_ = 0; 
        total_bytes_ = 0;
    }
};

std::atomic<size_t> TrackedAllocator::allocations_{0};
std::atomic<size_t> TrackedAllocator::total_bytes_{0};

/**
 * @brief PoolAllocator - allocates from a fixed-size memory pool
 */
class PoolAllocator {
private:
    std::shared_ptr<char[]> pool_;
    size_t pool_size_;
    std::shared_ptr<size_t> offset_;  // Shared to support copying
    
public:
    PoolAllocator(size_t size) 
      : pool_(new char[size], std::default_delete<char[]>{}), 
        pool_size_(size),
        offset_(std::make_shared<size_t>(0)) {}
    
    // Duck-typed interface
    void* allocate(size_t bytes) {
        if (*offset_ + bytes > pool_size_) {
            throw std::bad_alloc();
        }
        void* ptr = pool_.get() + *offset_;
        *offset_ += bytes;
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        // Pool allocator doesn't deallocate individual allocations
        (void)ptr;    // Unused
        (void)bytes;  // Unused
    }
    
    size_t getUsedBytes() const { return *offset_; }
    size_t getRemainingBytes() const { return pool_size_ - *offset_; }
    void reset() { *offset_ = 0; }
};

/**
 * @brief AlignedAllocator - allocates memory with custom alignment
 */
class AlignedAllocator {
private:
    size_t alignment_;
    
public:
    AlignedAllocator(size_t alignment = 64) : alignment_(alignment) {}
    
    // Duck-typed interface
    void* allocate(size_t bytes) {
        size_t aligned_bytes = (bytes + alignment_ - 1) & ~(alignment_ - 1);
        void* ptr = std::aligned_alloc(alignment_, aligned_bytes);
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        (void)bytes;  // Unused - size not needed for std::free
        std::free(ptr);
    }
    
    size_t getAlignment() const { return alignment_; }
};

/**
 * @brief DebugAllocator - fills memory with patterns and checks for overwrites
 */
class DebugAllocator {
private:
    static constexpr uint8_t ALLOC_PATTERN = 0xAA;
    static constexpr uint8_t FREE_PATTERN = 0xDD;
    
public:
    void* allocate(size_t bytes) {
        void* ptr = std::malloc(bytes);
        if (ptr) {
            std::memset(ptr, ALLOC_PATTERN, bytes);
        }
        return ptr;
    }
    
    void deallocate(void* ptr, size_t bytes) {
        if (ptr) {
            std::memset(ptr, FREE_PATTERN, bytes);
        }
        std::free(ptr);
    }
};

// Test fixture for Storage tests
class StorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        TrackedAllocator::reset();
    }
    
    void TearDown() override {
        // Verify all allocations are freed
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0) 
            << "Memory leak detected: " << TrackedAllocator::getActiveAllocations() 
            << " allocations still active";
    }
};

// Test basic storage creation with default allocator
TEST_F(StorageTest, DefaultAllocator) {
    Storage<float> storage(1000);
    EXPECT_EQ(storage.size(), 1000);
    EXPECT_NE(storage.data(), nullptr);
    
    // Write and read data
    storage.data()[0] = 3.14f;
    storage.data()[999] = 2.718f;
    EXPECT_FLOAT_EQ(storage.data()[0], 3.14f);
    EXPECT_FLOAT_EQ(storage.data()[999], 2.718f);
}

// Test non-owning storage
TEST_F(StorageTest, NonOwningStorage) {
    float data[100];
    Storage<float> storage(data, 100);
    
    EXPECT_EQ(storage.size(), 100);
    EXPECT_EQ(storage.data(), data);
    
    // Verify we can read/write
    storage.data()[0] = 1.0f;
    EXPECT_FLOAT_EQ(data[0], 1.0f);
}

// Test TrackedAllocator with object
TEST_F(StorageTest, TrackedAllocatorObject) {
    TrackedAllocator alloc("TestAlloc");
    
    {
        Storage<float> storage(1000, alloc);
        EXPECT_EQ(storage.size(), 1000);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        EXPECT_EQ(TrackedAllocator::getTotalBytes(), 1000 * sizeof(float));
        
        // Use the storage
        storage.data()[0] = 42.0f;
        EXPECT_FLOAT_EQ(storage.data()[0], 42.0f);
    }
    
    // Verify cleanup
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
}

// Test TrackedAllocator with pointer
TEST_F(StorageTest, TrackedAllocatorPointer) {
    TrackedAllocator alloc("TestAlloc");
    
    {
        Storage<double> storage(500, &alloc);
        EXPECT_EQ(storage.size(), 500);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        EXPECT_EQ(TrackedAllocator::getTotalBytes(), 500 * sizeof(double));
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
}

// Test PoolAllocator
TEST_F(StorageTest, PoolAllocator) {
    const size_t pool_size = 10000;
    PoolAllocator pool(pool_size);
    
    Storage<int> storage1(100, pool);
    Storage<float> storage2(200, pool);
    
    EXPECT_EQ(storage1.size(), 100);
    EXPECT_EQ(storage2.size(), 200);
    
    // Verify pool usage
    size_t expected_usage = 100 * sizeof(int) + 200 * sizeof(float);
    EXPECT_EQ(pool.getUsedBytes(), expected_usage);
    EXPECT_EQ(pool.getRemainingBytes(), pool_size - expected_usage);
    
    // Write data to both storages
    storage1.data()[0] = 42;
    storage2.data()[0] = 3.14f;
    EXPECT_EQ(storage1.data()[0], 42);
    EXPECT_FLOAT_EQ(storage2.data()[0], 3.14f);
}

// Test PoolAllocator exhaustion
TEST_F(StorageTest, PoolAllocatorExhaustion) {
    const size_t pool_size = 1000;
    PoolAllocator pool(pool_size);
    
    // This should succeed
    Storage<char> storage1(500, pool);
    
    // This should also succeed
    Storage<char> storage2(400, pool);
    
    // This should fail (not enough space)
    EXPECT_THROW({
        Storage<char> storage3(200, pool);
    }, std::bad_alloc);
}

// Test AlignedAllocator
TEST_F(StorageTest, AlignedAllocator) {
    const size_t alignment = 128;
    AlignedAllocator aligned(alignment);
    
    Storage<double> storage(64, aligned);
    
    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(storage.data());
    EXPECT_EQ(addr % alignment, 0) << "Memory not properly aligned";
    
    // Verify we can use the storage
    storage.data()[0] = 1.23;
    EXPECT_DOUBLE_EQ(storage.data()[0], 1.23);
}

// Test DebugAllocator
TEST_F(StorageTest, DebugAllocator) {
    DebugAllocator debug;
    
    {
        Storage<uint8_t> storage(100, debug);
        
        // Verify memory is filled with allocation pattern
        bool all_match = true;
        for (size_t i = 0; i < 100; ++i) {
            if (storage.data()[i] != 0xAA) {
                all_match = false;
                break;
            }
        }
        EXPECT_TRUE(all_match) << "Debug allocator didn't fill memory with expected pattern";
        
        // Modify some data
        storage.data()[0] = 0xFF;
        storage.data()[99] = 0xEE;
    }
    // Storage destroyed, memory should be freed
}

// Test make_owning_storage factory function
TEST_F(StorageTest, FactoryFunction) {
    TrackedAllocator alloc;
    
    {
        auto storage = make_owning_storage<float>(777, alloc);
        EXPECT_EQ(storage.size(), 777);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
        
        // Use the storage
        storage.data()[0] = 3.14f;
        EXPECT_FLOAT_EQ(storage.data()[0], 3.14f);
    }
    
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 0);
}

// Test shared ownership
TEST_F(StorageTest, SharedOwnership) {
    TrackedAllocator alloc;
    
    Storage<int> original(100, alloc);
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
    
    // Write some data
    original.data()[0] = 42;
    original.data()[99] = 999;
    
    {
        Storage<int> copy = original;
        EXPECT_EQ(original.use_count(), 2);
        EXPECT_EQ(copy.use_count(), 2);
        EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1); // Still just one allocation
        
        // Verify they share the same data
        EXPECT_EQ(original.data(), copy.data());
        EXPECT_EQ(copy.data()[0], 42);
        EXPECT_EQ(copy.data()[99], 999);
        
        // Modify through copy
        copy.data()[50] = 500;
    }
    
    // Copy destroyed, original should still be valid
    EXPECT_EQ(original.use_count(), 1);
    EXPECT_EQ(original.data()[50], 500); // Change made through copy is visible
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
}

// Test copy semantics (Storage uses shared_ptr, so it's copy-based sharing)
TEST_F(StorageTest, CopySemantics) {
    TrackedAllocator alloc;
    
    Storage<float> original(200, alloc);
    original.data()[0] = 1.23f;
    
    float* original_ptr = original.data();
    
    // Copy construction (Storage doesn't have explicit move semantics)
    Storage<float> copied(original);
    EXPECT_EQ(copied.data(), original_ptr);
    EXPECT_EQ(copied.size(), 200);
    EXPECT_FLOAT_EQ(copied.data()[0], 1.23f);
    
    // Both should share the same data
    EXPECT_EQ(original.data(), copied.data());
    EXPECT_EQ(original.use_count(), 2);
    EXPECT_EQ(copied.use_count(), 2);
    
    // Copy assignment
    Storage<float> assigned;
    assigned = copied;
    EXPECT_EQ(assigned.data(), original_ptr);
    EXPECT_EQ(assigned.size(), 200);
    EXPECT_EQ(original.use_count(), 3);
    
    // Cleanup check - still only one allocation
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
}

// Test multiple allocators simultaneously
TEST_F(StorageTest, MultipleAllocators) {
    TrackedAllocator tracked1("Alloc1");
    TrackedAllocator tracked2("Alloc2");
    AlignedAllocator aligned(256);
    DebugAllocator debug;
    
    Storage<float> s1(100, tracked1);
    Storage<double> s2(200, tracked2);
    Storage<int> s3(50, aligned);
    Storage<char> s4(1000, debug);
    
    // Verify all storages are valid
    EXPECT_EQ(s1.size(), 100);
    EXPECT_EQ(s2.size(), 200);
    EXPECT_EQ(s3.size(), 50);
    EXPECT_EQ(s4.size(), 1000);
    
    // Verify tracked allocator is tracking correctly
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 2);
    
    // Verify alignment
    uintptr_t addr = reinterpret_cast<uintptr_t>(s3.data());
    EXPECT_EQ(addr % 256, 0);
}

// Test with tensor integration (if tensor_t is available)
TEST_F(StorageTest, TensorIntegration) {
    TrackedAllocator alloc;
    
    // Create storage with custom allocator
    auto storage = make_owning_storage<float>(1000, alloc);
    
    // Create tensor from storage
    DefaultDescriptor<1> desc({1000});
    tensor_t<float, 1> tensor(std::move(storage), std::move(desc));
    
    EXPECT_EQ(tensor.Size(0), 1000);
    EXPECT_EQ(TrackedAllocator::getActiveAllocations(), 1);
    
    // Use the tensor
    tensor(0) = 3.14f;
    tensor(999) = 2.718f;
    EXPECT_FLOAT_EQ(tensor(0), 3.14f);
    EXPECT_FLOAT_EQ(tensor(999), 2.718f);
}

// Test iterator interface
TEST_F(StorageTest, IteratorInterface) {
    Storage<int> storage(10);
    
    // Fill with values
    int value = 0;
    for (auto& elem : storage) {
        elem = value++;
    }
    
    // Verify with const iterator
    const Storage<int>& const_storage = storage;
    value = 0;
    for (const auto& elem : const_storage) {
        EXPECT_EQ(elem, value++);
    }
    
    // Test iterator arithmetic
    EXPECT_EQ(storage.end() - storage.begin(), 10);
    EXPECT_EQ(*(storage.begin() + 5), 5);
}

// Test empty storage
TEST_F(StorageTest, EmptyStorage) {
    Storage<float> empty;
    
    EXPECT_EQ(empty.size(), 0);
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_EQ(empty.begin(), empty.end());
    EXPECT_EQ(empty.use_count(), 0);
    
    // Copy of empty storage
    Storage<float> copy = empty;
    EXPECT_EQ(copy.size(), 0);
    EXPECT_EQ(copy.data(), nullptr);
}

// Test type traits for allocator detection
TEST_F(StorageTest, AllocatorTypeTraits) {
    // These should have allocator interface
    EXPECT_TRUE(has_allocator_interface<TrackedAllocator>::value);
    EXPECT_TRUE(has_allocator_interface<PoolAllocator>::value);
    EXPECT_TRUE(has_allocator_interface<AlignedAllocator>::value);
    EXPECT_TRUE(has_allocator_interface<DebugAllocator>::value);
    
    // Pointer versions should also work
    EXPECT_TRUE(has_allocator_interface<TrackedAllocator*>::value);
    EXPECT_TRUE(has_allocator_interface<PoolAllocator*>::value);
    
    // These should not have allocator interface
    struct NotAnAllocator {};
    EXPECT_FALSE(has_allocator_interface<NotAnAllocator>::value);
    EXPECT_FALSE(has_allocator_interface<int>::value);
    EXPECT_FALSE(has_allocator_interface<float*>::value);
}
