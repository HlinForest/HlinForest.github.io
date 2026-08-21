#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <span>

class Buffer {
public:
    explicit Buffer(std::size_t size)
        : data_(std::make_unique<float[]>(size)), size_(size) {}

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) noexcept = default;
    Buffer& operator=(Buffer&&) noexcept = default;
    ~Buffer() = default;

    std::span<float> view() { return {data_.get(), size_}; }
    std::span<const float> view() const { return {data_.get(), size_}; }

private:
    std::unique_ptr<float[]> data_;
    std::size_t size_;
};

void scale(std::span<float> values, float factor) {
    for (float& value : values) {
        value *= factor;
    }
}

int main() {
    Buffer buffer(4);
    auto values = buffer.view();
    std::ranges::copy(std::initializer_list<float>{1, 2, 3, 4}, values.begin());
    scale(values, 2.0F);
    assert(values[0] == 2.0F && values[3] == 8.0F);
    std::cout << "PASS: RAII buffer owns " << values.size() << " values\n";
}

// EXERCISES
// 1. Return a span to a local Buffer from a function. Explain why it dangles,
//    then remove the unsafe code.
// 2. Add a copy constructor and measure the semantic cost. Decide whether
//    copying should remain deleted.
// 3. Build with AddressSanitizer when your compiler supports it.
// 4. Extend Buffer with a const-correct sum() operation.
