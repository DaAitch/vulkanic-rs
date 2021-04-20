# `vulkanic`

Ultra-thin Vulkan API wrapper sitting on top of [vk-sys](https://docs.rs/vk-sys).

## Why another Vulkan wrapper?

Vulkan API is a very performant, low-level graphics and computation API.
Drawbacks are the complex usage and e.g. a simple "hello world"-triangle with Vulkan has about
1000 lines of code using the the plain vk-sys bindings and that's a lot.

Many resources out there about Vulkan are (still) in C++. It's way easier to adapt C++ examples
without any library indirection, so using Vulkan through `vulkanic` crate will allow you
to understand the API reading the official specs.

## Other Vulkan wrappers / libraries
- [ash](https://docs.rs/ash): *A very lightweight wrapper around Vulkan.*
- [vulkano](https://docs.rs/vulkano): *A Rust wrapper around the Vulkan graphics API.*

## `vulkanic` features

- adapter for vk-sys ✅
  - `vk::EntryPoints` ✅
  - `vk::InstancePointers` ✅
  - `vk::DevicePointers` ✅
- fewer `unsafe`s ✅
- zero-cost adapter ✅
- no Vulkan allocation-callback utilization ❌
- no window creation ❌
- no validatation ❌
- no builder patterns ❌
- no Vulkan abstraction ❌
- no shader compilation ❌
