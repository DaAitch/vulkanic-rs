//! # `vulkanic` - Vulkan API adapter
//!
//! [Vulkan Specs](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/index.html)
//!
//! Ultra-thin wrapper sitting on top of [vk-sys](https://docs.rs/vk-sys).
//!
//! ## Examples
//!
//! **GLFW:**
//!
//! ```
//! let ep: EntryPoints = vk::EntryPoints::load(|procname| {
//!     glfw_window.get_instance_proc_address(0, procname.to_str().unwrap())
//! }).into();
//!
//! // let instance_info = ...
//!
//! let instance = unsafe { ep.create_instance(&instance_info) }?;
//!
//! let ip: InstancePointers = vk::InstancePointers::load(|procname| {
//!     glfw_window.get_instance_proc_address(instance, procname.to_str().unwrap())
//! }).into();
//! let dp: DevicePointers = vk::DevicePointers::load(|procname| {
//!     glfw_window.get_instance_proc_address(instance, procname.to_str().unwrap())
//! }).into();
//!
//! for physical_device in ip.enumerate_physical_devices(instance)? {
//!     // ...
//! }
//!
//! // ...
//!
//! // use pointers directly, if wrapper implementation is missing
//! // unsafe { dp.InvalidateMappedMemoryRanges(..) }
//!
//! ```
//!
//! ## Production readiness
//!
//!
//!
//!
//!
//! ## Crate conventions
//!
//! - same partitioning like `vk-sys` with `EntryPoints`, `InstancePointers` and `DevicePointers`
//! - Rust naming conventions are met
//!   - function names e.g. `CreateDevice` renamed to `create_device`
//!   - parameter names e.g. `physicalDevice` renamed to `physical_device`
//!   - C type prefixes are omitted e.g. `pCreateInfo` to `create_info`
//! - type mapping
//!   - `vk::Bool32` to `bool`
//!   - pointers `pCreateInfo: *const DeviceCreateInfo` to borrows `create_info: &DeviceCreateInfo`
//!   - API calls having array parameters e.g. `fenceCount: u32, pFences: *const Fence` will be presented
//!     as Rust slices e.g. `fences: &[vk::Fence]`.
//! - return values
//!   - `std::result::Result<T, vk::Result>` if API call has a result
//!   - `T` if API call has no result
//!   - API calls with a result, will return `Ok(..)` **only** for `vk::SUCCESS`
//!
//! ## Missing parts
//!
//! - many Vulkan API calls are still not yet implemented
//! - API calls like `vkAcquireNextImageKHR` can return non-error results which are currently handled
//!   as error. It's not clear (even from the specs) if out-parameter are valid in this case.
//!
//! ## Safety
//!
//! - many Vulkan API calls are safe Rust regarding they may fail, but will not have undefined behaviour
//! - all Vulkan API calls with structs containing pointers like `pNext: *const c_void` have to be
//!   considered unsafe, because caller has to make sure they are valid.
//! - using `Vec` or slices as ffi pointers is safe, because all vk-sys structs are `#[repr(C)]`, which
//!   ensures correct alignment without padding.
//!

use std::{ffi::CString, ops::Deref, ptr};
use util::{
    vk_bool, vk_call, vk_get, vk_get_many, vk_query, vk_query_many, vk_query_n, vk_run,
    vk_slice_len,
};
use vk_sys as vk;

/// Wrapper for `vk::EntryPoints`.
pub struct EntryPoints {
    ptr: vk::EntryPoints,
}

/// Result type for most Vulkan API calls, if they support them.
/// Currently non-error return values are handled as `Err(..)` (also see Missing parts on top).
pub type Result<T> = std::result::Result<T, vk::Result>;

impl EntryPoints {
    /// Create new `vk::EntryPoints` wrapper.
    pub fn new(ptr: vk::EntryPoints) -> Self {
        Self { ptr }
    }

    /// Unwraps `vk::EntryPoints`.
    pub fn into_inner(self) -> vk::EntryPoints {
        self.ptr
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateInstance.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_instance(
        &self,
        create_info: &vk::InstanceCreateInfo,
    ) -> Result<vk::Instance> {
        vk_query(|data| self.ptr.CreateInstance(create_info, ptr::null(), data))
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkEnumerateInstanceExtensionProperties.html>
    #[inline]
    pub fn enumerate_instance_extension_properties(&self) -> Result<Vec<vk::ExtensionProperties>> {
        vk_query_many(|count, data| unsafe {
            self.ptr
                .EnumerateInstanceExtensionProperties(std::ptr::null(), count, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkEnumerateInstanceLayerProperties.html>
    #[inline]
    pub fn enumerate_instance_layer_properties(&self) -> Result<Vec<vk::LayerProperties>> {
        vk_query_many(|count, data| unsafe {
            self.ptr.EnumerateInstanceLayerProperties(count, data)
        })
    }
}

impl Deref for EntryPoints {
    type Target = vk::EntryPoints;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl<T: Into<vk::EntryPoints>> From<T> for EntryPoints {
    fn from(ptr: T) -> Self {
        EntryPoints::new(ptr.into())
    }
}

/// Wrapper for `vk::InstancePointers`.
pub struct InstancePointers {
    ptr: vk::InstancePointers,
}

impl InstancePointers {
    /// Create new `vk::InstancePointers` wrapper.
    pub fn new(ptr: vk::InstancePointers) -> Self {
        Self { ptr }
    }

    /// Unwraps `vk::InstancePointers`.
    pub fn into_inner(self) -> vk::InstancePointers {
        self.ptr
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyInstance.html>
    #[inline]
    pub fn destroy_instance(&self, instance: vk::Instance) {
        vk_run(|| unsafe {
            self.ptr.DestroyInstance(instance, ptr::null());
        });
    }

    // get_device_proc_addr

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkEnumeratePhysicalDevices.html>
    #[inline]
    pub fn enumerate_physical_devices(
        &self,
        instance: vk::Instance,
    ) -> Result<Vec<vk::PhysicalDevice>> {
        vk_query_many(|count, data| unsafe {
            self.ptr.EnumeratePhysicalDevices(instance, count, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkEnumerateDeviceExtensionProperties.html>
    #[inline]
    pub fn enumerate_device_extension_properties<T: Into<Vec<u8>>>(
        &self,
        physical_device: vk::PhysicalDevice,
        layer_name: Option<&CString>,
    ) -> Result<Vec<vk::ExtensionProperties>> {
        vk_query_many(|count, data| unsafe {
            self.ptr.EnumerateDeviceExtensionProperties(
                physical_device,
                layer_name.map(|cstr| cstr.as_ptr()).unwrap_or(ptr::null()),
                count,
                data,
            )
        })
    }

    // enumerate_device_layer_properties

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateDevice.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_device(
        &self,
        physical_device: vk::PhysicalDevice,
        create_info: &vk::DeviceCreateInfo,
    ) -> Result<vk::Device> {
        vk_query(|data| {
            self.ptr
                .CreateDevice(physical_device, create_info, std::ptr::null(), data)
        })
    }

    // get_physical_device_features

    // get_physical_device_format_properties

    // get_physical_device_image_format_properties

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceProperties.html>
    #[inline]
    pub fn get_physical_device_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> vk::PhysicalDeviceProperties {
        vk_get(|data| {
            unsafe { self.ptr.GetPhysicalDeviceProperties(physical_device, data) };
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceQueueFamilyProperties.html>
    #[inline]
    pub fn get_physical_device_queue_family_properties(
        &self,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<vk::QueueFamilyProperties> {
        vk_get_many(|count, data| unsafe {
            self.ptr
                .GetPhysicalDeviceQueueFamilyProperties(physical_device, count, data)
        })
    }

    // get_physical_device_memory_properties

    // get_physical_device_sparse_image_format_properties

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroySurfaceKHR.html>
    #[inline]
    pub fn destroy_surface_khr(&self, instance: vk::Instance, surface: vk::SurfaceKHR) {
        vk_run(|| unsafe {
            self.ptr.DestroySurfaceKHR(instance, surface, ptr::null());
        });
    }

    // create_xlib_surface_khr

    // get_physical_device_xlib_presentation_support_khr

    // create_xcb_surface_khr

    // get_physical_device_xcb_presentation_support_khr

    // create_wayland_surface_khr

    // get_physical_device_wayland_presentation_support_khr

    // create_android_surface_khr

    // create_win32_surface_khr

    // get_physical_device_win32_presentation_support_khr

    // get_physical_device_display_properties_khr

    // get_physical_device_display_plane_properties_khr

    // get_display_plane_supported_displays_khr

    // get_display_mode_properties_khr

    // create_display_mode_khr

    // get_display_plane_capabilities_khr

    // create_display_plane_surface_khr

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceSurfaceSupportKHR.html>
    #[inline]
    pub fn get_physical_device_surface_support_khr(
        &self,
        physical_device: vk::PhysicalDevice,
        queue_family_index: u32,
        surface: vk::SurfaceKHR,
    ) -> Result<bool> {
        vk_query(|data| unsafe {
            self.ptr.GetPhysicalDeviceSurfaceSupportKHR(
                physical_device,
                queue_family_index,
                surface,
                data,
            )
        })
        .map(|supported| supported == vk::TRUE)
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceSurfaceCapabilitiesKHR.html>
    #[inline]
    pub fn get_physical_device_surface_capabilities_khr(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<vk::SurfaceCapabilitiesKHR> {
        vk_query(|data| unsafe {
            self.ptr
                .GetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceSurfaceFormatsKHR.html>
    #[inline]
    pub fn get_physical_device_surface_formats_khr(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Vec<vk::SurfaceFormatKHR>> {
        vk_query_many(|count, data| unsafe {
            self.ptr
                .GetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, count, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetPhysicalDeviceSurfacePresentModesKHR.html>
    #[inline]
    pub fn get_physical_device_surface_present_modes_khr(
        &self,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Vec<vk::PresentModeKHR>> {
        vk_query_many(|count, data| unsafe {
            self.ptr
                .GetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, count, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateDebugUtilsMessengerEXT.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_debug_utils_messenger_ext(
        &self,
        instance: vk::Instance,
        create_info: &vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> Result<vk::DebugUtilsMessengerEXT> {
        vk_query(|data| {
            self.ptr
                .CreateDebugUtilsMessengerEXT(instance, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyDebugUtilsMessengerEXT.html>
    #[inline]
    pub fn destroy_debug_utils_messenger_ext(
        &self,
        instance: vk::Instance,
        messenger: vk::DebugUtilsMessengerEXT,
    ) -> Result<()> {
        vk_call(|| unsafe {
            self.ptr
                .DestroyDebugUtilsMessengerEXT(instance, messenger, ptr::null())
        })
    }

    // create_ios_surface_mvk

    // create_macos_surface_mvk

    // activate_moltenvk_license_mvk

    // activate_moltenvk_licenses_mvk

    // get_moltenvk_device_configuration_mvk

    // set_moltenvk_device_configuration_mvk

    // get_physical_device_metal_features_mvk

    // get_swapchain_performance_mvk

    // create_vi_surface_nn

    // get_physical_device_features2_khr

    // get_physical_device_properties2_khr

    // get_physical_device_format_properties2_khr

    // get_physical_device_image_format_properties2_khr

    // get_physical_device_queue_family_properties2_khr

    // get_physical_device_memory_properties2_khr

    // get_physical_device_sparse_image_format_properties2_khr
}

impl Deref for InstancePointers {
    type Target = vk::InstancePointers;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl<T: Into<vk::InstancePointers>> From<T> for InstancePointers {
    fn from(ptr: T) -> Self {
        InstancePointers::new(ptr.into())
    }
}

/// Wrapper for `vk::DevicePointers`.
pub struct DevicePointers {
    ptr: vk::DevicePointers,
}

impl DevicePointers {
    /// Create new `vk::DevicePointers` wrapper.
    pub fn new(ptr: vk::DevicePointers) -> Self {
        Self { ptr }
    }

    /// Unwraps `vk::DevicePointers`.
    pub fn into_inner(self) -> vk::DevicePointers {
        self.ptr
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyDevice.html>
    #[inline]
    pub fn destroy_device(&self, device: vk::Device) {
        vk_run(|| unsafe {
            self.ptr.DestroyDevice(device, ptr::null());
        });
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetDeviceQueue.html>
    #[inline]
    pub fn get_device_queue(
        &self,
        device: vk::Device,
        queue_family_index: u32,
        queue_index: u32,
    ) -> vk::Queue {
        vk_get(|data| unsafe {
            self.ptr
                .GetDeviceQueue(device, queue_family_index, queue_index, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueSubmit.html>
    ///
    /// # Safety
    /// `submits` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn queue_submit(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<()> {
        vk_call(|| {
            self.ptr
                .QueueSubmit(queue, vk_slice_len(submits), submits.as_ptr(), fence)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueueWaitIdle.html>
    #[inline]
    pub fn queue_wait_idle(&self, queue: vk::Queue) -> Result<()> {
        vk_call(|| unsafe { self.ptr.QueueWaitIdle(queue) })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDeviceWaitIdle.html>
    #[inline]
    pub fn device_wait_idle(&self, device: vk::Device) -> Result<()> {
        vk_call(|| unsafe { self.ptr.DeviceWaitIdle(device) })
    }

    // allocate_memory

    // free_memory

    // map_memory

    // unmap_memory

    // flush_mapped_memory_ranges

    // invalidate_mapped_memory_ranges

    // get_device_memory_commitment

    // bind_buffer_memory

    // bind_image_memory

    // get_buffer_memory_requirements

    // get_image_memory_requirements

    // get_image_sparse_memory_requirements

    // queue_bind_sparse

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateFence.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_fence(
        &self,
        device: vk::Device,
        create_info: &vk::FenceCreateInfo,
    ) -> Result<vk::Fence> {
        vk_query(|data| self.ptr.CreateFence(device, create_info, ptr::null(), data))
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyFence.html>
    #[inline]
    pub fn destroy_fence(&self, device: vk::Device, fence: vk::Fence) {
        vk_run(|| unsafe {
            self.ptr.DestroyFence(device, fence, ptr::null());
        });
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkResetFences.html>
    #[inline]
    pub fn reset_fences(&self, device: vk::Device, fences: &[vk::Fence]) -> Result<()> {
        vk_call(|| unsafe {
            self.ptr
                .ResetFences(device, vk_slice_len(&fences), fences.as_ptr())
        })
    }

    // get_fence_status

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkWaitForFences.html>
    #[inline]
    pub fn wait_for_fences(
        &self,
        device: vk::Device,
        fences: &[vk::Fence],
        wait_all: bool,
        timeout: u64,
    ) -> Result<()> {
        vk_call(|| unsafe {
            self.ptr.WaitForFences(
                device,
                vk_slice_len(fences),
                fences.as_ptr(),
                vk_bool(wait_all),
                timeout,
            )
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateSemaphore.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_semaphore(
        &self,
        device: vk::Device,
        create_info: &vk::SemaphoreCreateInfo,
    ) -> Result<vk::Semaphore> {
        vk_query(|data| {
            self.ptr
                .CreateSemaphore(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroySemaphore.html>
    #[inline]
    pub fn destroy_semaphore(&self, device: vk::Device, semaphore: vk::Semaphore) {
        vk_run(|| unsafe {
            self.ptr.DestroySemaphore(device, semaphore, ptr::null());
        });
    }

    // create-event

    // destroy_event

    // get_event_status

    // set_event

    // reset_event

    // create_query_pool

    // destroy_query_pool

    // get_query_pool_results

    // create_buffer

    // destroy_buffer

    // create_buffer_view

    // destroy_buffer_view

    // create_image

    // destroy_image

    // get_image_subresource_layout

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateImageView.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_image_view(
        &self,
        device: vk::Device,
        create_info: &vk::ImageViewCreateInfo,
    ) -> Result<vk::ImageView> {
        vk_query(|data| {
            self.ptr
                .CreateImageView(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyImageView.html>
    #[inline]
    pub fn destroy_image_view(&self, device: vk::Device, image_view: vk::ImageView) {
        vk_run(|| unsafe {
            self.ptr.DestroyImageView(device, image_view, ptr::null());
        });
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateShaderModule.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_shader_module(
        &self,
        device: vk::Device,
        create_info: &vk::ShaderModuleCreateInfo,
    ) -> Result<vk::ShaderModule> {
        vk_query(|data| {
            self.ptr
                .CreateShaderModule(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyShaderModule.html>
    #[inline]
    pub fn destroy_shader_module(&self, device: vk::Device, shader_module: vk::ShaderModule) {
        vk_run(|| unsafe {
            self.ptr
                .DestroyShaderModule(device, shader_module, ptr::null());
        });
    }

    // create_pipeline_cache

    // destroy_pipeline_cache

    // get_pipeline_cache_data

    // merge_pipeline_cache_data

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateGraphicsPipelines.html>
    ///
    /// # Safety
    /// `create_infos` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_graphics_pipelines(
        &self,
        device: vk::Device,
        pipeline_cache: vk::PipelineCache,
        create_infos: &[vk::GraphicsPipelineCreateInfo],
    ) -> Result<Vec<vk::Pipeline>> {
        let count = vk_slice_len(create_infos);
        vk_query_n(count, |data| {
            self.ptr.CreateGraphicsPipelines(
                device,
                pipeline_cache,
                count,
                create_infos.as_ptr(),
                ptr::null(),
                data,
            )
        })
    }

    // create_compute_pipeline

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyPipeline.html>
    #[inline]
    pub fn destroy_pipeline(&self, device: vk::Device, pipeline: vk::Pipeline) {
        vk_run(|| unsafe {
            self.ptr.DestroyPipeline(device, pipeline, ptr::null());
        });
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreatePipelineLayout.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_pipeline_layout(
        &self,
        device: vk::Device,
        create_info: &vk::PipelineLayoutCreateInfo,
    ) -> Result<vk::PipelineLayout> {
        vk_query(|data| {
            self.ptr
                .CreatePipelineLayout(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyPipelineLayout.html>
    #[inline]
    pub fn destroy_pipeline_layout(&self, device: vk::Device, pipeline_layout: vk::PipelineLayout) {
        vk_run(|| unsafe {
            self.ptr
                .DestroyPipelineLayout(device, pipeline_layout, ptr::null());
        });
    }

    // create_sampler

    // destroy_sampler

    // create_descriptor_set_layout

    // destroy_descriptor_set_layout

    // create_description_pool

    // destroy_description_pool

    // reset_description_pool

    // allocate_descriptor_sets

    // free_descriptor_sets

    // update_descriptor_sets

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateFramebuffer.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_framebuffer(
        &self,
        device: vk::Device,
        create_info: &vk::FramebufferCreateInfo,
    ) -> Result<vk::Framebuffer> {
        vk_query(|data| {
            self.ptr
                .CreateFramebuffer(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyFramebuffer.html>
    #[inline]
    pub fn destroy_framebuffer(&self, device: vk::Device, framebuffer: vk::Framebuffer) {
        vk_run(|| unsafe {
            self.ptr
                .DestroyFramebuffer(device, framebuffer, ptr::null());
        });
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateRenderPass.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_render_pass(
        &self,
        device: vk::Device,
        create_info: &vk::RenderPassCreateInfo,
    ) -> Result<vk::RenderPass> {
        vk_query(|data| {
            self.ptr
                .CreateRenderPass(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyRenderPass.html>
    #[inline]
    pub fn destroy_render_pass(&self, device: vk::Device, render_pass: vk::RenderPass) {
        vk_run(|| unsafe {
            self.ptr.DestroyRenderPass(device, render_pass, ptr::null());
        });
    }

    // get_render_area_granularity

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateCommandPool.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_command_pool(
        &self,
        device: vk::Device,
        create_info: &vk::CommandPoolCreateInfo,
    ) -> Result<vk::CommandPool> {
        vk_query(|data| {
            self.ptr
                .CreateCommandPool(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroyCommandPool.html>
    #[inline]
    pub fn destroy_command_pool(&self, device: vk::Device, command_pool: vk::CommandPool) {
        vk_run(|| unsafe {
            self.ptr
                .DestroyCommandPool(device, command_pool, ptr::null());
        });
    }

    // reset_command_pool

    // trim_command_pool

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkAllocateCommandBuffers.html>
    ///
    /// Will read `allocate_info.commandBufferCount` for size of the return vector.
    ///
    /// # Safety
    /// `allocate_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn allocate_command_buffers(
        &self,
        device: vk::Device,
        allocate_info: &vk::CommandBufferAllocateInfo,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let count = allocate_info.commandBufferCount;

        vk_query_n(count, |data| {
            self.ptr.AllocateCommandBuffers(device, allocate_info, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkFreeCommandBuffers.html>
    #[inline]
    pub fn free_command_buffers(
        &self,
        device: vk::Device,
        command_pool: vk::CommandPool,
        command_buffers: &[vk::CommandBuffer],
    ) {
        vk_run(|| unsafe {
            self.ptr.FreeCommandBuffers(
                device,
                command_pool,
                vk_slice_len(command_buffers),
                command_buffers.as_ptr(),
            );
        });
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkBeginCommandBuffer.html>
    ///
    /// # Safety
    /// `begin_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn begin_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        begin_info: &vk::CommandBufferBeginInfo,
    ) -> Result<()> {
        vk_call(|| self.ptr.BeginCommandBuffer(command_buffer, begin_info))
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkEndCommandBuffer.html>
    #[inline]
    pub fn end_command_buffer(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        vk_call(|| unsafe { self.ptr.EndCommandBuffer(command_buffer) })
    }

    // reset_command_buffer

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBindPipeline.html>
    #[inline]
    pub fn cmd_bind_pipeline(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        vk_run(|| unsafe {
            self.ptr
                .CmdBindPipeline(command_buffer, pipeline_bind_point, pipeline)
        })
    }

    // cmd_set_viewport

    // cmd_set_scissor

    // cmd_set_line_width

    // cmd_set_depth_bias

    // cmd_set_blend_constants

    // cmd_set_depth_bounds

    // cmd_set_stencil_compare_mask

    // cmd_set_stencil_write_mask

    // cmd_set_stencil_reference

    // cmd_bind_descriptor_sets

    // cmd_bind_index_buffer

    // cmd_bind_vertex_buffers

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdDraw.html>
    #[inline]
    pub fn cmd_draw(
        &self,
        command_buffer: vk::CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        vk_run(|| unsafe {
            self.ptr.CmdDraw(
                command_buffer,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        });
    }

    // cmd_draw_indexed

    // cmd_draw_indirect

    // cmd_draw_indexed_indirect

    // cmd_dispatch

    // cmd_dispatch_indirect

    // cmd_copy_buffer

    // cmd_copy_image

    // cmd_blit_image

    // cmd_copy_buffer_to_image

    // cmd_copy_image_to_buffer

    // cmd_update_buffer

    // cmd_fill_buffer

    // cmd_clear_color_image

    // cmd_clear_depth_stencil_image

    // cmd_clear_attachments

    // cmd_resolve_image

    // cmd_set_event

    // cmd_reset_event

    // cmd_wait_events

    // cmd_pipeline_barrier

    // cmd_begin_query

    // cmd_end_query

    // cmd_reset_query_pool

    // cmd_write_timestamp

    // cmd_copy_query_pool_results

    // cmd_push_constants

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdBeginRenderPass.html>
    ///
    /// # Safety
    /// `render_pass_begin` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn cmd_begin_render_pass(
        &self,
        command_buffer: vk::CommandBuffer,
        render_pass_begin: &vk::RenderPassBeginInfo,
        contents: vk::SubpassContents,
    ) {
        vk_run(|| {
            self.ptr
                .CmdBeginRenderPass(command_buffer, render_pass_begin, contents);
        });
    }

    // cmd_next_subpass

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCmdEndRenderPass.html>
    #[inline]
    pub fn cmd_end_render_pass(&self, command_buffer: vk::CommandBuffer) {
        vk_run(|| unsafe {
            self.ptr.CmdEndRenderPass(command_buffer);
        });
    }

    // cmd_execute_commands

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkCreateSwapchainKHR.html>
    ///
    /// # Safety
    /// `create_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn create_swapchain_khr(
        &self,
        device: vk::Device,
        create_info: &vk::SwapchainCreateInfoKHR,
    ) -> Result<vk::SwapchainKHR> {
        vk_query(|data| {
            self.ptr
                .CreateSwapchainKHR(device, create_info, ptr::null(), data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkDestroySwapchainKHR.html>
    #[inline]
    pub fn destroy_swapchain_khr(&self, device: vk::Device, swapchain: vk::SwapchainKHR) {
        vk_run(|| unsafe {
            self.ptr.DestroySwapchainKHR(device, swapchain, ptr::null());
        });
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkGetSwapchainImagesKHR.html>
    #[inline]
    pub fn get_swapchain_images_khr(
        &self,
        device: vk::Device,
        swapchain: vk::SwapchainKHR,
    ) -> Result<Vec<vk::Image>> {
        vk_query_many(|count, data| unsafe {
            self.ptr
                .GetSwapchainImagesKHR(device, swapchain, count, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkAcquireNextImageKHR.html>
    #[inline]
    pub fn acquire_next_image_khr(
        &self,
        device: vk::Device,
        swapchain: vk::SwapchainKHR,
        timeout: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<u32> {
        vk_query(|data| unsafe {
            self.ptr
                .AcquireNextImageKHR(device, swapchain, timeout, semaphore, fence, data)
        })
    }

    /// <https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/vkQueuePresentKHR.html>
    ///
    /// # Safety
    /// `present_info` should be valid regarding its containing pointers.
    ///
    #[inline]
    pub unsafe fn queue_present_khr(
        &self,
        queue: vk::Queue,
        present_info: &vk::PresentInfoKHR,
    ) -> Result<()> {
        vk_call(|| self.ptr.QueuePresentKHR(queue, present_info))
    }

    // create_shared_swapchains_khr

    // cmd_push_descriptor_set_khr

    // create_descriptor_update_template_khr

    // destroy_descriptor_update_template_khr

    // update_descriptor_set_with_template_khr

    // cmd_push_descriptor_set_with_template_khr

    // get_image_memory_requirements2_khr

    // get_buffer_memory_requirements2_khr

    // set_debug_utils_object_name_ext

    // cmd_begin_debug_utils_label_ext

    // set_debug_utils_object_name_ext

    // cmd_end_debug_utils_label_ext

    // cmd_insert_debug_utils_label_ext

    // acquire_full_screen_exclusive_mode_ext

    // release_full_screen_exclusive_mode_ext

    // get_buffer_device_address_ext
}

impl Deref for DevicePointers {
    type Target = vk::DevicePointers;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl<T: Into<vk::DevicePointers>> From<T> for DevicePointers {
    fn from(ptr: T) -> Self {
        DevicePointers::new(ptr.into())
    }
}

/// Some util structs and functions not considered for public use.
mod util {
    use super::Result;
    use std::mem;
    use vk_sys as vk;

    #[inline]
    pub fn vk_bool(b: bool) -> vk::Bool32 {
        if b {
            vk::TRUE
        } else {
            vk::FALSE
        }
    }

    #[inline]
    pub fn vk_slice_len<T>(slice: &[T]) -> u32 {
        let len = slice.len();
        debug_assert!(len <= u32::MAX as usize);
        len as u32
    }

    #[inline]
    pub fn vk_run<F: FnOnce()>(vk_api_call: F) {
        vk_api_call();
    }

    pub fn vk_call<F: FnOnce() -> vk::Result>(vk_api_call: F) -> Result<()> {
        let result = vk_api_call();

        if result == vk::SUCCESS {
            Ok(())
        } else {
            Err(result)
        }
    }

    pub fn vk_query_many<T, F: Fn(*mut u32, *mut T) -> vk::Result>(
        vk_api_call: F,
    ) -> Result<Vec<T>> {
        let mut count = mem::MaybeUninit::<u32>::uninit();
        let result = vk_api_call(count.as_mut_ptr(), std::ptr::null_mut());

        if result != vk::SUCCESS {
            return Err(result);
        }

        let len = unsafe { count.assume_init() } as usize;
        let mut data = Vec::<T>::with_capacity(len);
        let result = vk_api_call(count.as_mut_ptr(), data.as_mut_ptr());

        if result != vk::SUCCESS {
            return Err(result);
        }

        unsafe { data.set_len(len) };

        Ok(data)
    }

    pub fn vk_query_n<T, F: FnOnce(*mut T) -> vk::Result>(
        count: u32,
        vk_api_call: F,
    ) -> Result<Vec<T>> {
        let len = count as usize;
        let mut data = Vec::<T>::with_capacity(len);
        let result = vk_api_call(data.as_mut_ptr());

        if result != vk::SUCCESS {
            return Err(result);
        }

        unsafe { data.set_len(len) };

        Ok(data)
    }

    pub fn vk_get<T, F: FnOnce(*mut T)>(vk_api_call: F) -> T {
        let mut data = mem::MaybeUninit::<T>::uninit();
        vk_api_call(data.as_mut_ptr());

        unsafe { data.assume_init() }
    }

    pub fn vk_get_many<T, F: Fn(*mut u32, *mut T)>(vk_api_call: F) -> Vec<T> {
        let mut count = mem::MaybeUninit::<u32>::uninit();
        vk_api_call(count.as_mut_ptr(), std::ptr::null_mut());

        let len = unsafe { count.assume_init() } as usize;
        let mut data = Vec::<T>::with_capacity(len);

        vk_api_call(count.as_mut_ptr(), data.as_mut_ptr());
        unsafe { data.set_len(len) };

        data
    }

    pub fn vk_query<T, F: FnOnce(*mut T) -> vk::Result>(vk_api_call: F) -> Result<T> {
        let mut data = mem::MaybeUninit::<T>::uninit();
        let result = vk_api_call(data.as_mut_ptr());

        if result != vk::SUCCESS {
            return Err(result);
        }

        Ok(unsafe { data.assume_init() })
    }
}
