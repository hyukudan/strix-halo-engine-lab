#include "backend.hpp"
#include "strix_vulkan_rms_shader.hpp"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <utility>

namespace strix::vulkan {
namespace {

using core::gpu_port::Capabilities;
using core::gpu_port::DeviceObservation;
using core::gpu_port::ReferenceBackend;
using core::gpu_port::RmsNormParameters;
using core::gpu_port::refuse;

void require(VkResult status, std::string_view action) {
    if (status != VK_SUCCESS)
        throw std::runtime_error("Vulkan: " + std::string(action) + " (VkResult " +
                                 std::to_string(static_cast<int>(status)) + ")");
}

template<class T>
T vk_struct(VkStructureType kind) {
    T result{};
    result.sType = kind;
    return result;
}

template<class T, class Enumerate>
std::vector<T> enumerate(Enumerate query) {
    for (;;) {
        std::uint32_t count = 0;
        require(query(&count, nullptr), "count enumeration");
        if (!count) return {};
        std::vector<T> items(count);
        const auto status = query(&count, items.data());
        if (status == VK_INCOMPLETE) continue;
        require(status, "read enumeration");
        items.resize(count);
        return items;
    }
}

struct Loader {
    void* handle = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    ~Loader() { if (handle) (void)dlclose(handle); }
};

struct Instance {
    Loader loader;
    VkInstance handle = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr get = nullptr;
    PFN_vkDestroyInstance vkDestroyInstance = nullptr;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = nullptr;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2 = nullptr;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = nullptr;
    PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2 = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = nullptr;
    PFN_vkCreateDevice vkCreateDevice = nullptr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = nullptr;

    template<class Proc>
    void load(Proc& function, const char* name) {
        function = reinterpret_cast<Proc>(get(handle, name));
        if (!function) refuse("Vulkan", name + std::string(" unavailable"));
    }
    ~Instance() { if (handle && vkDestroyInstance) vkDestroyInstance(handle, nullptr); }
};

#define STRIX_VK_DEVICE_PROCS(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkUnmapMemory) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCmdPipelineBarrier) X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkQueueSubmit) X(vkQueueWaitIdle)

struct Device {
    std::shared_ptr<Instance> instance;
    VkDevice handle = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkPhysicalDeviceMemoryProperties memory{};
    VkPhysicalDeviceLimits limits{};
#define STRIX_VK_DECLARE(name) PFN_##name name = nullptr;
    STRIX_VK_DEVICE_PROCS(STRIX_VK_DECLARE)
#undef STRIX_VK_DECLARE

    VkDescriptorSetLayout bindings = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    ~Device() {
        if (pipeline) vkDestroyPipeline(handle, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(handle, layout, nullptr);
        if (bindings) vkDestroyDescriptorSetLayout(handle, bindings, nullptr);
        if (handle && vkDestroyDevice) vkDestroyDevice(handle, nullptr);
    }
};

struct PhysicalDevice {
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory{};
    std::optional<std::uint32_t> queue_family;
};

constexpr VkMemoryPropertyFlags mapped_flags =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

std::optional<bool> cooperative_matrix(const Instance& instance, VkPhysicalDevice device) {
#if defined(VK_KHR_cooperative_matrix)
    const auto extensions = enumerate<VkExtensionProperties>([&](auto* count, VkExtensionProperties* items) {
        return instance.vkEnumerateDeviceExtensionProperties(device, nullptr, count, items);
    });
    for (const auto& extension : extensions) {
        if (std::strcmp(extension.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) != 0) continue;
        auto matrix = vk_struct<VkPhysicalDeviceCooperativeMatrixFeaturesKHR>(
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR);
        auto features = vk_struct<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        features.pNext = &matrix;
        instance.vkGetPhysicalDeviceFeatures2(device, &features);
        return matrix.cooperativeMatrix == VK_TRUE;
    }
    return false;
#else
    (void)instance;
    (void)device;
    return std::nullopt;
#endif
}

DeviceObservation observe(const Instance& instance, VkPhysicalDevice device, PhysicalDevice& info) {
    instance.vkGetPhysicalDeviceProperties(device, &info.properties);
    DeviceObservation observation{info.properties.deviceName, false, 0, {}};
    if (info.properties.apiVersion < VK_API_VERSION_1_1) return observation;
    auto subgroup = vk_struct<VkPhysicalDeviceSubgroupProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
    auto properties = vk_struct<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    properties.pNext = &subgroup;
    instance.vkGetPhysicalDeviceProperties2(device, &properties);
    if ((subgroup.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0)
        observation.subgroup_width = subgroup.subgroupSize;
    observation.cooperative_matrix = cooperative_matrix(instance, device);

    std::uint32_t count = 0;
    instance.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    if (!count) return observation;
    std::vector<VkQueueFamilyProperties> families(count);
    instance.vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            info.queue_family = i;
            break;
        }
    }
    if (!info.queue_family) return observation;
    instance.vkGetPhysicalDeviceMemoryProperties(device, &info.memory);
    for (std::uint32_t i = 0; i < info.memory.memoryTypeCount; ++i) {
        if ((info.memory.memoryTypes[i].propertyFlags & mapped_flags) == mapped_flags) {
            observation.compute = true;
            break;
        }
    }
    return observation;
}

void make_pipeline(Device& device) {
#if defined(STRIX_VULKAN_HAS_SPIRV)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (std::uint32_t i = 0; i < bindings.size(); ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    auto set_layout = vk_struct<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    set_layout.bindingCount = static_cast<std::uint32_t>(bindings.size());
    set_layout.pBindings = bindings.data();
    require(device.vkCreateDescriptorSetLayout(device.handle, &set_layout, nullptr, &device.bindings), "create bindings");

    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<std::uint32_t>(sizeof(RmsNormParameters))};
    auto layout = vk_struct<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    layout.setLayoutCount = 1;
    layout.pSetLayouts = &device.bindings;
    layout.pushConstantRangeCount = 1;
    layout.pPushConstantRanges = &push;
    require(device.vkCreatePipelineLayout(device.handle, &layout, nullptr, &device.layout), "create pipeline layout");

    auto shader_info = vk_struct<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    shader_info.codeSize = sizeof(rms_norm_spirv);
    shader_info.pCode = rms_norm_spirv;
    VkShaderModule shader = VK_NULL_HANDLE;
    require(device.vkCreateShaderModule(device.handle, &shader_info, nullptr, &shader), "create RMSNorm shader");
    auto pipeline_info = vk_struct<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    pipeline_info.stage = vk_struct<VkPipelineShaderStageCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = device.layout;
    const auto status = device.vkCreateComputePipelines(device.handle, VK_NULL_HANDLE, 1, &pipeline_info,
                                                        nullptr, &device.pipeline);
    device.vkDestroyShaderModule(device.handle, shader, nullptr);
    require(status, "create RMSNorm pipeline");
#else
    static_assert(sizeof(rms_norm_glsl) > 1);
    (void)device;
#endif
}

class Backend final : public ReferenceBackend {
    struct Buffer final : MappedBuffer {
        std::shared_ptr<Device> device;
        VkBuffer handle = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        ~Buffer() override {
            if (mapped) device->vkUnmapMemory(device->handle, memory);
            if (handle) device->vkDestroyBuffer(device->handle, handle, nullptr);
            if (memory) device->vkFreeMemory(device->handle, memory, nullptr);
        }
    };
    struct Dispatch {
        std::shared_ptr<Device> device;
        std::array<BoundBuffer, 3> buffers;
        VkCommandPool commands = VK_NULL_HANDLE;
        VkDescriptorPool descriptors = VK_NULL_HANDLE;
        ~Dispatch() {
            if (commands) device->vkDestroyCommandPool(device->handle, commands, nullptr);
            if (descriptors) device->vkDestroyDescriptorPool(device->handle, descriptors, nullptr);
        }
    };
public:
    explicit Backend(std::shared_ptr<Device> device) : device_(std::move(device)) { initialize_queue(); }
    ~Backend() override {
        settle();
        // Unknown completion must retain the submitted command and its resources.
        if (pending_) (void)pending_.release();
    }
    std::string_view backend_name() const noexcept override { return "vulkan"; }

private:
    std::shared_ptr<Device> device_;
    std::unique_ptr<Dispatch> pending_;

    void settle() noexcept {
        if (!pending_) return;
        const auto status = device_->vkQueueWaitIdle(device_->queue);
        if (status == VK_SUCCESS || status == VK_ERROR_DEVICE_LOST) pending_.reset();
    }
    void drain(session::detail::QueueState&) noexcept override { settle(); }

    std::shared_ptr<session::detail::BufferState> allocate_buffer(std::size_t bytes, std::size_t alignment) override {
        auto buffer = std::make_shared<Buffer>();
        buffer->device = device_;
        auto info = vk_struct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        info.size = static_cast<VkDeviceSize>(bytes);
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        require(device_->vkCreateBuffer(device_->handle, &info, nullptr, &buffer->handle), "create buffer");
        VkMemoryRequirements requirements{};
        device_->vkGetBufferMemoryRequirements(device_->handle, buffer->handle, &requirements);
        auto allocation = vk_struct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocation.allocationSize = requirements.size;
        for (std::uint32_t type = 0; type < device_->memory.memoryTypeCount; ++type) {
            if ((requirements.memoryTypeBits & (std::uint32_t{1} << type)) == 0 ||
                (device_->memory.memoryTypes[type].propertyFlags & mapped_flags) != mapped_flags) continue;
            allocation.memoryTypeIndex = type;
            require(device_->vkAllocateMemory(device_->handle, &allocation, nullptr, &buffer->memory), "allocate buffer memory");
            break;
        }
        if (!buffer->memory) refuse("Vulkan", "no coherent host-visible memory type for this buffer");
        require(device_->vkBindBufferMemory(device_->handle, buffer->handle, buffer->memory, 0), "bind buffer memory");
        void* mapped = nullptr;
        require(device_->vkMapMemory(device_->handle, buffer->memory, 0, VK_WHOLE_SIZE, 0, &mapped), "map buffer memory");
        buffer->mapped = static_cast<std::byte*>(mapped);
        if (!mapped || reinterpret_cast<std::uintptr_t>(mapped) % alignment != 0)
            refuse("Vulkan", "mapped buffer alignment unavailable");
        return buffer;
    }

    void check_rms_norm(const RmsNormParameters& params, std::size_t bytes) const override {
        if (!device_->pipeline) refuse("Vulkan", "RMSNorm SPIR-V unavailable; build with a GLSL compiler");
        if (params.rows > device_->limits.maxComputeWorkGroupCount[0] || bytes > device_->limits.maxStorageBufferRange)
            refuse("Vulkan", "RMSNorm exceeds device dispatch or buffer limits");
    }

    void launch_rms_norm(const RmsNormParameters& params, const BoundBuffer& input,
                         const BoundBuffer& scale, const BoundBuffer& output) override {
        auto& d = *device_;
        auto work = std::make_unique<Dispatch>();
        work->device = device_;
        work->buffers = {input, scale, output};

        auto command_pool = vk_struct<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        command_pool.queueFamilyIndex = d.queue_family;
        require(d.vkCreateCommandPool(d.handle, &command_pool, nullptr, &work->commands), "create command pool");
        auto command_alloc = vk_struct<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        command_alloc.commandPool = work->commands;
        command_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_alloc.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        require(d.vkAllocateCommandBuffers(d.handle, &command_alloc, &command), "allocate command buffer");

        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        auto descriptor_pool = vk_struct<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
        descriptor_pool.maxSets = 1;
        descriptor_pool.poolSizeCount = 1;
        descriptor_pool.pPoolSizes = &size;
        require(d.vkCreateDescriptorPool(d.handle, &descriptor_pool, nullptr, &work->descriptors), "create descriptor pool");
        auto descriptor_alloc = vk_struct<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
        descriptor_alloc.descriptorPool = work->descriptors;
        descriptor_alloc.descriptorSetCount = 1;
        descriptor_alloc.pSetLayouts = &d.bindings;
        VkDescriptorSet set = VK_NULL_HANDLE;
        require(d.vkAllocateDescriptorSets(d.handle, &descriptor_alloc, &set), "allocate descriptor set");
        std::array<VkDescriptorBufferInfo, 3> views{};
        std::array<VkWriteDescriptorSet, 3> writes{};
        for (std::uint32_t i = 0; i < views.size(); ++i) {
            const auto& bytes = work->buffers[i];
            views[i] = {static_cast<const Buffer&>(*bytes.state).handle, bytes.offset, bytes.bytes};
            writes[i] = vk_struct<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
            writes[i].dstSet = set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &views[i];
        }
        d.vkUpdateDescriptorSets(d.handle, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

        auto begin = vk_struct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        require(d.vkBeginCommandBuffer(command, &begin), "begin RMSNorm command");
        d.vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.pipeline);
        d.vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE, d.layout, 0, 1, &set, 0, nullptr);
        d.vkCmdPushConstants(command, d.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             static_cast<std::uint32_t>(sizeof(params)), &params);
        d.vkCmdDispatch(command, params.rows, 1, 1);
        auto barrier = vk_struct<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        d.vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                               0, 1, &barrier, 0, nullptr, 0, nullptr);
        require(d.vkEndCommandBuffer(command), "end RMSNorm command");
        auto submit = vk_struct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        require(d.vkQueueSubmit(d.queue, 1, &submit, VK_NULL_HANDLE), "submit RMSNorm command");
        pending_ = std::move(work);
        const auto status = d.vkQueueWaitIdle(d.queue);
        if (status == VK_SUCCESS || status == VK_ERROR_DEVICE_LOST) pending_.reset();
        require(status, "complete RMSNorm command");
    }
};

} // namespace

std::unique_ptr<core::DenseReference> open_native(Capabilities* caps) {
    auto instance = std::make_shared<Instance>();
    if (!instance->loader.handle) refuse("Vulkan", "loader unavailable (libvulkan.so.1)");
    instance->get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(instance->loader.handle, "vkGetInstanceProcAddr"));
    if (!instance->get) refuse("Vulkan", "loader has no vkGetInstanceProcAddr");
    PFN_vkEnumerateInstanceVersion version = nullptr;
    instance->load(version, "vkEnumerateInstanceVersion");
    std::uint32_t api_version = 0;
    require(version(&api_version), "query instance version");
    if (api_version < VK_API_VERSION_1_1) refuse("Vulkan", "loader requires Vulkan 1.1 or newer");
    PFN_vkCreateInstance create = nullptr;
    instance->load(create, "vkCreateInstance");
    auto application = vk_struct<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    application.apiVersion = VK_API_VERSION_1_1;
    auto create_info = vk_struct<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    create_info.pApplicationInfo = &application;
    const auto status = create(&create_info, nullptr, &instance->handle);
    if (status == VK_ERROR_INCOMPATIBLE_DRIVER) refuse("Vulkan", "no compatible driver for Vulkan 1.1");
    require(status, "create instance");
#define STRIX_VK_INSTANCE_LOAD(name) instance->load(instance->name, #name)
    STRIX_VK_INSTANCE_LOAD(vkDestroyInstance);
    STRIX_VK_INSTANCE_LOAD(vkEnumeratePhysicalDevices);
    STRIX_VK_INSTANCE_LOAD(vkGetPhysicalDeviceProperties);
    STRIX_VK_INSTANCE_LOAD(vkGetPhysicalDeviceProperties2);
    STRIX_VK_INSTANCE_LOAD(vkGetPhysicalDeviceQueueFamilyProperties);
    STRIX_VK_INSTANCE_LOAD(vkGetPhysicalDeviceFeatures2);
    STRIX_VK_INSTANCE_LOAD(vkEnumerateDeviceExtensionProperties);
    STRIX_VK_INSTANCE_LOAD(vkGetPhysicalDeviceMemoryProperties);
    STRIX_VK_INSTANCE_LOAD(vkCreateDevice);
    STRIX_VK_INSTANCE_LOAD(vkGetDeviceProcAddr);
#undef STRIX_VK_INSTANCE_LOAD

    const auto devices = enumerate<VkPhysicalDevice>([&](auto* count, VkPhysicalDevice* items) {
        return instance->vkEnumeratePhysicalDevices(instance->handle, count, items);
    });
    std::vector<PhysicalDevice> physical(devices.size());
    std::vector<DeviceObservation> observations;
    observations.reserve(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) observations.push_back(observe(*instance, devices[i], physical[i]));
    auto selected = core::gpu_port::select_compute_device("Vulkan", observations);
    const auto& info = physical[selected.ordinal];

    auto device = std::make_shared<Device>();
    device->instance = instance;
    device->queue_family = *info.queue_family;
    device->memory = info.memory;
    device->limits = info.properties.limits;
    const float priority = 1.0F;
    auto queue = vk_struct<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queue.queueFamilyIndex = device->queue_family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    auto logical = vk_struct<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    logical.queueCreateInfoCount = 1;
    logical.pQueueCreateInfos = &queue;
    require(instance->vkCreateDevice(devices[selected.ordinal], &logical, nullptr, &device->handle), "create compute device");
#define STRIX_VK_DEVICE_LOAD(name) \
    device->name = reinterpret_cast<PFN_##name>(instance->vkGetDeviceProcAddr(device->handle, #name)); \
    if (!device->name) refuse("Vulkan", #name " unavailable");
    STRIX_VK_DEVICE_PROCS(STRIX_VK_DEVICE_LOAD)
#undef STRIX_VK_DEVICE_LOAD
#undef STRIX_VK_DEVICE_PROCS
    device->vkGetDeviceQueue(device->handle, device->queue_family, 0, &device->queue);
    make_pipeline(*device);
    auto backend = std::make_unique<Backend>(std::move(device));
    if (caps) *caps = std::move(selected);
    return backend;
}

} // namespace strix::vulkan
