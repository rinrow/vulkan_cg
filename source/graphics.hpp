#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>

namespace graphics {

// ---------- Буфер (vertex/index/uniform) через VMA ----------
struct Buffer {
    VkBuffer        buffer = VK_NULL_HANDLE;
    VmaAllocation   allocation = VK_NULL_HANDLE;
    VkDeviceSize    size = 0;

    // Создать буфер с заданными данными. usage — например
    // VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT.
    static Buffer create(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);

    // Уничтожить (вызвать из shutdown).
    void destroy();

    // Обновить содержимое (только для HOST_VISIBLE, напр. uniform).
    void upload(const void* data, VkDeviceSize size);
};

// ---------- Загрузка SPIR-V ----------
std::vector<uint32_t> loadShader(const std::string& path);
VkShaderModule createShaderModule(const std::vector<uint32_t>& code);
void destroyShaderModule(VkShaderModule module);

// ---------- Графический конвейер ----------
struct Pipeline {
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkPipelineLayout      layout   = VK_NULL_HANDLE;

    // Создать pipeline под context.render_pass.
    // vert_spv / frag_spv — байткод.
    // vertex_stride — размер одной вершины.
    // attribute_descriptions — описание атрибутов (position, color, ...).
    // push_constant_size — размер push constant блока (0 если не нужен).
    static Pipeline create(const std::vector<uint32_t>& vert_spv,
                           const std::vector<uint32_t>& frag_spv,
                           uint32_t vertex_stride,
                           const std::vector<VkVertexInputAttributeDescription>& attributes,
                           uint32_t push_constant_size);

    void destroy();
};

// ---------- Меш: вершины + индексы ----------
struct Mesh {
    Buffer vertex_buffer;
    Buffer index_buffer;
    uint32_t index_count = 0;

    static Mesh create(const void* vertices, VkDeviceSize vertex_bytes,
                       const void* indices,  VkDeviceSize index_bytes,
                       uint32_t index_count);

    void destroy();

    void bind(VkCommandBuffer cmd) const;
    void draw(VkCommandBuffer cmd) const;
};

// ---------- Параллелепипед ----------
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;
};

// Создаёт параллелепипед с размерами (sx, sy, sz), центрированный в начале координат.
// 6 граней * 4 вершины = 24 вершины, 36 индексов.
Mesh createBox(float sx, float sy, float sz);

} // namespace graphics