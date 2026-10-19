#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <vk_mem_alloc.h>
#define GLM_ENABLE_EXPERIMENTAL

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

// Один источник света. Для начала — направленный.
struct alignas(16) Light {
    alignas(16) glm::vec4 position_or_dir;  // .xyz — позиция (для point/spot), .w = тип
    alignas(16) glm::vec4 direction;        // .xyz — направление (для spot), .w = unused
    alignas(16) glm::vec4 color;            // .rgb — цвет * интенсивность
    alignas(16) glm::vec4 params;           // .x=radius, .y=inner_cos, .z=outer_cos
};

// Один источник света. Для начала — направленный.
struct alignas(16) SceneUBO {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::vec4 camera_pos;
    alignas(16) glm::vec4 ambient_color;
    alignas(16) glm::ivec4 light_counts;  // .x = dir, .y = point, .z = spot, .w = unused
};

// // UBO — общие параметры, одинаковые для всех объектов
// struct alignas(16) SceneUBO {
//     alignas(16) glm::mat4 view;
//     alignas(16) glm::mat4 proj;
//     alignas(16) glm::vec4 camera_pos;     // .xyz
//     alignas(16) glm::vec4 light_dir;      // .xyz — направленный свет
//     alignas(16) glm::vec4 light_color;    // .rgb
//     alignas(16) glm::vec4 ambient_color;  // .rgb
// };

struct alignas(16) ModelUBO {
    alignas(16) glm::mat4 model;
    alignas(16) glm::vec4 albedo;         // .rgb
    alignas(16) glm::vec4 specular_color; // .rgb
    alignas(16) glm::vec4 params;         // .x = shininess
};

struct DescriptorSet {
    VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;

    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkDescriptorSet       scene_set = VK_NULL_HANDLE;
    VkDescriptorSet       model_set = VK_NULL_HANDLE;

    static DescriptorSet create();           // создаёт оба layout'а + pool + 2 set'а
    void destroy();
    void updateScene(VkBuffer ubo, VkDeviceSize ubo_size,
                    VkBuffer ssbo, VkDeviceSize ssbo_size);
    void updateModel(VkBuffer ubo, VkDeviceSize ubo_size);
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
                       const std::vector<VkDescriptorSetLayout>& layouts);

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
    glm::vec3 normal;
};

// Параметры процедурного окрашивания вершин
struct ColorParams {
    glm::vec3 weight_r = {1.0f, 0.0f, 0.0f};
    glm::vec3 weight_g = {0.0f, 1.0f, 0.0f};
    glm::vec3 weight_b = {0.0f, 0.0f, 1.0f};
    glm::vec3 base     = {0.0f, 0.0f, 0.0f};
    float     gamma    = 1.0f;

    // 0 = линейный градиент, 1 = HSV по углу, 2 = шахматка
    int mode = 0;

    glm::vec3 tint = {1.0f, 1.0f, 1.0f};  // цвет от ColorEdit, умножается на процедурный
};

// Создаёт параллелепипед с размерами (sx, sy, sz), центрированный в начале координат.
// 6 граней * 4 вершины = 24 вершины, 36 индексов.
Mesh createBox(float sx, float sy, float sz, ColorParams& colors);

} // namespace graphics