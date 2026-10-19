
#define GLM_ENABLE_EXPERIMENTAL
#include "graphics.hpp"
#include <glm/gtc/constants.hpp>

#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <array>

#include "graphics_internal.hpp"

namespace graphics {

using graphics::internal::context;

namespace {
glm::vec3 hsv2rgb(float h, float s, float v) {
    const float h6 = h * 6.0f;
    const int   i  = int(h6);
    const float f  = h6 - float(i);
    const float p  = v * (1.0f - s);
    const float q  = v * (1.0f - s * f);
    const float t  = v * (1.0f - s * (1.0f - f));

    switch (i % 6) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default: return {v, p, q};
    }
}
}

// ... evalColor идёт ниже ...

// ============================================================
// Buffer
// ============================================================

Buffer Buffer::create(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer b;

    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    // Хотим видеть память с CPU, чтобы сразу залить данные.
    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };

    VmaAllocationInfo info{};
    if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info,
                        &b.buffer, &b.allocation, &info) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan buffer\n";
        return {};
    }

    b.size = size;

    if (data != nullptr) {
        std::memcpy(info.pMappedData, data, size);
        vmaFlushAllocation(context.allocator, b.allocation, 0, size);
    }

    return b;
}

void Buffer::destroy() {
    if (buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(context.allocator, buffer, allocation);
        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
        size = 0;
    }
}

void Buffer::upload(const void* data, VkDeviceSize bytes) {
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(context.allocator, allocation, &info);
    std::memcpy(info.pMappedData, data, bytes);
    vmaFlushAllocation(context.allocator, allocation, 0, bytes);
}

// ============================================================
// SPIR-V
// ============================================================

std::vector<uint32_t> loadShader(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << path << '\n';
        return {};
    }

    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> code(size / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    return code;
}

VkShaderModule createShaderModule(const std::vector<uint32_t>& code) {
    if (code.empty()) return VK_NULL_HANDLE;

    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode = code.data(),
    };

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan shader module\n";
        return VK_NULL_HANDLE;
    }
    return module;
}

void destroyShaderModule(VkShaderModule module) {
    if (module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(context.device, module, nullptr);
    }
}

// ============================================================
// Pipeline
// ============================================================

Pipeline Pipeline::create(const std::vector<uint32_t>& vert_spv,
                          const std::vector<uint32_t>& frag_spv,
                          uint32_t vertex_stride,
                          const std::vector<VkVertexInputAttributeDescription>& attributes,
                          const std::vector<VkDescriptorSetLayout>& layouts) {
    Pipeline p;

    VkShaderModule vert_module = createShaderModule(vert_spv);
    VkShaderModule frag_module = createShaderModule(frag_spv);
    if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
        destroyShaderModule(vert_module);
        destroyShaderModule(frag_module);
        return p;
    }

    const VkPipelineShaderStageCreateInfo stages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_module,
            .pName = "main",
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_module,
            .pName = "main",
        },
    };

    const VkVertexInputBindingDescription binding = {
        .binding = 0,
        .stride = vertex_stride,
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = uint32_t(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    const VkPipelineViewportStateCreateInfo viewport_state = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    const VkPipelineRasterizationStateCreateInfo raster = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };

    const VkPipelineColorBlendAttachmentState color_attachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
    };

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamic_states,
    };

    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = uint32_t(layouts.size()),
        .pSetLayouts = layouts.data(),
    };

    if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &p.layout) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan pipeline layout\n";
        destroyShaderModule(vert_module);
        destroyShaderModule(frag_module);
        return {};
    }

    const VkGraphicsPipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth_stencil,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic,
        .layout = p.layout,
        .renderPass = context.render_pass,
        .subpass = 0,
    };

    if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
                                  &pipeline_info, nullptr, &p.pipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan graphics pipeline\n";
        vkDestroyPipelineLayout(context.device, p.layout, nullptr);
        p.layout = VK_NULL_HANDLE;
    }

    destroyShaderModule(vert_module);
    destroyShaderModule(frag_module);
    return p;
}

void Pipeline::destroy() {
    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }
    if (layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(context.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
}

// ============================================================
// Mesh
// ============================================================

Mesh Mesh::create(const void* vertices, VkDeviceSize vertex_bytes,
                  const void* indices, VkDeviceSize index_bytes,
                  uint32_t index_count) {
    Mesh m;
    m.vertex_buffer = Buffer::create(vertices, vertex_bytes,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m.index_buffer = Buffer::create(indices, index_bytes,
                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    m.index_count = index_count;
    return m;
}

void Mesh::destroy() {
    vertex_buffer.destroy();
    index_buffer.destroy();
    index_count = 0;
}

void Mesh::bind(VkCommandBuffer cmd) const {
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);
}

void Mesh::draw(VkCommandBuffer cmd) const {
    vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
}

namespace {
glm::vec3 evalColor(const glm::vec3& local_pos, const glm::vec3& half_size,
                    const ColorParams& p) {
    // Нормализуем локальную позицию в [0,1]
    const glm::vec3 uvw = {
        (local_pos.x + half_size.x) / (2.0f * half_size.x),
        (local_pos.y + half_size.y) / (2.0f * half_size.y),
        (local_pos.z + half_size.z) / (2.0f * half_size.z),
    };

    glm::vec3 c;

    if (p.mode == 0) {
        // Линейный градиент: канал = скалярное произведение (u,v,w) с весом
        c.r = glm::dot(uvw, p.weight_r) + p.base.r;
        c.g = glm::dot(uvw, p.weight_g) + p.base.g;
        c.b = glm::dot(uvw, p.weight_b) + p.base.b;
    } else if (p.mode == 1) {
        // HSV: угол вокруг оси Y → оттенок, Y → яркость
        const float angle = std::atan2(local_pos.z, local_pos.x);
        const float hue   = (angle + glm::pi<float>()) / (2.0f * glm::pi<float>());
        const float sat   = 0.9f;
        const float val   = 0.4f + 0.6f * uvw.y;
        c = hsv2rgb(hue, sat, val);
    } else {
        // Шахматка: чётность ячейки по u,v,w
        const int cell = int(std::floor(uvw.x * 4.0f)) +
                         int(std::floor(uvw.y * 4.0f)) +
                         int(std::floor(uvw.z * 4.0f));
        const float k = (cell & 1) ? 1.0f : 0.3f;
        c = glm::vec3(0.9f, 0.6f, 0.3f) * k;
    }

    c *= p.tint;

    // Clamp + гамма
    c = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
    if (p.gamma != 1.0f) {
        c = glm::pow(c, glm::vec3(1.0f / p.gamma));
    }
    return c;
}
}

// ============================================================
// Параллелепипед
// ============================================================

Mesh createBox(float sx, float sy, float sz, ColorParams& colors) {
    const float hx = sx * 0.5f;
    const float hy = sy * 0.5f;
    const float hz = sz * 0.5f;
    const glm::vec3 half = {hx, hy, hz};

    // Локальные позиции 24 вершин (по 4 на грань)
    const glm::vec3 local[24] = {
        // Front (+Z)
        {-hx, -hy,  hz}, { hx, -hy,  hz}, { hx,  hy,  hz}, {-hx,  hy,  hz},
        // Back (-Z)
        { hx, -hy, -hz}, {-hx, -hy, -hz}, {-hx,  hy, -hz}, { hx,  hy, -hz},
        // Top (+Y)
        {-hx,  hy,  hz}, { hx,  hy,  hz}, { hx,  hy, -hz}, {-hx,  hy, -hz},
        // Bottom (-Y)
        {-hx, -hy, -hz}, { hx, -hy, -hz}, { hx, -hy,  hz}, {-hx, -hy,  hz},
        // Right (+X)
        { hx, -hy,  hz}, { hx, -hy, -hz}, { hx,  hy, -hz}, { hx,  hy,  hz},
        // Left (-X)
        {-hx, -hy, -hz}, {-hx, -hy,  hz}, {-hx,  hy,  hz}, {-hx,  hy, -hz},
    };
    
    // Нормали по граням: front, back, top, bottom, right, left
    const glm::vec3 face_normals[6] = {
        { 0,  0,  1},   // front  (+Z)
        { 0,  0, -1},   // back   (-Z)
        { 0,  1,  0},   // top    (+Y)
        { 0, -1,  0},   // bottom (-Y)
        { 1,  0,  0},   // right  (+X)
        {-1,  0,  0},   // left   (-X)
    };

    Vertex verts[24];
    for (int face = 0; face < 6; ++face) {
        const glm::vec3 n = face_normals[face];
        for (int v = 0; v < 4; ++v) {
            const int i = face * 4 + v;
            verts[i].position = local[i];
            verts[i].color    = evalColor(local[i], half, colors);
            verts[i].normal   = n;
        }
    }

    const uint16_t idx[] = {
         0,  1,  2,   2,  3,  0,
         4,  5,  6,   6,  7,  4,
         8,  9, 10,  10, 11,  8,
        12, 13, 14,  14, 15, 12,
        16, 17, 18,  18, 19, 16,
        20, 21, 22,  22, 23, 20,
    };

    return Mesh::create(verts, sizeof(verts), idx, sizeof(idx),
                        uint32_t(sizeof(idx) / sizeof(idx[0])));
}
DescriptorSet DescriptorSet::create() {
    DescriptorSet d;

    // Scene: UBO на binding 0, SSBO на binding 1
    const VkDescriptorSetLayoutBinding scene_bindings[2] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = scene_bindings,
    };
    vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr, &d.scene_layout);

    // Model: UBO на binding 0
    const VkDescriptorSetLayoutBinding model_binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo model_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &model_binding,
    };
    vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr, &d.model_layout);

    // Pool: 2 UBO + 1 SSBO
    const VkDescriptorPoolSize pool_sizes[2] = {
        { .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 2 },
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    vkCreateDescriptorPool(context.device, &pool_info, nullptr, &d.pool);

    // Sets
    const VkDescriptorSetLayout layouts[2] = { d.scene_layout, d.model_layout };
    const VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = d.pool,
        .descriptorSetCount = 2,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet sets[2];
    vkAllocateDescriptorSets(context.device, &alloc_info, sets);
    d.scene_set = sets[0];
    d.model_set = sets[1];

    return d;
}
void DescriptorSet::updateScene(VkBuffer ubo, VkDeviceSize ubo_size,
                                VkBuffer ssbo, VkDeviceSize ssbo_size) {
    const VkDescriptorBufferInfo ubo_info = { ubo, 0, ubo_size };
    const VkDescriptorBufferInfo ssbo_info = { ssbo, 0, ssbo_size };

    const VkWriteDescriptorSet writes[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &ubo_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &ssbo_info,
        },
    };
    vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);
}

void DescriptorSet::updateModel(VkBuffer buffer, VkDeviceSize size) {
    const VkDescriptorBufferInfo info = { buffer, 0, size };
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = model_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &info,
    };
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

void DescriptorSet::destroy() {
    if (pool) { vkDestroyDescriptorPool(context.device, pool, nullptr); pool = VK_NULL_HANDLE; scene_set = VK_NULL_HANDLE; model_set = VK_NULL_HANDLE; }
    if (scene_layout) { vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr); scene_layout = VK_NULL_HANDLE; }
    if (model_layout) { vkDestroyDescriptorSetLayout(context.device, model_layout, nullptr); model_layout = VK_NULL_HANDLE; }
}

} // namespace graphics