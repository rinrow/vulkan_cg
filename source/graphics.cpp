#include "graphics.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "graphics_internal.hpp"

namespace graphics {

using graphics::internal::context;

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
                          uint32_t push_constant_size) {
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

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset = 0,
        .size = push_constant_size,
    };

    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = push_constant_size > 0 ? 1u : 0u,
        .pPushConstantRanges = push_constant_size > 0 ? &push_range : nullptr,
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

// ============================================================
// Параллелепипед
// ============================================================

Mesh createBox(float sx, float sy, float sz) {
    const float hx = sx * 0.5f;
    const float hy = sy * 0.5f;
    const float hz = sz * 0.5f;

    // Цвета граней (для наглядности разные).
    const glm::vec3 c_front  = {0.90f, 0.30f, 0.30f}; // +Z  красная
    const glm::vec3 c_back   = {0.30f, 0.90f, 0.30f}; // -Z  зелёная
    const glm::vec3 c_top    = {0.30f, 0.50f, 0.95f}; // +Y  синяя
    const glm::vec3 c_bottom = {0.95f, 0.85f, 0.25f}; // -Y  жёлтая
    const glm::vec3 c_right  = {0.85f, 0.35f, 0.90f}; // +X  фиолетовая
    const glm::vec3 c_left   = {0.25f, 0.85f, 0.90f}; // -X  голубая

    // 24 вершины (по 4 на грань). Порядок: CCW при взгляде снаружи.
    const Vertex verts[] = {
        // Front (+Z)
        {{-hx, -hy,  hz}, c_front}, {{ hx, -hy,  hz}, c_front},
        {{ hx,  hy,  hz}, c_front}, {{-hx,  hy,  hz}, c_front},
        // Back (-Z)
        {{ hx, -hy, -hz}, c_back},  {{-hx, -hy, -hz}, c_back},
        {{-hx,  hy, -hz}, c_back},  {{ hx,  hy, -hz}, c_back},
        // Top (+Y)
        {{-hx,  hy,  hz}, c_top},   {{ hx,  hy,  hz}, c_top},
        {{ hx,  hy, -hz}, c_top},   {{-hx,  hy, -hz}, c_top},
        // Bottom (-Y)
        {{-hx, -hy, -hz}, c_bottom},{{ hx, -hy, -hz}, c_bottom},
        {{ hx, -hy,  hz}, c_bottom},{{-hx, -hy,  hz}, c_bottom},
        // Right (+X)
        {{ hx, -hy,  hz}, c_right}, {{ hx, -hy, -hz}, c_right},
        {{ hx,  hy, -hz}, c_right}, {{ hx,  hy,  hz}, c_right},
        // Left (-X)
        {{-hx, -hy, -hz}, c_left},  {{-hx, -hy,  hz}, c_left},
        {{-hx,  hy,  hz}, c_left},  {{-hx,  hy, -hz}, c_left},
    };

    const uint16_t idx[] = {
         0,  1,  2,   2,  3,  0, // front
         4,  5,  6,   6,  7,  4, // back
         8,  9, 10,  10, 11,  8, // top
        12, 13, 14,  14, 15, 12, // bottom
        16, 17, 18,  18, 19, 16, // right
        20, 21, 22,  22, 23, 20, // left
    };

    return Mesh::create(verts, sizeof(verts), idx, sizeof(idx),
                        uint32_t(sizeof(idx) / sizeof(idx[0])));
}

} // namespace graphics