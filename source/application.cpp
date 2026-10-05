#include "application.hpp"

#include <array>
#include <cmath>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "graphics.hpp"
#include "graphics_internal.hpp"

namespace application {

namespace {

using graphics::internal::context;

// ---------- Состояние лабы ----------
struct BoxState {
    float size_x = 1.0f;
    float size_y = 0.7f;
    float size_z = 1.3f;

    glm::vec3 position = {0.0f, 0.0f, 0.0f};
    glm::vec3 rotation = {0.0f, 0.0f, 0.0f}; // градусы, XYZ
    glm::vec3 scale    = {1.0f, 1.0f, 1.0f};

    // projection
    bool  use_perspective = false;
    float fov_deg  = 60.0f;
    float near_z   = 0.1f;
    float far_z    = 100.0f;
    float ortho_half_height = 2.0f;

    // camera
    glm::vec3 camera_pos = {0.0f, 0.0f, 3.0f};
    glm::vec3 camera_target = {0.0f, 0.0f, 0.0f};
};

BoxState  g_state;
graphics::Pipeline g_pipeline;
graphics::Mesh     g_mesh;
float     g_last_size[3] = {1.0f, 0.7f, 1.3f};

// Пересобирает меш, если пользователь поменял размеры.
void rebuildMeshIfNeeded() {
    if (g_state.size_x == g_last_size[0] &&
        g_state.size_y == g_last_size[1] &&
        g_state.size_z == g_last_size[2]) {
        return;
    }
    g_mesh.destroy();
    g_mesh = graphics::createBox(g_state.size_x, g_state.size_y, g_state.size_z);
    g_last_size[0] = g_state.size_x;
    g_last_size[1] = g_state.size_y;
    g_last_size[2] = g_state.size_z;
}

glm::mat4 buildModel() {
    glm::mat4 m = glm::mat4(1.0f);
    m = glm::translate(m, g_state.position);
    m = glm::rotate(m, glm::radians(g_state.rotation.x), {1, 0, 0});
    m = glm::rotate(m, glm::radians(g_state.rotation.y), {0, 1, 0});
    m = glm::rotate(m, glm::radians(g_state.rotation.z), {0, 0, 1});
    m = glm::scale(m, g_state.scale);
    return m;
}

glm::mat4 buildView() {
    return glm::lookAt(g_state.camera_pos, g_state.camera_target, {0, 1, 0});
}

glm::mat4 buildProjection() {
    const float aspect = float(context.swapchain_extent.width) /
                         float(context.swapchain_extent.height);
    if (g_state.use_perspective) {
        return glm::perspective(glm::radians(g_state.fov_deg), aspect,
                                g_state.near_z, g_state.far_z);
    } else {
        const float h = g_state.ortho_half_height;
        const float w = h * aspect;
        return glm::ortho(-w, w, -h, h, -100.f, 100.f);
    }
}

} // namespace

bool initialize() {
    // --- Шейдеры ---
    const auto vert_spv = graphics::loadShader("shaders/cube.vert.spv");
    const auto frag_spv = graphics::loadShader("shaders/cube.frag.spv");
    if (vert_spv.empty() || frag_spv.empty()) {
        return false;
    }

    // --- Описание атрибутов вершин ---
    const std::vector<VkVertexInputAttributeDescription> attributes = {
        { .location = 0, .binding = 0,
          .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(graphics::Vertex, position) },
        { .location = 1, .binding = 0,
          .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(graphics::Vertex, color) },
    };

    // --- Pipeline с push constant под mat4 (64 байта) ---
    g_pipeline = graphics::Pipeline::create(
        vert_spv, frag_spv,
        sizeof(graphics::Vertex),
        attributes,
        sizeof(glm::mat4));

    if (g_pipeline.pipeline == VK_NULL_HANDLE) {
        return false;
    }

    // --- Меш ---
    g_mesh = graphics::createBox(g_state.size_x, g_state.size_y, g_state.size_z);

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    g_mesh.destroy();
    g_pipeline.destroy();
}

void update([[maybe_unused]] double time) {
    ImGui::Begin("Lab 1 — Box");

    ImGui::SeparatorText("Box size");
    ImGui::SliderFloat("Size X", &g_state.size_x, 0.1f, 5.0f);
    ImGui::SliderFloat("Size Y", &g_state.size_y, 0.1f, 5.0f);
    ImGui::SliderFloat("Size Z", &g_state.size_z, 0.1f, 5.0f);

    ImGui::SeparatorText("Transform");
    ImGui::SliderFloat3("Position", &g_state.position.x, -5.0f, 5.0f);
    ImGui::SliderFloat3("Rotation (deg)", &g_state.rotation.x, -180.0f, 180.0f);
    ImGui::SliderFloat3("Scale", &g_state.scale.x, 0.1f, 5.0f);

    int* use_presp = (int*)(&g_state.use_perspective);
    ImGui::SeparatorText("Projection");
    ImGui::RadioButton("Perspective", use_presp, 1); ImGui::SameLine();
    ImGui::RadioButton("Orthographic", use_presp, 0);

    if (g_state.use_perspective) {
        ImGui::SliderFloat("FOV (deg)", &g_state.fov_deg, 10.0f, 120.0f);
        ImGui::SliderFloat("Near", &g_state.near_z, 0.01f, 5.0f);
        ImGui::SliderFloat("Far",  &g_state.far_z,  1.0f, 200.0f);
    } else {
        ImGui::SliderFloat("Ortho half-height", &g_state.ortho_half_height, 0.1f, 10.0f);
        ImGui::SliderFloat("Near", &g_state.near_z, -10.0f, 0.0f);
        ImGui::SliderFloat("Far",  &g_state.far_z,  0.1f, 200.0f);
    }

    ImGui::SeparatorText("Camera");
    ImGui::SliderFloat3("Camera pos", &g_state.camera_pos.x, -10.0f, 10.0f);
    ImGui::SliderFloat3("Camera target", &g_state.camera_target.x, -10.0f, 10.0f);

    ImGui::End();

    rebuildMeshIfNeeded();
}

void render(const graphics::internal::FrameData& fd) {
    VkCommandBuffer cmd = fd.command_buffer;

    vkResetCommandBuffer(cmd, 0);

    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmd, &begin);

    const VkClearValue clear_values[] = {
        { .color = {{0.05f, 0.05f, 0.08f, 1.0f}} },
        { .depthStencil = {1.0f, 0} },
    };

    const VkRenderPassBeginInfo rp = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

    const VkViewport viewport = {
        .x = 0.0f, .y = 0.0f,
        .width  = float(context.swapchain_extent.width),
        .height = float(context.swapchain_extent.height),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor = { .extent = context.swapchain_extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline.pipeline);

    const glm::mat4 mvp = buildProjection() * buildView() * buildModel();
    vkCmdPushConstants(cmd, g_pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(glm::mat4), &mvp);

    g_mesh.bind(cmd);
    g_mesh.draw(cmd);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application