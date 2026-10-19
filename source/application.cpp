#include "application.hpp"

#include <stdio.h>
#include <array>
#include <cmath>
#include <cstring>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "graphics.hpp"
#include "graphics_internal.hpp"
#include <iostream>

namespace application {

namespace {

using graphics::internal::context;

graphics::ColorParams g_last_colors;
int                   g_last_mode = -1;

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
    glm::vec3 camera_pos   = {0, 0, 3};
    float     camera_yaw   = -90.0f;   // градусы, вокруг Y
    float     camera_pitch = 0.0f;     // градусы, вверх/вниз
    float     camera_fov   = 60.0f;
    float     camera_near  = 0.1f;
    float     camera_far   = 100.0f;
    
    bool  anim_playing   = false;
    float anim_speed     = 1.0f;    // множитель скорости
    float anim_time      = 0.0f;    // накопленное время анимации (в секундах)

    // Траектория: «восьмёрка» (лемниската) вокруг начала координат
    float traj_radius    = 2.0f;    // основной радиус
    float traj_radius_y  = 1.0f;    // вертикальная амплитуда (для 3D-восьмёрки)
    float traj_speed     = 1.0f;    // базовая угловая скорость

    // Вращение
    float rot_speed_x    = 30.0f;   // градусов в секунду
    float rot_speed_y    = 45.0f;
    float rot_speed_z    = 0.0f;

    // Флаг: пользователь двигает вручную или анимация управляет
    bool  anim_drives_transform = true;

    // --- Процедурные цвета ---
    graphics::ColorParams color_params;
    int color_mode = 0;   // 0=Linear, 1=HSV, 2=Checker

    // свет
    // Направленный свет (оставим)
    glm::vec3 light_dir      = {-0.5f, -1.0f, -0.3f};
    glm::vec3 light_color    = {1.0f, 1.0f, 1.0f};
    float     light_intensity = 1.0f;

    // Точечные источники
    struct PointLight {
        glm::vec3 position = {2.0f, 2.0f, 2.0f};
        glm::vec3 color    = {1.0f, 0.5f, 0.5f};
        float intensity = 5.0f;
        float radius    = 5.0f;
    };
    std::vector<PointLight> point_lights;   

    struct SpotLight {
        glm::vec3 position    = {0, 5, 0};
        glm::vec3 direction   = {0, -1, 0};
        glm::vec3 color       = {1, 1, 1};
        float intensity       = 10.0f;
        float radius          = 10.0f;
        float inner_angle_deg = 15.0f;
        float outer_angle_deg = 30.0f;
    };
    std::vector<SpotLight> spot_lights;

    // материал
    glm::vec3 albedo         = {1.0f, 1.0f, 1.0f};
    glm::vec3 specular_color = {1.0f, 1.0f, 1.0f};
    float     shininess      = 32.0f;
    glm::vec3 ambient_color  = {0.1f, 0.1f, 0.1f};
};

BoxState                    g_state;
graphics::Pipeline          g_pipeline;
graphics::Mesh              g_mesh;
graphics::Buffer            g_ubo;
graphics::DescriptorSet     g_descriptor_set;
float                       g_last_size[3] = {1.0f, 0.7f, 1.3f};
graphics::Buffer            g_scene_ubo;
graphics::Buffer            g_model_ubo;
graphics::Buffer            g_light_ssbo;
static constexpr uint32_t   MAX_LIGHTS = 16;

// Пересобирает меш, если пользователь поменял размеры.
void rebuildMeshIfNeeded() {
    const bool size_changed =
        g_state.size_x != g_last_size[0] ||
        g_state.size_y != g_last_size[1] ||
        g_state.size_z != g_last_size[2];

    const bool color_changed =
        std::memcmp(&g_state.color_params, &g_last_colors,
                    sizeof(graphics::ColorParams)) != 0 ||
        g_state.color_mode != g_last_mode;

    if (!size_changed && !color_changed) return;

    g_mesh.destroy();
    g_mesh = graphics::createBox(g_state.size_x, g_state.size_y, g_state.size_z,
                                 g_state.color_params);

    g_last_size[0] = g_state.size_x;
    g_last_size[1] = g_state.size_y;
    g_last_size[2] = g_state.size_z;
    g_last_colors  = g_state.color_params;
    g_last_mode    = g_state.color_mode;
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

glm::vec3 cameraFront() {
    const float y = glm::radians(g_state.camera_yaw);
    const float p = glm::radians(g_state.camera_pitch);
    return glm::normalize(glm::vec3(
        cos(y) * cos(p),
        sin(p),
        sin(y) * cos(p)
    ));
}

glm::mat4 buildView() {
    return glm::lookAt(g_state.camera_pos,
                       g_state.camera_pos + cameraFront(),
                       {0, 1, 0});
}

glm::mat4 buildProjection() {
    const float aspect = float(context.swapchain_extent.width) /
                         float(context.swapchain_extent.height);
    return glm::perspective(glm::radians(g_state.camera_fov), aspect,
                            g_state.camera_near, g_state.camera_far);
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
        { .location = 2, .binding = 0, 
            .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(graphics::Vertex, normal) },
    };

    // --- Pipeline с push constant под mat4 (64 байта) ---
    // 1. Создаём UBO-буфер и descriptor set
    g_scene_ubo = graphics::Buffer::create(nullptr, sizeof(graphics::SceneUBO),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    g_model_ubo = graphics::Buffer::create(nullptr, sizeof(graphics::ModelUBO),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    g_light_ssbo = graphics::Buffer::create(nullptr, sizeof(graphics::Light) * MAX_LIGHTS,
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    g_descriptor_set = graphics::DescriptorSet::create();
    g_descriptor_set.updateScene(g_scene_ubo.buffer, sizeof(graphics::SceneUBO),
                                g_light_ssbo.buffer, sizeof(graphics::Light) * MAX_LIGHTS);
    g_descriptor_set.updateModel(g_model_ubo.buffer, sizeof(graphics::ModelUBO));

    g_state.point_lights.push_back({glm::vec3(2, 2, 2), glm::vec3(1, 0.5, 0.5), 5.0f, 5.0f});
    g_state.point_lights.push_back({glm::vec3(-2, 2, 0), glm::vec3(0.5, 0.5, 1), 5.0f, 5.0f});
    
    g_state.spot_lights.push_back({
        {0, 5, 0}, {0, -1, 0}, {1, 1, 1}, 10.0f, 10.0f, 15.0f, 30.0f
    });

    // Pipeline с двумя layout'ами
    const std::vector<VkDescriptorSetLayout> layouts = {
        g_descriptor_set.scene_layout,
        g_descriptor_set.model_layout,
    };
    g_pipeline = graphics::Pipeline::create(vert_spv, frag_spv,
                                            sizeof(graphics::Vertex),
                                            attributes, layouts);

    if (g_pipeline.pipeline == VK_NULL_HANDLE) {
        return false;
    }

    // --- Меш ---
    g_mesh = graphics::createBox(g_state.size_x, g_state.size_y, g_state.size_z, g_state.color_params);

    return true;
}

void shutdown() {
    auto& ctx = graphics::internal::context;
    vkQueueWaitIdle(ctx.graphics_queue);

    g_descriptor_set.destroy();
    g_scene_ubo.destroy();
    g_model_ubo.destroy();
    g_mesh.destroy();
    g_pipeline.destroy();
    g_light_ssbo.destroy();
}

void update([[maybe_unused]] double time) {
    // --- накопление времени анимации ---
    static double last_time = 0.0;
    const double dt = time - last_time;
    last_time = time;

    if (g_state.anim_playing) {
        g_state.anim_time += float(dt) * g_state.anim_speed;
    }

    if (g_state.anim_drives_transform) {
        const float t = g_state.anim_time * g_state.traj_speed;

        // Лемниската (восьмёрка) в плоскости XZ + колебание по Y
        const float r  = g_state.traj_radius;
        const float ry = g_state.traj_radius_y;

        g_state.position.x = r * std::sin(t);
        g_state.position.z = r * std::sin(t) * std::cos(t); // восьмёрка
        g_state.position.y = ry * std::sin(t * 2.0f);       // покачивание

        // Вращение
        g_state.rotation.x = g_state.rot_speed_x * g_state.anim_time;
        g_state.rotation.y = g_state.rot_speed_y * g_state.anim_time;
        g_state.rotation.z = g_state.rot_speed_z * g_state.anim_time;
    }

    ImGui::Begin("Lab 1 — Box");

    ImGui::SeparatorText("Box size");
    ImGui::SliderFloat("Size X", &g_state.size_x, 0.1f, 5.0f);
    ImGui::SliderFloat("Size Y", &g_state.size_y, 0.1f, 5.0f);
    ImGui::SliderFloat("Size Z", &g_state.size_z, 0.1f, 5.0f);

    // ImGui::SeparatorText("Transform");
    // ImGui::SliderFloat3("Position", &g_state.position.x, -5.0f, 5.0f);
    // ImGui::SliderFloat3("Rotation (deg)", &g_state.rotation.x, -180.0f, 180.0f);
    // ImGui::SliderFloat3("Scale", &g_state.scale.x, 0.1f, 5.0f);

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
    ImGui::SliderFloat3("Position", &g_state.camera_pos.x, -10.0f, 10.0f);
    ImGui::SliderFloat("Yaw",   &g_state.camera_yaw,   -180.0f, 180.0f);
    ImGui::SliderFloat("Pitch", &g_state.camera_pitch, -89.0f,  89.0f);
    ImGui::SliderFloat("FOV",   &g_state.camera_fov,   10.0f,  120.0f);
    ImGui::SliderFloat("Near",  &g_state.camera_near,  0.01f,  5.0f);
    ImGui::SliderFloat("Far",   &g_state.camera_far,   1.0f,   200.0f);

    ImGui::SeparatorText("Animation");
    ImGui::Checkbox("Play", &g_state.anim_playing);

    ImGui::SameLine();
    if (ImGui::Button("Reset time")) {
        g_state.anim_time = 0.0f;
    }

    ImGui::SliderFloat("Speed", &g_state.anim_speed, 0.0f, 5.0f);
    ImGui::Checkbox("Animation drives transform", &g_state.anim_drives_transform);

    ImGui::SeparatorText("Trajectory");
    ImGui::SliderFloat("Radius XZ", &g_state.traj_radius,   0.0f, 10.0f);
    ImGui::SliderFloat("Radius Y",  &g_state.traj_radius_y, 0.0f, 10.0f);
    ImGui::SliderFloat("Traj speed", &g_state.traj_speed,   0.0f, 5.0f);

    ImGui::SeparatorText("Rotation speed (deg/sec)");
    ImGui::SliderFloat("Rot X", &g_state.rot_speed_x, -360.0f, 360.0f);
    ImGui::SliderFloat("Rot Y", &g_state.rot_speed_y, -360.0f, 360.0f);
    ImGui::SliderFloat("Rot Z", &g_state.rot_speed_z, -360.0f, 360.0f);

    ImGui::SeparatorText("Light");
    ImGui::SliderFloat3("Dir", &g_state.light_dir.x, -1.0f, 1.0f);
    ImGui::ColorEdit3("Color", &g_state.light_color.x);
    ImGui::SliderFloat("Intensity", &g_state.light_intensity, 0.0f, 5.0f);

    ImGui::SeparatorText("Material");
    ImGui::ColorEdit3("Albedo", &g_state.albedo.x);
    ImGui::ColorEdit3("Specular", &g_state.specular_color.x);
    ImGui::SliderFloat("Shininess", &g_state.shininess, 1.0f, 256.0f);
    ImGui::ColorEdit3("Ambient", &g_state.ambient_color.x);

    ImGui::SeparatorText("Vertex color");

    ImGui::Combo("Mode", &g_state.color_mode,
                "Linear gradient\0HSV by angle\0Checkerboard\0");

    g_state.color_params.mode = g_state.color_mode;

    if (g_state.color_mode == 0) {
        ImGui::SliderFloat3("R weight (u,v,w)",
                            &g_state.color_params.weight_r.x, -2.0f, 2.0f);
        ImGui::SliderFloat3("G weight (u,v,w)",
                            &g_state.color_params.weight_g.x, -2.0f, 2.0f);
        ImGui::SliderFloat3("B weight (u,v,w)",
                            &g_state.color_params.weight_b.x, -2.0f, 2.0f);
        ImGui::SliderFloat3("Base", &g_state.color_params.base.x, 0.0f, 1.0f);
    }

    if (g_state.color_mode != 2) {
        ImGui::SliderFloat("Gamma", &g_state.color_params.gamma, 0.2f, 3.0f);
    }

    ImGui::ColorEdit3("Tint color", &g_state.color_params.tint.x);



    ImGui::SeparatorText("Point Lights");

    if (ImGui::Button("Add point light")) {
        g_state.point_lights.push_back({
            glm::vec3(2, 2, 2),
            glm::vec3(1, 1, 1),
            5.0f,
            5.0f,
        });
    }

    for (int i = 0; i < (int)g_state.point_lights.size(); ++i) {
        auto& p = g_state.point_lights[i];
        ImGui::PushID(i);

        ImGui::Text("Light #%d", i);
        ImGui::SliderFloat3("Position", &p.position.x, -10.0f, 10.0f);
        ImGui::ColorEdit3("Color", &p.color.x);
        ImGui::SliderFloat("Intensity", &p.intensity, 0.0f, 50.0f);
        ImGui::SliderFloat("Radius", &p.radius, 0.1f, 20.0f);

        if (ImGui::Button("Remove")) {
            g_state.point_lights.erase(g_state.point_lights.begin() + i);
            ImGui::PopID();
            break;
        }

        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::SeparatorText("Spot Lights");

    if (ImGui::Button("Add spot light")) {
        g_state.spot_lights.push_back({
            {0, 5, 0}, {0, -1, 0}, {1, 1, 1}, 10.0f, 10.0f, 15.0f, 30.0f
        });
    }

    for (int i = 0; i < (int)g_state.spot_lights.size(); ++i) {
        auto& s = g_state.spot_lights[i];
        ImGui::PushID(i + 1000);   // уникальный ID

        ImGui::Text("Spot #%d", i);
        ImGui::SliderFloat3("Position",  &s.position.x, -10.0f, 10.0f);
        ImGui::SliderFloat3("Direction", &s.direction.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", &s.color.x);
        ImGui::SliderFloat("Intensity", &s.intensity, 0.0f, 50.0f);
        ImGui::SliderFloat("Radius", &s.radius, 0.1f, 30.0f);
        ImGui::SliderFloat("Inner angle", &s.inner_angle_deg, 1.0f, 60.0f);
        ImGui::SliderFloat("Outer angle", &s.outer_angle_deg, 2.0f, 90.0f);

        // Гарантируем, что outer > inner
        if (s.outer_angle_deg <= s.inner_angle_deg) {
            s.outer_angle_deg = s.inner_angle_deg + 1.0f;
        }

        if (ImGui::Button("Remove")) {
            g_state.spot_lights.erase(g_state.spot_lights.begin() + i);
            ImGui::PopID();
            break;
        }

        ImGui::Separator();
        ImGui::PopID();
    }

    ImGui::Separator();
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

    graphics::Light lights[MAX_LIGHTS] = {};
    int idx = 0;

    // Направленные
    lights[idx].position_or_dir = glm::vec4(glm::normalize(g_state.light_dir), 0.0f);
    lights[idx].color           = glm::vec4(g_state.light_color * g_state.light_intensity, 1.0f);
    idx++;
    int dir_count = 1;

    // Точечные
    int point_count = std::min((int)g_state.point_lights.size(), (int)MAX_LIGHTS - idx);
    for (int i = 0; i < point_count; ++i) {
        const auto& p = g_state.point_lights[i];
        lights[idx + i].position_or_dir = glm::vec4(p.position, 1.0f);
        lights[idx + i].color           = glm::vec4(p.color * p.intensity, 1.0f);
        lights[idx + i].params          = glm::vec4(p.radius, 0, 0, 0);
    }
    idx += point_count;

    // Прожекторные
    int spot_count = std::min((int)g_state.spot_lights.size(), (int)MAX_LIGHTS - idx);
    for (int i = 0; i < spot_count; ++i) {
        const auto& s = g_state.spot_lights[i];
        lights[idx + i].position_or_dir = glm::vec4(s.position, 1.0f);
        lights[idx + i].color           = glm::vec4(s.color * s.intensity, 1.0f);
        lights[idx + i].params          = glm::vec4(
            s.radius,
            glm::cos(glm::radians(s.inner_angle_deg)),
            glm::cos(glm::radians(s.outer_angle_deg)),
            0.0f
        );
        lights[idx + i].direction = glm::vec4(glm::normalize(s.direction), 0.0f);
    }
    idx += spot_count;


    g_light_ssbo.upload(lights, sizeof(graphics::Light) * MAX_LIGHTS);

    // Scene UBO
    graphics::SceneUBO scene{};
    scene.view          = buildView();
    scene.proj          = buildProjection();
    scene.camera_pos    = glm::vec4(g_state.camera_pos, 1.0f);
    scene.ambient_color = glm::vec4(g_state.ambient_color, 1.0f);
    scene.light_counts = glm::ivec4(dir_count, point_count, spot_count, 0);
    g_scene_ubo.upload(&scene, sizeof(scene));

    // Model UBO — как было

    graphics::ModelUBO model{};
    model.model          = buildModel();
    model.albedo         = glm::vec4(g_state.albedo, 1.0f);
    model.specular_color = glm::vec4(g_state.specular_color, 1.0f);
    model.params         = glm::vec4(g_state.shininess, 0, 0, 0);
    g_model_ubo.upload(&model, sizeof(model));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, g_pipeline.pipeline);

    const VkDescriptorSet sets[] = { g_descriptor_set.scene_set, g_descriptor_set.model_set };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            g_pipeline.layout, 0, 2, sets, 0, nullptr);

    g_mesh.bind(cmd);
    g_mesh.draw(cmd);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

} // namespace application