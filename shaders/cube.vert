#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_normal;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 ambient_color;
    ivec4 light_counts;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular_color;
    vec4 params;
} model_ubo;

layout(location = 0) out vec3 frag_color;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_world_pos;

void main() {
    vec4 world = model_ubo.model * vec4(in_position, 1.0);
    gl_Position = scene.proj * scene.view * world;
    frag_color     = in_color;
    frag_normal    = mat3(transpose(inverse(model_ubo.model))) * in_normal;
    frag_world_pos = world.xyz;
}