#version 450

layout(location = 0) in vec3 frag_color;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_world_pos;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 ambient_color;
    ivec4 light_counts;
} scene;

struct Light {
    vec4 position_or_dir;
    vec4 direction;
    vec4 color;
    vec4 params;
};

layout(set = 0, binding = 1) readonly buffer LightBuffer {
    Light lights[16];
} light_buffer;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    vec4 albedo;
    vec4 specular_color;
    vec4 params;
} model_ubo;

void main() {
    vec3 N = normalize(frag_normal);
    vec3 V = normalize(scene.camera_pos.xyz - frag_world_pos);
    vec3 base = frag_color * model_ubo.albedo.rgb;

    vec3 result = scene.ambient_color.rgb * base;

    int dir_count   = scene.light_counts.x;
    int point_count = scene.light_counts.y;
    int spot_count  = scene.light_counts.z;

    int idx = 0;

    // Направленные
    for (int i = 0; i < dir_count; ++i, ++idx) {
        vec3 L = normalize(-light_buffer.lights[idx].position_or_dir.xyz);
        vec3 light_col = light_buffer.lights[idx].color.rgb;
        vec3 H = normalize(L + V);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model_ubo.params.x);
        result += light_col * (diff * base + spec * model_ubo.specular_color.rgb);
    }

    // Точечные
    for (int i = 0; i < point_count; ++i, ++idx) {
        vec3 light_pos = light_buffer.lights[idx].position_or_dir.xyz;
        vec3 light_col = light_buffer.lights[idx].color.rgb;
        float radius   = light_buffer.lights[idx].params.x;

        vec3 to_light = light_pos - frag_world_pos;
        float dist    = length(to_light);
        vec3 L        = to_light / dist;

        float atten = 1.0 / (1.0 + (dist * dist) / (radius * radius));

        vec3 H = normalize(L + V);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model_ubo.params.x);
        result += atten * light_col * (diff * base + spec * model_ubo.specular_color.rgb);
    }

    // Прожекторные
    for (int i = 0; i < spot_count; ++i, ++idx) {
        vec3 light_pos = light_buffer.lights[idx].position_or_dir.xyz;
        vec3 spot_dir  = normalize(light_buffer.lights[idx].direction.xyz);
        vec3 light_col = light_buffer.lights[idx].color.rgb;
        float radius   = light_buffer.lights[idx].params.x;
        float inner_cos = light_buffer.lights[idx].params.y;
        float outer_cos = light_buffer.lights[idx].params.z;

        vec3 to_light = light_pos - frag_world_pos;
        float dist = length(to_light);
        vec3 L = to_light / dist;

        // Мягкие края: smoothstep между внешним и внутренним углом
        float theta = dot(-L, spot_dir);   // cos угла между направлением на фрагмент и осью конуса
        float spot_factor = smoothstep(outer_cos, inner_cos, theta);

        float atten = 1.0 / (1.0 + (dist * dist) / (radius * radius));

        vec3 H = normalize(L + V);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model_ubo.params.x);

        result += atten * spot_factor * light_col *
                (diff * base + spec * model_ubo.specular_color.rgb);
    }

    out_color = vec4(result, 1.0);
}