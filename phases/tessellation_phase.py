import wgpu


class TessellationPhase:
    tessellation_compute_shader_source = """
        @group(0) @binding(0) var<storage, read>  positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read>  colors: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read_write> vertex_positions: array<vec2<f32>>;
        @group(0) @binding(3) var<storage, read_write> vertex_colors: array<vec4<f32>>;
        override point_resolution: u32;
        override point_radius: f32;
        @compute @workgroup_size(1, 1, 1) fn generate_verticies(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let pi = acos(-1.0);
            let pos = positions[i];
            let col = colors[i];
            for (var j = 0u; j < point_resolution; j++) {
                vertex_positions[i * point_resolution * 3u + j * 3u + 0u] = pos;
                vertex_positions[i * point_resolution * 3u + j * 3u + 1u] = pos + vec2<f32>(cos(f32(j) / f32(point_resolution) * 2 * pi), sin(f32(j) / f32(point_resolution) * 2 * pi)) * point_radius;
                vertex_positions[i * point_resolution * 3u + j * 3u + 2u] = pos + vec2<f32>(cos(f32(j + 1u) / f32(point_resolution) * 2 * pi), sin(f32(j + 1u) / f32(point_resolution) * 2 * pi)) * point_radius;
                vertex_colors[i * point_resolution * 3u + j * 3u + 0u] = col;
                vertex_colors[i * point_resolution * 3u + j * 3u + 1u] = col;
                vertex_colors[i * point_resolution * 3u + j * 3u + 2u] = col;
            }
        }
    """

    def __init__(self, number_of_points, point_resolution, point_radius, device):
        self.point_radius = point_radius
        self.number_of_points = number_of_points
        self.point_resolution = point_resolution

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

        self.vertex_colors_buffer = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.VERTEX,
        )

        self.vertex_colors_buffer_staging = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.STORAGE,
        )

        self.vertex_positions_buffer = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.VERTEX,
        )

        self.vertex_positions_buffer_staging = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.STORAGE,
        )

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(
            code=self.tessellation_compute_shader_source
        )
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "generate_verticies",
                "constants": {
                    "point_radius": self.point_radius,
                    "point_resolution": self.point_resolution,
                },
            },
        )

    def tesselation_pass(
        self,
        device,
        positions_buffer,
        colors_buffer,
        command_encoder,
    ):
        command_encoder.copy_buffer_to_buffer(
            source=self.vertex_positions_buffer_staging,
            source_offset=0,
            destination=self.vertex_positions_buffer,
            destination_offset=0,
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
        )

        command_encoder.copy_buffer_to_buffer(
            source=self.vertex_colors_buffer_staging,
            source_offset=0,
            destination=self.vertex_colors_buffer,
            destination_offset=0,
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
        )

        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": colors_buffer.buffer}},
                {
                    "binding": 2,
                    "resource": {"buffer": self.vertex_positions_buffer_staging},
                },
                {
                    "binding": 3,
                    "resource": {"buffer": self.vertex_colors_buffer_staging},
                },
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(self.number_of_points)
        compute_pass.end()
