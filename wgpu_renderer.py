import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np

compute_shader_source = """
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

shader_source = """
    struct VertexInput {
        @location(0) pos : vec2<f32>,
        @location(1) color : vec4<f32>,
        @builtin(vertex_index) vertex_index : u32,
    };
    struct VertexOutput {
        @location(0) color     : vec4<f32>,
        @builtin(position) pos : vec4<f32>,
    };

    @vertex
    fn vs_main(in: VertexInput) -> VertexOutput {
        let index = i32(in.vertex_index);
        var out: VertexOutput;
        out.pos = vec4<f32>(in.pos, 0.0, 1.0);
        out.color = in.color;
        return out;
    }

    struct FragmentOutput {
        @location(0) present_color : vec4<f32>,   
        @location(1) save_color    : vec4<f32>,
    };

    @fragment
    fn fs_main(in: VertexOutput) -> FragmentOutput {
        var out: FragmentOutput;
        let physical_color = pow(in.color.rgb, vec3<f32>(2.2));  // gamma correct
        out.present_color = vec4<f32>(physical_color, in.color.a);
        out.save_color = vec4<f32>(physical_color, in.color.a);
        return out;
    }
    """


class WgpuRenderer:

    def __init__(
        self,
        window_width,
        window_height,
        simluation,
        point_resolution,
        point_radius,
        number_of_points,
        title="wgpu renderer",
    ):
        self.window_width = window_width
        self.window_height = window_height
        self.simulation = simluation
        self.point_resolution = point_resolution
        self.point_radius = point_radius
        self.number_of_points = number_of_points
        self.canvas = WgpuCanvas(
            size=(window_width, window_height), title=title, max_fps=1000, vsync=False
        )
        draw_frame = self.setup_drawing()
        self.canvas.request_draw(draw_frame)
        run()

    def setup_drawing(self, power_preference="high-performance"):
        adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        self.device = adapter.request_device_sync(required_limits=None)
        self.context = self.canvas.get_context("wgpu")
        self.context.configure(device=self.device, format=None)
        compute_pipeline_kwargs = self.get_compute_pipeline_kwargs()
        pipeline_kwargs = self.get_render_pipeline_kwargs()
        self.compute_pipeline = self.device.create_compute_pipeline(
            **compute_pipeline_kwargs
        )
        self.render_pipeline = self.device.create_render_pipeline(**pipeline_kwargs)

        self.position_buffer = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.STORAGE,
        )

        self.color_buffer = self.device.create_buffer(
            size=self.number_of_points * 4 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.STORAGE,
        )

        self.position_buffer_staging = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.MAP_WRITE + wgpu.BufferUsage.COPY_SRC,
        )

        self.color_buffer_staging = self.device.create_buffer(
            size=self.number_of_points * 4 * 4,
            usage=wgpu.BufferUsage.MAP_WRITE + wgpu.BufferUsage.COPY_SRC,
        )

        self.vertex_color_buffer_staging = self.device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.STORAGE,
        )

        self.vertex_position_buffer = self.device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.VERTEX,
        )

        self.vertex_color_buffer = self.device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.VERTEX,
        )

        self.vertex_position_buffer_staging = self.device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.STORAGE,
        )

        width_blocks = (self.window_width * 4 + 255) // 256
        self.color_output_buffer = self.device.create_buffer(
            size=width_blocks * 256 * self.window_height,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.MAP_READ,
        )

        self.color_texture = self.device.create_texture(
            format=self.context.get_preferred_format(self.device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            size={"width": self.window_width, "height": self.window_height},
            sample_count=4,
        )

        self.color_resolve_texture = self.device.create_texture(
            format=self.context.get_preferred_format(self.device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT + wgpu.TextureUsage.COPY_SRC,
            size={"width": self.window_width, "height": self.window_height},
            sample_count=1,
        )

        self.present_multisampled_texture = self.device.create_texture(
            format=self.context.get_preferred_format(self.device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            size={"width": self.window_width, "height": self.window_height},
            sample_count=4,
        )

        return self.get_draw_function()

    def get_compute_pipeline_kwargs(self):
        shader = self.device.create_shader_module(code=compute_shader_source)
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

    def get_render_pipeline_kwargs(self):
        render_texture_format = self.context.get_preferred_format(self.device.adapter)
        shader = self.device.create_shader_module(code=shader_source)
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[])

        return dict(
            layout=pipeline_layout,
            vertex={
                "buffers": [
                    {
                        "array_stride": 2 * 4,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x2,
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                        "step_mode": wgpu.VertexStepMode.vertex,
                    },
                    {
                        "array_stride": 4 * 4,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x4,
                                "offset": 0,
                                "shader_location": 1,
                            }
                        ],
                        "step_mode": wgpu.VertexStepMode.vertex,
                    },
                ],
                "module": shader,
                "entry_point": "vs_main",
            },
            depth_stencil=None,
            multisample={
                "count": 4,
            },
            fragment={
                "module": shader,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": render_texture_format,
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                    {
                        "format": render_texture_format,
                        "blend": {
                            "color": {},
                            "alpha": {},
                        },
                    },
                ],
            },
        )

    def get_draw_function(self):
        def draw_frame():
            positions_colors = self.simulation.start_frame()
            if positions_colors is None:
                self.canvas.close()
                return
            positions, colors = positions_colors

            current_texture = self.context.get_current_texture()

            command_encoder = self.device.create_command_encoder()

            self.position_buffer_staging.map_sync(wgpu.MapMode.WRITE)
            self.color_buffer_staging.map_sync(wgpu.MapMode.WRITE)
            self.position_buffer_staging.write_mapped(positions)
            self.color_buffer_staging.write_mapped(colors)
            self.position_buffer_staging.unmap()
            self.color_buffer_staging.unmap()

            command_encoder.copy_buffer_to_buffer(
                source=self.position_buffer_staging,
                source_offset=0,
                destination=self.position_buffer,
                destination_offset=0,
                size=self.number_of_points * 2 * 4,
            )

            command_encoder.copy_buffer_to_buffer(
                source=self.color_buffer_staging,
                source_offset=0,
                destination=self.color_buffer,
                destination_offset=0,
                size=self.number_of_points * 4 * 4,
            )

            bind_group = self.device.create_bind_group(
                layout=self.compute_pipeline.get_bind_group_layout(0),
                entries=[
                    {"binding": 0, "resource": {"buffer": self.position_buffer}},
                    {"binding": 1, "resource": {"buffer": self.color_buffer}},
                    {
                        "binding": 2,
                        "resource": {"buffer": self.vertex_position_buffer_staging},
                    },
                    {
                        "binding": 3,
                        "resource": {"buffer": self.vertex_color_buffer_staging},
                    },
                ],
            )

            compute_pass = command_encoder.begin_compute_pass()
            compute_pass.set_pipeline(self.compute_pipeline)
            compute_pass.set_bind_group(0, bind_group)
            compute_pass.dispatch_workgroups(self.number_of_points)
            compute_pass.end()

            command_encoder.copy_buffer_to_buffer(
                source=self.vertex_position_buffer_staging,
                source_offset=0,
                destination=self.vertex_position_buffer,
                destination_offset=0,
                size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
            )

            command_encoder.copy_buffer_to_buffer(
                source=self.vertex_color_buffer_staging,
                source_offset=0,
                destination=self.vertex_color_buffer,
                destination_offset=0,
                size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
            )

            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.present_multisampled_texture.create_view(),
                        "resolve_target": current_texture.create_view(),
                        "clear_value": (0, 0, 0, 1),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    },
                    {
                        "view": self.color_texture.create_view(),
                        "resolve_target": self.color_resolve_texture.create_view(),
                        "clear_value": (0, 0, 0, 1),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    },
                ],
            )
            render_pass.set_pipeline(self.render_pipeline)
            render_pass.set_vertex_buffer(
                slot=0, buffer=self.vertex_position_buffer, offset=0, size=None
            )
            render_pass.set_vertex_buffer(
                slot=1, buffer=self.vertex_color_buffer, offset=0, size=None
            )
            render_pass.draw(self.number_of_points * self.point_resolution * 3, 1, 0, 0)
            render_pass.end()

            width_blocks = (self.window_width * 4 + 255) // 256
            command_encoder.copy_texture_to_buffer(
                {
                    "aspect": wgpu.TextureAspect.all,
                    "mip_level": 0,
                    "origin": {"x": 0, "y": 0, "z": 0},
                    "texture": self.color_resolve_texture,
                },
                {
                    "buffer": self.color_output_buffer,
                    "bytes_per_row": width_blocks * 256,
                    "offset": 0,
                },
                {"width": current_texture.width, "height": current_texture.height},
            )

            self.device.queue.submit([command_encoder.finish()])

            self.color_output_buffer.map_sync(wgpu.MapMode.READ)
            image_mem = self.color_output_buffer.read_mapped()
            self.color_output_buffer.unmap()
            padding = width_blocks * 256 - self.window_width * 4
            image_bytes_padded = np.frombuffer(
                image_mem.tobytes(), dtype=np.byte
            ).reshape((self.window_height, self.window_width + padding, 4))
            image_array = image_bytes_padded[:, : self.window_width, :]
            self.simulation.end_frame(image_array)
            self.canvas.request_draw(draw_frame)

        return draw_frame
