import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np

shader_source = """
    struct VertexInput {
        @location(0) pos : vec2<f32>,
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
        out.color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
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
        vertices,
        simluation,
        point_resolution,
        point_radius,
        number_of_points,
        title="wgpu renderer",
    ):
        self.vertices = vertices
        self.window_width = window_width
        self.window_height = window_height
        self.simulation = simluation
        self.point_resolution = point_resolution
        self.point_radius = point_radius
        self.number_of_points = number_of_points
        canvas = WgpuCanvas(
            size=(window_width, window_height), title=title, max_fps=1000, vsync=False
        )
        draw_frame = self.setup_drawing(canvas)
        canvas.request_draw(draw_frame)
        run()

    def setup_drawing(self, canvas, power_preference="high-performance"):
        adapter = wgpu.gpu.request_adapter_sync(power_preference=power_preference)
        device = adapter.request_device_sync(required_limits=None)
        context = canvas.get_context("wgpu")
        pipeline_kwargs = self.get_render_pipeline_kwargs(context, device)
        render_pipeline = device.create_render_pipeline(**pipeline_kwargs)

        vertex_buffer = device.create_buffer(
            size=len(self.vertices) * 2 * 4,
            usage=wgpu.BufferUsage.VERTEX + wgpu.BufferUsage.COPY_DST,
        )
        vertex_buffer_staging = device.create_buffer(
            size=len(self.vertices) * 2 * 4,
            usage=wgpu.BufferUsage.MAP_WRITE + wgpu.BufferUsage.COPY_SRC,
        )

        width_blocks = (self.window_width * 4 + 255) // 256
        color_output_buffer = device.create_buffer(
            size=width_blocks * 256 * self.window_height,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.MAP_READ,
        )

        color_texture = device.create_texture(
            format=context.get_preferred_format(device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            size={"width": self.window_width, "height": self.window_height},
            sample_count=4,
        )

        color_resolve_texture = device.create_texture(
            format=context.get_preferred_format(device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT + wgpu.TextureUsage.COPY_SRC,
            size={"width": self.window_width, "height": self.window_height},
            sample_count=1,
        )

        present_multisampled_texture = device.create_texture(
            format=context.get_preferred_format(device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            size={"width": self.window_width, "height": self.window_height},
            sample_count=4,
        )

        return self.get_draw_function(
            canvas,
            context,
            device,
            render_pipeline,
            vertex_buffer,
            vertex_buffer_staging,
            color_output_buffer,
            color_texture,
            color_resolve_texture,
            present_multisampled_texture,
        )

    def get_render_pipeline_kwargs(self, context, device):
        context.configure(device=device, format=None)
        render_texture_format = context.get_preferred_format(device.adapter)
        shader = device.create_shader_module(code=shader_source)
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

        return dict(
            layout=pipeline_layout,
            vertex={
                "buffers": [
                    {
                        "array_stride": 8,
                        "attributes": [
                            {
                                "format": wgpu.VertexFormat.float32x2,
                                "offset": 0,
                                "shader_location": 0,
                            }
                        ],
                        "step_mode": wgpu.VertexStepMode.vertex,
                    }
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

    def get_draw_function(
        self,
        canvas,
        context,
        device,
        render_pipeline,
        vertex_buffer,
        vertex_buffer_staging,
        color_output_buffer,
        color_texture,
        color_resolve_texture,
        present_multisampled_texture,
    ):
        def draw_frame():
            positions = self.simulation.start_frame()
            if positions is None:
                canvas.close()
                return

            angles = (
                np.arange(self.point_resolution + 1) / self.point_resolution * 2 * np.pi
            )
            a_offsets = np.zeros(shape=(self.point_resolution, 2), dtype=float)
            b_offsets = (
                np.stack([np.cos(angles[:-1]), np.sin(angles[:-1])], axis=-1)
                * self.point_radius
            )
            c_offsets = (
                np.stack([np.cos(angles[1:]), np.sin(angles[1:])], axis=-1)
                * self.point_radius
            )
            offsets = np.stack([a_offsets, b_offsets, c_offsets], axis=-2)
            self.vertices[:] = (
                positions.reshape((self.number_of_points, 1, 2))
                + offsets.reshape((1, self.point_resolution * 3, 2))
            ).reshape((self.number_of_points * self.point_resolution * 3, 2))
            vertex_buffer_staging.map_sync(wgpu.MapMode.WRITE)
            vertex_buffer_staging.write_mapped(self.vertices)
            vertex_buffer_staging.unmap()
            current_texture = context.get_current_texture()
            command_encoder = device.create_command_encoder()
            command_encoder.copy_buffer_to_buffer(
                source=vertex_buffer_staging,
                source_offset=0,
                destination=vertex_buffer,
                destination_offset=0,
                size=len(self.vertices) * 2 * 4,
            )
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": present_multisampled_texture.create_view(),
                        "resolve_target": current_texture.create_view(),
                        "clear_value": (0, 0, 0, 1),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    },
                    {
                        "view": color_texture.create_view(),
                        "resolve_target": color_resolve_texture.create_view(),
                        "clear_value": (0, 0, 0, 1),
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                    },
                ],
            )
            render_pass.set_pipeline(render_pipeline)
            render_pass.set_vertex_buffer(
                slot=0, buffer=vertex_buffer, offset=0, size=None
            )
            render_pass.draw(len(self.vertices), 1, 0, 0)
            render_pass.end()

            width_blocks = (self.window_width * 4 + 255) // 256
            command_encoder.copy_texture_to_buffer(
                {
                    "aspect": wgpu.TextureAspect.all,
                    "mip_level": 0,
                    "origin": {"x": 0, "y": 0, "z": 0},
                    "texture": color_resolve_texture,
                },
                {
                    "buffer": color_output_buffer,
                    "bytes_per_row": width_blocks * 256,
                    "offset": 0,
                },
                {"width": current_texture.width, "height": current_texture.height},
            )

            device.queue.submit([command_encoder.finish()])

            color_output_buffer.map_sync(wgpu.MapMode.READ)
            image_mem = color_output_buffer.read_mapped()
            color_output_buffer.unmap()
            padding = width_blocks * 256 - self.window_width * 4
            image_bytes_padded = np.frombuffer(
                image_mem.tobytes(), dtype=np.byte
            ).reshape((self.window_height, self.window_width + padding, 4))
            image_array = image_bytes_padded[:, : self.window_width, :]
            self.simulation.end_frame(image_array)
            canvas.request_draw(draw_frame)

        return draw_frame
