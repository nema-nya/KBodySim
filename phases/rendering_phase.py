from buffer import Buffer
import numpy as np
import wgpu


class RenderingPhase:
    rendering_shader_source = """
        struct VertexInput {
            @location(0) pos : vec2<f32>,
            @location(1) color : vec4<f32>,
            @builtin(vertex_index) vertex_index : u32,
        };

        struct VertexOutput {
            @location(0) color     : vec4<f32>,
            @builtin(position) pos : vec4<f32>,
        };

        @group(0) @binding(0) var<uniform> mvp: mat4x4<f32>;

        @vertex
        fn vs_main(in: VertexInput) -> VertexOutput {
            let index = i32(in.vertex_index);
            var out: VertexOutput;
            out.pos = mvp * vec4<f32>(in.pos, 0.0, 1.0);
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
            let physical_color = pow(in.color.rgb, vec3<f32>(2.2));
            out.present_color = vec4<f32>(physical_color, in.color.a);
            out.save_color = vec4<f32>(physical_color, in.color.a);
            return out;
        }
    """

    def __init__(self, device, context, window_width, window_height):
        self.mvp_buffer = Buffer(
            device=device,
            shape=(1, 4, 4),
            dtype=np.dtype("f4"),
            usage=wgpu.BufferUsage.UNIFORM,
            staging=True,
        )
        self.bind_group_layout = device.create_bind_group_layout(
            entries=[
                {"binding": 0, "visibility": wgpu.ShaderStage.VERTEX, "buffer": {}}
            ]
        )
        self.bind_group = device.create_bind_group(
            layout=self.bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.mvp_buffer.buffer,
                        "offset": 0,
                        "size": self.mvp_buffer.nbytes,
                    },
                }
            ],
        )
        pipeline_kwargs = self.get_pipeline_kwargs(device, context)
        self.pipeline = device.create_render_pipeline(**pipeline_kwargs)
        self.color_texture = device.create_texture(
            format=context.get_preferred_format(device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            size={"width": window_width, "height": window_height},
            sample_count=4,
        )

        self.color_resolve_texture = device.create_texture(
            format=context.get_preferred_format(device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT + wgpu.TextureUsage.COPY_SRC,
            size={"width": window_width, "height": window_height},
            sample_count=1,
        )

        self.present_multisampled_texture = device.create_texture(
            format=context.get_preferred_format(device.adapter),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            size={"width": window_width, "height": window_height},
            sample_count=4,
        )

    def render_pass(
        self,
        command_encoder,
        current_texture,
        vertex_positions_buffer,
        vertex_colors_buffer,
        number_of_points,
        point_resolution,
    ):

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": self.present_multisampled_texture.create_view(),
                    "resolve_target": current_texture.create_view(),
                    "clear_value": (
                        (22 / 255) ** 2.2,
                        (22 / 255) ** 2.2,
                        (29 / 255) ** 2.2,
                        1,
                    ),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                },
                {
                    "view": self.color_texture.create_view(),
                    "resolve_target": self.color_resolve_texture.create_view(),
                    "clear_value": (
                        (22 / 255) ** 2.2,
                        (22 / 255) ** 2.2,
                        (29 / 255) ** 2.2,
                        1,
                    ),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                },
            ],
        )
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(
            slot=0, buffer=vertex_positions_buffer, offset=0, size=None
        )
        render_pass.set_vertex_buffer(
            slot=1, buffer=vertex_colors_buffer, offset=0, size=None
        )
        render_pass.set_bind_group(0, self.bind_group)
        render_pass.draw(number_of_points * point_resolution * 3, 1, 0, 0)
        render_pass.end()

    def get_pipeline_kwargs(self, device, context):
        render_texture_format = context.get_preferred_format(device.adapter)
        shader = device.create_shader_module(code=self.rendering_shader_source)
        pipeline_layout = device.create_pipeline_layout(
            bind_group_layouts=[self.bind_group_layout]
        )

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
