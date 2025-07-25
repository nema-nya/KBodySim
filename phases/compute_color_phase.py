import wgpu


class ComputeColorPhase:
    compute_color_shader_source = """
        override dt: f32;
        @group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read> prev_positions: array<vec2<f32>>;
        @group(0) @binding(2) var<storage, read_write> colors: array<vec4<f32>>;

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            let prev_p = prev_positions[i];
            
            let velocity = (p - prev_p) / dt;
            let len = length(velocity);

            if len < 1e-5 {
                colors[i] = vec4<f32>(1.0, 1.0, 1.0, 1.0);
            } else { 
                let direction = velocity / len;
                let scale = 100.0;
                let saturation = 1.0 - exp(-scale*len*len / 2.0);
                let hue = vec4<f32>(direction.x, direction.y, 0.0, 1.0) / 2.0 + 0.5;
                colors[i] = saturation * hue + (1.0 - saturation) * vec4<f32>(1.0, 1.0, 1.0, 1.0);
            }


        }
    """

    def __init__(self, device, dt):
        self.dt = dt
        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(code=self.compute_color_shader_source)
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
                "constants": {
                    "dt": self.dt,
                },
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        prev_positions_buffer,
        command_encoder,
        number_of_bodies,
        colors_buffer,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer.buffer}},
                {"binding": 2, "resource": {"buffer": colors_buffer.buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()
