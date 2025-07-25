import wgpu


class ApplyAccelerationsPhase:
    apply_accelerations_shader_source = """
        override gravitational_constant: f32;
        override dt: f32;
        @group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read_write> prev_positions: array<vec2<f32>>;
        @group(0) @binding(2) var<storage, read> accelerations: array<vec2<f32>>;

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            let prev_p = prev_positions[i];
            let accel = accelerations[i];
            let new_p = 2.0 * p - prev_p + accel * dt * dt * gravitational_constant;
            prev_positions[i] = p;
            positions[i] = new_p;
        }
    """

    def __init__(self, device, gravitational_constant, dt):
        self.gravitational_constant = gravitational_constant
        self.dt = dt

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(
            code=self.apply_accelerations_shader_source
        )
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
                "constants": {
                    "gravitational_constant": self.gravitational_constant,
                    "dt": self.dt,
                },
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        prev_positions_buffer,
        acceleratios_buffer,
        command_encoder,
        number_of_bodies,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer.buffer}},
                {"binding": 2, "resource": {"buffer": acceleratios_buffer.buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()
