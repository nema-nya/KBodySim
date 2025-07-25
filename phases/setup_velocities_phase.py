import wgpu


class SetupVelocitiesPhase:
    setup_velocities_shader_source = """
        override initial_spin: f32;
        override dt: f32;
        @group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read_write> prev_positions: array<vec2<f32>>;

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            let p_conj = vec2<f32>(-p.y, p.x);
            let prev_p = prev_positions[i];

            prev_positions[i] = p;
            positions[i] = p + p_conj * initial_spin * dt;
        }
    """

    def __init__(self, device, initial_spin, dt):
        self.initial_spin = initial_spin
        self.dt = dt

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(code=self.setup_velocities_shader_source)
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
                "constants": {
                    "initial_spin": self.initial_spin,
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
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer.buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()
