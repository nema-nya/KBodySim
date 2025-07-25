import wgpu


class ComputeImpulsesPhase:
    compute_impulses_shader_source = """
        override separation: f32;
        override number_of_bodies: u32;
        @group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read_write> impulses: array<vec2<f32>>;

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            var impulse: vec2<f32> = vec2<f32>(0, 0);
            for (var j = 0u; j < number_of_bodies; j++) {
                let q = positions[j];
                let d = distance(q, p);
                if (d > 1e-5) && (d < 2 * separation) {
                    impulse = impulse + (p - q) / d * (2 * separation - d) / 2;
                }
            }
            impulses[i] = impulse;
        }
    """

    def __init__(self, number_of_bodies, device, separation):
        self.number_of_bodies = number_of_bodies
        self.separation = separation

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(code=self.compute_impulses_shader_source)
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
                "constants": {
                    "separation": self.separation,
                    "number_of_bodies": self.number_of_bodies,
                },
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        impulses_buffer,
        command_encoder,
        number_of_bodies,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": impulses_buffer.buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()
