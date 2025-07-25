import wgpu


class ApplyImpulsesPhase:
    apply_impulses_shader_source = """
        @group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read> impulses: array<vec2<f32>>;
        @group(0) @binding(2) var<storage, read_write> prev_positions: array<vec2<f32>>;

        fn warp_boundary(i: u32) {
            let p = positions[i];
            let prev_p = prev_positions[i];
            let l = length(p);
            let boundary = 3.0;
            if l > boundary {
                let new_p = -p + p / l * (l - boundary);
                positions[i] = new_p;
                prev_positions[i] = prev_p + new_p - p;
            }

        }

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            let prev_p = prev_positions[i];
            positions[i] = p + impulses[i];
            prev_positions[i] = prev_p;
            warp_boundary(i);
        }
    """

    def __init__(self, device):
        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(code=self.apply_impulses_shader_source)
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        impulses_buffer,
        command_encoder,
        number_of_bodies,
        prev_positions_buffer,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": impulses_buffer.buffer}},
                {"binding": 2, "resource": {"buffer": prev_positions_buffer.buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()
