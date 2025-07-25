from buffer import Buffer
import numpy as np
import wgpu


class ComputePointTilePhase:
    shader_source = """
        struct Uniforms {
            tree_bb_min: vec2<f32>,
            tree_bb_max: vec2<f32>,
            separation: f32,
        };
        
        @group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read_write> tiles: array<u32>;
        @group(0) @binding(2) var<uniform> uniforms: Uniforms;

        fn point_to_tile(point: vec2<f32>) -> u32 {
            let tile_x = u32(point.x / uniforms.separation); 
            let tile_y = u32(point.y / uniforms.separation);
            let max_tile_y = u32((uniforms.tree_bb_max.y - uniforms.tree_bb_min.y) / uniforms.separation);
            return tile_x * max_tile_y + tile_y;
        }

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            if uniforms.separation > 1e5 {
                return;
            }
            tiles[i] = point_to_tile(p);
        }

    """

    def __init__(self, device, number_of_points, tree_bb_min, tree_bb_max):
        self.number_of_points = number_of_points
        self.tree_bb_min = tree_bb_min
        self.tree_bb_max = tree_bb_max

        self.uniform_values = np.zeros(shape=(1,), dtype=np.dtype("2f4,2f4,f4,4V"))

        self.uniform_buffer = Buffer(
            device,
            (1,),
            dtype=np.dtype("2f4,2f4,f4,4V"),
            usage=wgpu.BufferUsage.UNIFORM,
            staging=True,
        )

        self.tiles_buffer = Buffer(
            device,
            (number_of_points,),
            dtype=np.dtype("u4"),
            usage=wgpu.BufferUsage.STORAGE,
            query=True,
        )

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(code=self.shader_source)
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
        command_encoder,
        separation,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": self.tiles_buffer.buffer}},
                {"binding": 2, "resource": {"buffer": self.uniform_buffer.buffer}},
            ],
        )

        self.uniform_values[0][0] = self.tree_bb_min
        self.uniform_values[0][1] = self.tree_bb_max
        self.uniform_values[0][2] = separation

        self.uniform_buffer.write_staging(self.uniform_values)

        self.uniform_buffer.load_staging(command_encoder)

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(self.number_of_points)
        compute_pass.end()
