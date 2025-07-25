from buffer import Buffer
import numpy as np
import wgpu


class TileSortPhase:
    tile_sort_shader_source = """
        struct TileSortUniform {
            tree_bb_min: vec2<f32>,
            tree_bb_max: vec2<f32>,
            separation: f32,
            number_of_bodies: u32,
            k: u32,
            j: u32,
        };
        
        @group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read_write> prev_positions: array<vec2<f32>>;
        @group(0) @binding(2) var<storage, read_write> tiles: array<u32>;
        @group(0) @binding(3) var<uniform> tile_sort_uniform: TileSortUniform;

        fn compare(left: u32, right: u32) -> bool {
            return tiles[left] < tiles[right];
        }

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let l = i ^ tile_sort_uniform.j;

            if l >= tile_sort_uniform.number_of_bodies {
                return;
            }

            if l > i {
                if ( ((i & tile_sort_uniform.k) == 0u) && (compare(l, i))) || ( ((i & tile_sort_uniform.k) != 0u) && (compare(i, l))) {
                    let temp_i = positions[i];
                    let temp_prev_i = prev_positions[i];
                    let temp_tile_i = tiles[i];
                    positions[i] = positions[l];
                    prev_positions[i] = prev_positions[l];
                    tiles[i] = tiles[l];
                    positions[l] = temp_i;
                    prev_positions[l] = temp_prev_i;
                    tiles[l] = temp_tile_i;
                }
            }
        }

    """

    def __init__(self, device, tree_bb_min, tree_bb_max):
        self.tree_bb_min = tree_bb_min
        self.tree_bb_max = tree_bb_max

        self.uniform_values = np.zeros(
            shape=(1,), dtype=np.dtype("2f4,2f4,f4,u4,u4,u4")
        )

        self.uniform_buffer = Buffer(
            device=device,
            shape=(1,),
            dtype=np.dtype("2f4,2f4,f4,u4,u4,u4"),
            usage=wgpu.BufferUsage.UNIFORM,
            staging=True,
        )

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(code=self.tile_sort_shader_source)
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
        prev_positions_buffer,
        tiles_buffer,
        command_encoder,
        number_of_bodies,
        separation,
        k,
        j,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer.buffer}},
                {"binding": 2, "resource": {"buffer": tiles_buffer.buffer}},
                {"binding": 3, "resource": {"buffer": self.uniform_buffer.buffer}},
            ],
        )
        self.uniform_values = self.uniform_values.copy()
        self.uniform_values[0][0] = self.tree_bb_min
        self.uniform_values[0][1] = self.tree_bb_max
        self.uniform_values[0][2] = separation
        self.uniform_values[0][3] = number_of_bodies
        self.uniform_values[0][4] = k
        self.uniform_values[0][5] = j

        self.uniform_buffer.write_staging(self.uniform_values)
        self.uniform_buffer.load_staging(command_encoder)

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()
