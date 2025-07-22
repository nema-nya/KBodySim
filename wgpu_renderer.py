import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np


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
            tiles[i] = point_to_tile(p);
        }

    """

    def __init__(self, device, number_of_points):
        self.number_of_points = number_of_points
        self.uniform_values = np.zeros(shape=(1,), dtype=np.dtype("2f4,2f4,f4,4V"))

        self.uniform_buffer = device.create_buffer(
            size=self.uniform_values.nbytes,
            usage=wgpu.BufferUsage.UNIFORM + wgpu.BufferUsage.COPY_DST,
        )

        self.uniform_buffer_staging = device.create_buffer(
            size=self.uniform_values.nbytes,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.MAP_WRITE,
        )

        self.tiles_buffer = device.create_buffer(
            size=number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.STORAGE,
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
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": self.tiles_buffer}},
                {"binding": 2, "resource": {"buffer": self.uniform_buffer}},
            ],
        )

        self.uniform_values[0][0][0] = -2.0
        self.uniform_values[0][0][1] = -2.0
        self.uniform_values[0][1][0] = 2.0
        self.uniform_values[0][1][1] = 2.0
        self.uniform_values[0][2] = separation

        self.uniform_buffer_staging.map_sync(wgpu.MapMode.WRITE)
        self.uniform_buffer_staging.write_mapped(self.uniform_values)
        self.uniform_buffer_staging.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.uniform_buffer_staging,
            source_offset=0,
            destination=self.uniform_buffer,
            destination_offset=0,
            size=self.uniform_values.nbytes,
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(self.number_of_points)
        compute_pass.end()


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
            let p = positions[i];
            let l = i ^ tile_sort_uniform.j;
            if l >= tile_sort_uniform.number_of_bodies {
                return;
            }
            if l > i {
                if ( ((i & tile_sort_uniform.k) == 0) && (compare(l, i))) || ( ((i & tile_sort_uniform.k) != 0) && (compare(i, l))) {
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

    def __init__(self, device):
        self.tile_sort_uniform_values = np.zeros(
            shape=(1,), dtype=np.dtype("2f4,2f4,f4,u4,u4,u4")
        )
        self.tile_sort_uniform_buffer = device.create_buffer(
            size=self.tile_sort_uniform_values.nbytes,
            usage=wgpu.BufferUsage.UNIFORM + wgpu.BufferUsage.COPY_DST,
        )

        self.tile_sort_uniform_buffer_staging = device.create_buffer(
            size=self.tile_sort_uniform_values.nbytes,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.MAP_WRITE,
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
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer}},
                {"binding": 2, "resource": {"buffer": tiles_buffer}},
                {"binding": 3, "resource": {"buffer": self.tile_sort_uniform_buffer}},
            ],
        )

        self.tile_sort_uniform_values[0][0][0] = -2.0
        self.tile_sort_uniform_values[0][0][1] = -2.0
        self.tile_sort_uniform_values[0][1][0] = 2.0
        self.tile_sort_uniform_values[0][1][1] = 2.0
        self.tile_sort_uniform_values[0][2] = separation
        self.tile_sort_uniform_values[0][3] = number_of_bodies
        self.tile_sort_uniform_values[0][4] = k
        self.tile_sort_uniform_values[0][5] = j

        self.tile_sort_uniform_buffer_staging.map_sync(wgpu.MapMode.WRITE)
        self.tile_sort_uniform_buffer_staging.write_mapped(
            self.tile_sort_uniform_values
        )
        self.tile_sort_uniform_buffer_staging.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.tile_sort_uniform_buffer_staging,
            source_offset=0,
            destination=self.tile_sort_uniform_buffer,
            destination_offset=0,
            size=self.tile_sort_uniform_values.nbytes,
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


class ComputeQuadTreePhase:
    shader_source = """
        struct Node {
            bb_min: vec2<f32>,
            bb_max: vec2<f32>,
            mass_center: vec2<f32>,
            mass: f32,
            first_child: u32,
            child_count: u32,
            top_left: u32,
            top_right: u32,
            bottom_left: u32,
            bottom_right: u32,
            lock: atomic<u32>,
        };

        struct Uniforms {
            tree_bb_min: vec2<f32>,
            tree_bb_max: vec2<f32>,
            separation: f32,
            number_of_bodies: u32,
        };

        struct NodeQueue {
            lock: atomic<u32>,
            count: u32,
        }

        @group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
        @group(0) @binding(1) var<storage, read_write> node_queue: NodeQueue;
        @group(0) @binding(2) var<storage, read> tiles: array<u32>;
        @group(0) @binding(3) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(4) var<uniform> uniforms: Uniforms;
        
        fn lock_node(n: u32) {
            while atomicExchange(&nodes[n].lock, 1u) == 1u {};
        }
        
        fn unlock_node(n: u32) {
            atomicStore(&nodes[n].lock, 0u);
        }

        fn lock_root() {
            while atomicExchange(&node_queue.lock, 1u) == 1u {};
            if node_queue.count == 0 {
                node_queue.count = 1;
                nodes[0].bb_min = uniforms.tree_bb_min;
                nodes[0].bb_max = uniforms.tree_bb_max;
                nodes[0].mass_center = vec2<f32>(0.0, 0.0);
                nodes[0].mass = 0.0;
                nodes[0].first_child = 0u;
                nodes[0].child_count = 0u;
                nodes[0].top_left = 0u;
                nodes[0].top_right = 0u;
                nodes[0].bottom_left = 0u;
                nodes[0].bottom_right = 0u;
                atomicStore(&nodes[0].lock, 1u);
                atomicStore(&node_queue.lock, 0u);
            } else {
                atomicStore(&node_queue.lock, 0u);
                lock_node(0u);
            }
        }

        


        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            nodes[i].top_left = 0u;
            node_queue.count = tiles[i] + uniforms.number_of_bodies;
            nodes[i].mass = positions[i].x;
            lock_root();
            unlock_node(0u);
        }

    """

    def __init__(self, device, number_of_bodies):
        self.number_of_bodies = number_of_bodies

        self.uniform_values = np.zeros(shape=(1,), dtype=np.dtype("2f4,2f4,f4,u4,8V"))

        self.node_values = np.zeros(
            shape=(1,), dtype=np.dtype("2f4,2f4,f4,u4,u4,u4,u4,u4,u4,u4")
        )

        self.nodes_buffer = device.create_buffer(
            size=self.number_of_bodies * self.node_values.nbytes * 8,
            usage=wgpu.BufferUsage.STORAGE,
        )

        self.node_queue_buffer = device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.STORAGE + wgpu.BufferUsage.COPY_DST,
        )

        self.node_queue_buffer_staging = device.create_buffer(
            size=32,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.MAP_WRITE,
        )

        self.uniform_buffer = device.create_buffer(
            size=self.uniform_values.nbytes,
            usage=wgpu.BufferUsage.UNIFORM + wgpu.BufferUsage.COPY_DST,
        )

        self.uniform_buffer_staging = device.create_buffer(
            size=self.uniform_values.nbytes,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.MAP_WRITE,
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
        tiles_buffer,
        command_encoder,
        number_of_bodies,
        separation,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.nodes_buffer}},
                {"binding": 1, "resource": {"buffer": self.node_queue_buffer}},
                {"binding": 2, "resource": {"buffer": tiles_buffer}},
                {"binding": 3, "resource": {"buffer": positions_buffer}},
                {"binding": 4, "resource": {"buffer": self.uniform_buffer}},
            ],
        )

        self.uniform_values[0][0][0] = -2.0
        self.uniform_values[0][0][1] = -2.0
        self.uniform_values[0][1][0] = 2.0
        self.uniform_values[0][1][1] = 2.0
        self.uniform_values[0][2] = separation
        self.uniform_values[0][3] = number_of_bodies

        node_queue_values = np.zeros(shape=(8,), dtype=np.uint32)

        self.uniform_buffer_staging.map_sync(wgpu.MapMode.WRITE)
        self.node_queue_buffer_staging.map_sync(wgpu.MapMode.WRITE)
        self.uniform_buffer_staging.write_mapped(self.uniform_values)
        self.node_queue_buffer_staging.write_mapped(node_queue_values)
        self.uniform_buffer_staging.unmap()
        self.node_queue_buffer_staging.unmap()

        command_encoder.copy_buffer_to_buffer(
            source=self.uniform_buffer_staging,
            source_offset=0,
            destination=self.uniform_buffer,
            destination_offset=0,
            size=self.uniform_values.nbytes,
        )

        command_encoder.copy_buffer_to_buffer(
            source=self.node_queue_buffer_staging,
            source_offset=0,
            destination=self.node_queue_buffer,
            destination_offset=0,
            size=32,
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


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

    def __init__(self, device, context, window_width, window_height):
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
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(
            slot=0, buffer=vertex_positions_buffer, offset=0, size=None
        )
        render_pass.set_vertex_buffer(
            slot=1, buffer=vertex_colors_buffer, offset=0, size=None
        )
        render_pass.draw(number_of_points * point_resolution * 3, 1, 0, 0)
        render_pass.end()

    def get_pipeline_kwargs(self, device, context):
        render_texture_format = context.get_preferred_format(device.adapter)
        shader = device.create_shader_module(code=self.rendering_shader_source)
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[])

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


class ComputeAccelerationsPhase:
    compute_accelerations_shader_source = """
        override separation: f32;
        override gravitational_constant: f32;
        override dt: f32;
        override number_of_bodies: u32;
        @group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read_write> accelerations: array<vec2<f32>>;

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            var accel: vec2<f32> = vec2<f32>(0, 0);
            for (var j = 0u; j < number_of_bodies; j++) {
                let q = positions[j];
                let d = distance(q, p);
                if d >= 2 * separation {
                    accel = accel + (q - p) / pow(d, 3.0);
                } 
            } 
            accelerations[i] = accel;
        }
    """

    def __init__(
        self, number_of_bodies, device, separation, gravitational_constant, dt
    ):
        self.number_of_bodies = number_of_bodies
        self.separation = separation
        self.gravitational_constant = gravitational_constant
        self.dt = dt

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(
            code=self.compute_accelerations_shader_source
        )
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
                "constants": {
                    "separation": self.separation,
                    "gravitational_constant": self.gravitational_constant,
                    "dt": self.dt,
                    "number_of_bodies": self.number_of_bodies,
                },
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        acceleratios_buffer,
        command_encoder,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": acceleratios_buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(self.number_of_bodies)
        compute_pass.end()


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
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer}},
                {"binding": 2, "resource": {"buffer": acceleratios_buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


class ComputeAccelerationsPhase:
    compute_accelerations_shader_source = """
        override separation: f32;
        override gravitational_constant: f32;
        override dt: f32;
        override number_of_bodies: u32;
        @group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read_write> accelerations: array<vec2<f32>>;

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i];
            var accel: vec2<f32> = vec2<f32>(0, 0);
            for (var j = 0u; j < number_of_bodies; j++) {
                let q = positions[j];
                let d = distance(q, p);
                if d >= 2 * separation {
                    accel = accel + (q - p) / pow(d, 3.0);
                } 
            } 
            accelerations[i] = accel;
        }
    """

    def __init__(
        self, number_of_bodies, device, separation, gravitational_constant, dt
    ):
        self.number_of_bodies = number_of_bodies
        self.separation = separation
        self.gravitational_constant = gravitational_constant
        self.dt = dt

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(
            code=self.compute_accelerations_shader_source
        )
        return dict(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={
                "module": shader,
                "entry_point": "main",
                "constants": {
                    "separation": self.separation,
                    "gravitational_constant": self.gravitational_constant,
                    "dt": self.dt,
                    "number_of_bodies": self.number_of_bodies,
                },
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        acceleratios_buffer,
        command_encoder,
        number_of_bodies,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": acceleratios_buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


class ApplyImpulsesPhase:
    apply_impulses_shader_source = """
        @group(0) @binding(0) var<storage, read_write> positions: array<vec2<f32>>;
        @group(0) @binding(1) var<storage, read> impulses: array<vec2<f32>>;

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            positions[i] = positions[i] + impulses[i];
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
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": impulses_buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


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
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": impulses_buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


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
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


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
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": prev_positions_buffer}},
                {"binding": 2, "resource": {"buffer": colors_buffer}},
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()


class TessellationPhase:
    tessellation_compute_shader_source = """
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

    def __init__(self, number_of_points, point_resolution, point_radius, device):
        self.point_radius = point_radius
        self.number_of_points = number_of_points
        self.point_resolution = point_resolution

        pipeline_kwargs = self.get_pipeline_kwargs(device)
        self.pipeline = device.create_compute_pipeline(**pipeline_kwargs)

        self.vertex_colors_buffer = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.VERTEX,
        )

        self.vertex_colors_buffer_staging = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.STORAGE,
        )

        self.vertex_positions_buffer = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.VERTEX,
        )

        self.vertex_positions_buffer_staging = device.create_buffer(
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
            usage=wgpu.BufferUsage.COPY_SRC + wgpu.BufferUsage.STORAGE,
        )

    def get_pipeline_kwargs(self, device):
        shader = device.create_shader_module(
            code=self.tessellation_compute_shader_source
        )
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

    def tesselation_pass(
        self,
        device,
        positions_buffer,
        colors_buffer,
        command_encoder,
    ):
        command_encoder.copy_buffer_to_buffer(
            source=self.vertex_positions_buffer_staging,
            source_offset=0,
            destination=self.vertex_positions_buffer,
            destination_offset=0,
            size=self.number_of_points * self.point_resolution * 3 * 2 * 4,
        )

        command_encoder.copy_buffer_to_buffer(
            source=self.vertex_colors_buffer_staging,
            source_offset=0,
            destination=self.vertex_colors_buffer,
            destination_offset=0,
            size=self.number_of_points * self.point_resolution * 3 * 4 * 4,
        )

        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": positions_buffer}},
                {"binding": 1, "resource": {"buffer": colors_buffer}},
                {
                    "binding": 2,
                    "resource": {"buffer": self.vertex_positions_buffer_staging},
                },
                {
                    "binding": 3,
                    "resource": {"buffer": self.vertex_colors_buffer_staging},
                },
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(self.number_of_points)
        compute_pass.end()


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

        self.compute_point_tile_phase = ComputePointTilePhase(
            self.device, self.number_of_points
        )

        self.tile_sort_phase = TileSortPhase(self.device)

        self.compute_quad_tree_phase = ComputeQuadTreePhase(
            self.device, self.number_of_points
        )

        self.compute_accelerations_phase = ComputeAccelerationsPhase(
            self.number_of_points,
            self.device,
            self.simulation.separation,
            self.simulation.gravitational_constant,
            self.simulation.dt,
        )

        self.apply_accelerations_phase = ApplyAccelerationsPhase(
            self.device,
            self.simulation.gravitational_constant,
            self.simulation.dt,
        )

        self.compute_impulses_phase = ComputeImpulsesPhase(
            self.number_of_points,
            self.device,
            self.simulation.separation,
        )

        self.apply_impulses_phase = ApplyImpulsesPhase(self.device)

        self.setup_velocities_phase = SetupVelocitiesPhase(
            self.device, self.simulation.initial_spin, self.simulation.dt
        )

        self.compute_colors_phase = ComputeColorPhase(self.device, self.simulation.dt)

        self.tessellation_phase = TessellationPhase(
            self.number_of_points, self.point_resolution, self.point_radius, self.device
        )

        self.rendering_phase = RenderingPhase(
            self.device, self.context, self.window_width, self.window_height
        )

        self.colors_buffer = self.device.create_buffer(
            size=self.number_of_points * 4 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.STORAGE,
        )

        self.colors_buffer_staging = self.device.create_buffer(
            size=self.number_of_points * 4 * 4,
            usage=wgpu.BufferUsage.MAP_WRITE + wgpu.BufferUsage.COPY_SRC,
        )

        self.positions_buffer = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.STORAGE,
        )

        self.positions_buffer_staging = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.MAP_WRITE + wgpu.BufferUsage.COPY_SRC,
        )

        self.prev_positions_buffer = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.STORAGE,
        )

        self.prev_positions_buffer_staging = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.MAP_WRITE + wgpu.BufferUsage.COPY_SRC,
        )

        self.impulses_buffer = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )

        self.accelerations_buffer = self.device.create_buffer(
            size=self.number_of_points * 2 * 4,
            usage=wgpu.BufferUsage.STORAGE,
        )

        width_blocks = (self.window_width * 4 + 255) // 256
        self.color_output_buffer = self.device.create_buffer(
            size=width_blocks * 256 * self.window_height,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.MAP_READ,
        )
        self.loaded = False
        return self.get_draw_function()

    def get_draw_function(self):
        def draw_frame():
            positions_colors = self.simulation.start_frame()
            if positions_colors is None:
                self.canvas.close()
                return
            positions, prev_positions, colors = positions_colors

            current_texture = self.context.get_current_texture()

            command_encoder = self.device.create_command_encoder()

            if not self.loaded:
                self.positions_buffer_staging.map_sync(wgpu.MapMode.WRITE)
                self.prev_positions_buffer_staging.map_sync(wgpu.MapMode.WRITE)
                self.colors_buffer_staging.map_sync(wgpu.MapMode.WRITE)

                self.positions_buffer_staging.write_mapped(positions)
                self.prev_positions_buffer_staging.write_mapped(prev_positions)
                self.colors_buffer_staging.write_mapped(colors)

                self.positions_buffer_staging.unmap()
                self.prev_positions_buffer_staging.unmap()
                self.colors_buffer_staging.unmap()

                command_encoder.copy_buffer_to_buffer(
                    source=self.positions_buffer_staging,
                    source_offset=0,
                    destination=self.positions_buffer,
                    destination_offset=0,
                    size=self.number_of_points * 2 * 4,
                )

                command_encoder.copy_buffer_to_buffer(
                    source=self.prev_positions_buffer_staging,
                    source_offset=0,
                    destination=self.prev_positions_buffer,
                    destination_offset=0,
                    size=self.number_of_points * 2 * 4,
                )

                command_encoder.copy_buffer_to_buffer(
                    source=self.colors_buffer_staging,
                    source_offset=0,
                    destination=self.colors_buffer,
                    destination_offset=0,
                    size=self.number_of_points * 4 * 4,
                )
                self.compute_point_tile_phase.compute_pass(
                    self.device,
                    self.positions_buffer,
                    command_encoder,
                    self.simulation.separation,
                )
                for _ in range(self.simulation.init_substeps):
                    k = 2
                    while k <= self.number_of_points:
                        j = k // 2
                        while j > 0:
                            self.tile_sort_phase.compute_pass(
                                self.device,
                                self.positions_buffer,
                                self.prev_positions_buffer,
                                self.compute_point_tile_phase.tiles_buffer,
                                command_encoder,
                                self.number_of_points,
                                self.simulation.separation,
                                k,
                                j,
                            )
                            j //= 2
                        k *= 2

                        self.compute_quad_tree_phase.compute_pass(
                            self.device,
                            self.positions_buffer,
                            self.compute_point_tile_phase.tiles_buffer,
                            command_encoder,
                            self.number_of_points,
                            self.simulation.separation,
                        )

                    self.compute_impulses_phase.compute_pass(
                        self.device,
                        self.positions_buffer,
                        self.impulses_buffer,
                        command_encoder,
                        self.number_of_points,
                    )

                    self.apply_impulses_phase.compute_pass(
                        self.device,
                        self.positions_buffer,
                        self.impulses_buffer,
                        command_encoder,
                        self.number_of_points,
                    )

                self.setup_velocities_phase.compute_pass(
                    self.device,
                    self.positions_buffer,
                    self.prev_positions_buffer,
                    command_encoder,
                    self.number_of_points,
                )

                self.loaded = True

            for _ in range(self.simulation.substeps):
                k = 2
                while k <= self.number_of_points:
                    j = k // 2
                    while j > 0:
                        self.tile_sort_phase.compute_pass(
                            self.device,
                            self.positions_buffer,
                            self.prev_positions_buffer,
                            self.compute_point_tile_phase.tiles_buffer,
                            command_encoder,
                            self.number_of_points,
                            self.simulation.separation,
                            k,
                            j,
                        )
                        j //= 2
                    k *= 2

                    # self.compute_quad_tree_phase.compute_pass(
                    #     self.device,
                    #     self.positions_buffer,
                    #     self.tile_sort_phase.tiles_buffer,
                    #     command_encoder,
                    #     self.number_of_points,
                    #     self.simulation.separation,
                    # )

                self.compute_accelerations_phase.compute_pass(
                    self.device,
                    self.positions_buffer,
                    self.accelerations_buffer,
                    command_encoder,
                    self.number_of_points,
                )

                self.apply_accelerations_phase.compute_pass(
                    self.device,
                    self.positions_buffer,
                    self.prev_positions_buffer,
                    self.accelerations_buffer,
                    command_encoder,
                    self.number_of_points,
                )

                self.compute_impulses_phase.compute_pass(
                    self.device,
                    self.positions_buffer,
                    self.impulses_buffer,
                    command_encoder,
                    self.number_of_points,
                )

                self.apply_impulses_phase.compute_pass(
                    self.device,
                    self.positions_buffer,
                    self.impulses_buffer,
                    command_encoder,
                    self.number_of_points,
                )

            self.compute_colors_phase.compute_pass(
                self.device,
                self.positions_buffer,
                self.prev_positions_buffer,
                command_encoder,
                self.number_of_points,
                self.colors_buffer,
            )

            self.tessellation_phase.tesselation_pass(
                self.device, self.positions_buffer, self.colors_buffer, command_encoder
            )

            self.rendering_phase.render_pass(
                command_encoder,
                current_texture,
                self.tessellation_phase.vertex_positions_buffer,
                self.tessellation_phase.vertex_colors_buffer,
                self.number_of_points,
                self.point_resolution,
            )

            width_blocks = (self.window_width * 4 + 255) // 256
            command_encoder.copy_texture_to_buffer(
                {
                    "aspect": wgpu.TextureAspect.all,
                    "mip_level": 0,
                    "origin": {"x": 0, "y": 0, "z": 0},
                    "texture": self.rendering_phase.color_resolve_texture,
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
            image_bytes_padded = np.frombuffer(
                image_mem.tobytes(), dtype=np.byte
            ).reshape((self.window_height, width_blocks * 256 // 4, 4))
            image_array = image_bytes_padded[:, : self.window_width, :]
            self.simulation.end_frame(image_array)
            self.canvas.request_draw(draw_frame)

        return draw_frame
