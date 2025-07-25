from buffer import Buffer
import numpy as np
import wgpu


class ComputeQuadTreePhase:
    shader_source = """
        struct Node {
            bb_min: vec2<f32>,
            bb_max: vec2<f32>,
            mass_center: vec2<f32>,
            mass: f32,
            start_child: u32,
            end_child: u32,
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
        
        override max_tree_depth: u32;

        @group(0) @binding(0) var<storage, read_write> nodes: array<Node>;
        @group(0) @binding(1) var<storage, read_write> node_queue: NodeQueue;
        @group(0) @binding(2) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(3) var<uniform> uniforms: Uniforms;
        
        fn lock_node(n: u32) {
            while atomicExchange(&nodes[n].lock, 1u) == 1u {};
        }
        
        fn unlock_node(n: u32) {
            atomicStore(&nodes[n].lock, 0u);
        }

        fn ensure_root() {
            while atomicExchange(&node_queue.lock, 1u) == 1u {};
            if node_queue.count == 0 {
                node_queue.count = 1;
                nodes[0].bb_min = uniforms.tree_bb_min;
                nodes[0].bb_max = uniforms.tree_bb_max;
                nodes[0].mass_center = vec2<f32>(0.0, 0.0);
                nodes[0].mass = 0.0;
                nodes[0].start_child = 0u;
                nodes[0].end_child = 0u;
                nodes[0].top_left = 0u;
                nodes[0].top_right = 0u;
                nodes[0].bottom_left = 0u;
                nodes[0].bottom_right = 0u;
                atomicStore(&nodes[0].lock, 0u);
            } 
            atomicStore(&node_queue.lock, 0u);
        }

        fn allocate_node(bb_min: vec2<f32>, bb_max: vec2<f32>) -> u32 {
            while atomicExchange(&node_queue.lock, 1u) == 1u {};

            let n = node_queue.count;
            node_queue.count += 1u;

            nodes[n].bb_min = bb_min;
            nodes[n].bb_max = bb_max;
            nodes[n].mass_center = vec2<f32>(0.0, 0.0);
            nodes[n].mass = 0.0;
            nodes[n].start_child = 0u;
            nodes[n].end_child = 0u;
            nodes[n].top_left = 0u;
            nodes[n].top_right = 0u;
            nodes[n].bottom_left = 0u;
            nodes[n].bottom_right = 0u;
            atomicStore(&nodes[n].lock, 0u);

            atomicStore(&node_queue.lock, 0u);

            return n;
        }

        
        fn add_point_to_tree(point_index: u32) {
            let p = positions[point_index];
            var current_node: u32 = 0u;
            var depth: u32 = 0u;
            loop {
                let u = current_node;
                lock_node(u);
                nodes[u].mass_center = (nodes[u].mass_center * nodes[u].mass + p) / (nodes[u].mass + 1.0);
                nodes[u].mass += 1.0;
                if depth == max_tree_depth {
                    let child_count = nodes[u].end_child - nodes[u].start_child;
                    if child_count == 0u {
                        nodes[u].start_child = point_index;
                        nodes[u].end_child   = point_index + 1u;
                    } else {
                        nodes[u].start_child = min(nodes[u].start_child, point_index);
                        nodes[u].end_child   = max(nodes[u].end_child, point_index + 1u);
                    }
                    unlock_node(u);
                    break;
                }
                 
                let hori: f32 = (nodes[u].bb_max[1u] + nodes[u].bb_min[1u]) / 2.0;
                let vert: f32 = (nodes[u].bb_max[0u] + nodes[u].bb_min[0u]) / 2.0;
                if p.x < vert && p.y < hori {
                    var v: u32 = nodes[u].top_left;
                    if v == 0u {
                        v = allocate_node(nodes[u].bb_min, vec2<f32>(vert, hori));
                        nodes[u].top_left = v;
                    }
                    current_node = v;
                } else if p.x >= vert && p.y < hori {
                    var v: u32 = nodes[u].top_right;
                    if v == 0u {
                        v = allocate_node(vec2<f32>(vert, nodes[u].bb_min[1u]), vec2<f32>(nodes[u].bb_max[0u], hori));
                        nodes[u].top_right = v;
                    }
                    current_node = v;
                } else if p.x < vert && p.y >= hori {
                    var v: u32 = nodes[u].bottom_left;
                    if v == 0u {
                        v = allocate_node(vec2<f32>(nodes[u].bb_min[0u], hori), vec2<f32>(vert, nodes[u].bb_max[1u]));
                        nodes[u].bottom_left = v;
                    }
                    current_node = v;
                } else if p.x >= vert && p.y >= hori {
                    var v: u32 = nodes[u].bottom_right;
                    if v == 0u {
                        v = allocate_node(vec2<f32>(vert, hori), vec2<f32>(nodes[u].bb_max[0u], nodes[u].bb_max[1u]));
                        nodes[u].bottom_right = v;
                    }
                    current_node = v;
                }
                unlock_node(u);
                depth += 1u;
            }
        }
        


        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            ensure_root();
            let i = id.x;
            add_point_to_tree(i);
        }

    """

    def __init__(
        self, device, number_of_bodies, max_tree_depth, tree_bb_min, tree_bb_max
    ):
        self.number_of_bodies = number_of_bodies
        self.max_tree_depth = max_tree_depth
        self.tree_bb_min = tree_bb_min
        self.tree_bb_max = tree_bb_max

        self.dtype = np.dtype(
            [
                ("bb_min", "2f4"),
                ("bb_max", "2f4"),
                ("mass_center", "2f4"),
                ("mass", "f4"),
                ("start_child", "u4"),
                ("end_child", "u4"),
                ("top_left", "u4"),
                ("top_right", "u4"),
                ("bottom_left", "u4"),
                ("bottom_right", "u4"),
                ("lock", "u4"),
            ]
        )
        self.uniform_values = np.zeros(shape=(1,), dtype=np.dtype("2f4,2f4,f4,u4"))

        self.nodes_buffer = Buffer(
            device=device,
            shape=(self.number_of_bodies * self.max_tree_depth * 4,),
            dtype=self.dtype,
            usage=wgpu.BufferUsage.STORAGE,
            query=True,
        )

        self.node_queue_buffer = Buffer(
            device=device,
            shape=(1,),
            dtype=np.dtype("u4,u4"),
            usage=wgpu.BufferUsage.STORAGE,
            staging=True,
        )

        self.uniform_buffer = Buffer(
            device=device,
            shape=(1,),
            dtype=np.dtype("2f4,2f4,f4,u4"),
            usage=wgpu.BufferUsage.UNIFORM,
            staging=True,
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
                "constants": {
                    "max_tree_depth": self.max_tree_depth,
                },
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        command_encoder,
        number_of_bodies,
        separation,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": self.nodes_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": self.node_queue_buffer.buffer}},
                {"binding": 2, "resource": {"buffer": positions_buffer.buffer}},
                {"binding": 3, "resource": {"buffer": self.uniform_buffer.buffer}},
            ],
        )

        self.uniform_values[0][0] = self.tree_bb_min
        self.uniform_values[0][1] = self.tree_bb_max
        self.uniform_values[0][2] = separation
        self.uniform_values[0][3] = number_of_bodies

        node_queue_values = np.zeros(
            shape=self.uniform_buffer.shape, dtype=self.uniform_buffer.dtype
        )

        self.uniform_buffer.write_staging(self.uniform_values)
        self.node_queue_buffer.write_staging(node_queue_values)
        self.uniform_buffer.load_staging(command_encoder)
        self.node_queue_buffer.load_staging(command_encoder)

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()

        self.nodes_buffer.store_query(command_encoder)
