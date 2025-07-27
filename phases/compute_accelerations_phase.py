import wgpu
from buffer import Buffer
import numpy as np


class ComputeAccelerationsPhase:
    compute_accelerations_shader_source = """
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

        override separation: f32;
        override gravitational_constant: f32;
        override dt: f32;
        override number_of_bodies: u32;
        override max_tree_depth: u32;
        override theta: f32;

        @group(0) @binding(0) var<storage, read> nodes: array<Node>;
        @group(0) @binding(1) var<storage, read> positions: array<vec2<f32>>;
        @group(0) @binding(2) var<storage, read_write> accelerations: array<vec2<f32>>;

        var<private> stack_pointer: u32 = 1u;
        var<private> stack: array<u32, 64u>;

        fn compute_acceleration(position: vec2<f32> ) -> vec2<f32> {
            stack[0u] = 0u;
            var accel: vec2<f32> = vec2<f32>(0.0, 0.0);
            loop {
                if stack_pointer == 0u {
                    break;
                }
                stack_pointer -= 1u;
                let u = stack[stack_pointer];
                let s = max(nodes[u].bb_max[0u] - nodes[u].bb_min[0u], nodes[u].bb_max[1u] - nodes[u].bb_min[1u]);
                let dir = nodes[u].mass_center - position;
                let d = length(dir);

                if d / s > 1 / theta {
                    accel += dir / d / (d*d) * nodes[u].mass;
                    continue;
                } 

                if nodes[u].top_left != 0u {
                    stack[stack_pointer] = nodes[u].top_left;
                    stack_pointer += 1u;
                }

                if nodes[u].top_right != 0u {
                    stack[stack_pointer] = nodes[u].top_right;
                    stack_pointer += 1u;
                }

                if nodes[u].bottom_left != 0u {
                    stack[stack_pointer] = nodes[u].bottom_left;
                    stack_pointer += 1u;
                }

                if nodes[u].bottom_right != 0u {
                    stack[stack_pointer] = nodes[u].bottom_right;
                    stack_pointer += 1u;
                }

                for (var j = nodes[u].start_child; j < nodes[u].end_child; j++) {
                    let other = positions[j];
                    let dir = other - position;
                    let d = length(dir);
                    if d <= 2.0 * separation {
                        continue;
                    }
                    accel += dir / d / (d*d) * 1.0;
                }

            }
            
            return accel;
        }

        @compute @workgroup_size(1, 1, 1) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let i = id.x;
            let p = positions[i]; 
            accelerations[i] = compute_acceleration(p);
        }
    """

    def __init__(
        self,
        number_of_bodies,
        device,
        separation,
        gravitational_constant,
        dt,
        max_tree_depth,
        theta,
    ):
        self.number_of_bodies = number_of_bodies
        self.separation = separation
        self.gravitational_constant = gravitational_constant
        self.dt = dt
        self.max_tree_depth = max_tree_depth
        self.theta = theta

        self.accelerations_buffer = Buffer(
            device=device,
            shape=(self.number_of_bodies,),
            dtype=np.dtype("2f4"),
            usage=wgpu.BufferUsage.STORAGE,
        )

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
                    "max_tree_depth": self.max_tree_depth,
                    "theta": self.theta,
                },
            },
        )

    def compute_pass(
        self,
        device,
        positions_buffer,
        command_encoder,
        number_of_bodies,
        nodes_buffer,
    ):
        bind_group = device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": nodes_buffer.buffer}},
                {"binding": 1, "resource": {"buffer": positions_buffer.buffer}},
                {
                    "binding": 2,
                    "resource": {"buffer": self.accelerations_buffer.buffer},
                },
            ],
        )

        compute_pass = command_encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups(number_of_bodies)
        compute_pass.end()
