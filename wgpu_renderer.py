import wgpu
from wgpu.gui.auto import WgpuCanvas, run
import numpy as np
from buffer import Buffer
from phases import *


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
            self.device,
            self.number_of_points,
            self.simulation.tree_bb_min,
            self.simulation.tree_bb_max,
        )

        self.tile_sort_phase = {}

        k = 2
        while k <= self.number_of_points:
            j = k // 2
            while j > 0:
                self.tile_sort_phase[(k, j)] = TileSortPhase(
                    self.device,
                    self.simulation.tree_bb_min,
                    self.simulation.tree_bb_max,
                )
                j = j // 2
            k = k * 2

        self.compute_quad_tree_phase = ComputeQuadTreePhase(
            self.device,
            self.number_of_points,
            self.simulation.max_tree_depth,
            self.simulation.tree_bb_min,
            self.simulation.tree_bb_max,
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

        self.colors_buffer = Buffer(
            device=self.device,
            shape=(self.number_of_points,),
            dtype=np.dtype("4f4"),
            usage=wgpu.BufferUsage.STORAGE,
            staging=True,
        )

        self.positions_buffer = Buffer(
            device=self.device,
            shape=(self.number_of_points,),
            dtype=np.dtype("2f4"),
            usage=wgpu.BufferUsage.STORAGE,
            staging=True,
        )

        self.prev_positions_buffer = Buffer(
            device=self.device,
            shape=(self.number_of_points,),
            dtype=np.dtype("2f4"),
            usage=wgpu.BufferUsage.STORAGE,
            staging=True,
        )

        self.impulses_buffer = Buffer(
            device=self.device,
            shape=(self.number_of_points,),
            dtype=np.dtype("2f4"),
            usage=wgpu.BufferUsage.STORAGE,
        )

        self.accelerations_buffer = Buffer(
            device=self.device,
            shape=(self.number_of_points,),
            dtype=np.dtype("2f4"),
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

            aspect = current_texture.width / current_texture.height
            mvp_values = np.eye(4, dtype=np.float32)
            mvp_values[0, 0] = 1 / aspect
            self.rendering_phase.mvp_buffer.write_staging(mvp_values)

            command_encoder = self.device.create_command_encoder()
            self.rendering_phase.mvp_buffer.load_staging(command_encoder)

            if not self.loaded:

                self.positions_buffer.write_staging(positions)
                self.prev_positions_buffer.write_staging(prev_positions)
                self.colors_buffer.write_staging(colors)

                self.positions_buffer.load_staging(command_encoder)
                self.prev_positions_buffer.load_staging(command_encoder)
                self.colors_buffer.load_staging(command_encoder)

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
                            self.tile_sort_phase[(k, j)].compute_pass(
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
                            j = j // 2
                        k = k * 2

                    self.compute_point_tile_phase.tiles_buffer.store_query(
                        command_encoder
                    )

                    self.compute_quad_tree_phase.compute_pass(
                        self.device,
                        self.positions_buffer,
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
            # node_values = self.compute_quad_tree_phase.nodes_buffer.read_query()
            # node_array = np.frombuffer(
            #     node_values, dtype=self.compute_quad_tree_phase.nodes_buffer.dtype
            # )
            # print(node_array)

            # tiles_values = self.compute_point_tile_phase.tiles_buffer.read_query()

            # tiles_array = np.frombuffer(tiles_values, dtype=np.uint32)
            # print(tiles_array)

            self.color_output_buffer.map_sync(wgpu.MapMode.READ)
            image_mem = self.color_output_buffer.read_mapped()
            self.color_output_buffer.unmap()

            self.simulation.end_frame(image_mem)
            self.canvas.request_draw(draw_frame)

        return draw_frame
