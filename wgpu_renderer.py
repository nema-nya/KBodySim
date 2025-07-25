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

        self.collisions_phase = CollisionsPhase(
            self.device, self.number_of_points, self.simulation
        )

        self.gravity_phase = GravityPhase(
            self.device, self.number_of_points, self.simulation
        )

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

        width_blocks = (self.window_width * 4 + 255) // 256
        self.color_output_buffer = self.device.create_buffer(
            size=width_blocks * 256 * self.window_height,
            usage=wgpu.BufferUsage.COPY_DST + wgpu.BufferUsage.MAP_READ,
        )
        self.loaded = False
        return self.get_draw_function()

    def get_draw_function(self):
        def draw_frame():
            if not self.simulation.start_frame():
                self.canvas.close()
                return

            current_texture = self.context.get_current_texture()

            aspect = current_texture.width / current_texture.height
            mvp_values = np.eye(4, dtype=np.float32)
            mvp_values[0, 0] = 1 / aspect
            self.rendering_phase.mvp_buffer.write_staging(mvp_values)

            command_encoder = self.device.create_command_encoder()
            self.rendering_phase.mvp_buffer.load_staging(command_encoder)

            if not self.loaded:
                positions_values = (
                    np.random.randn(self.number_of_points, 2).astype(np.float32)
                    * self.simulation.spawn_radius
                )
                colors_values = np.ones(shape=(self.number_of_points, 4)).astype(
                    np.float32
                )
                self.positions_buffer.write_staging(positions_values)
                self.prev_positions_buffer.write_staging(positions_values)
                self.colors_buffer.write_staging(colors_values)

                self.positions_buffer.load_staging(command_encoder)
                self.prev_positions_buffer.load_staging(command_encoder)
                self.colors_buffer.load_staging(command_encoder)

                for _ in range(self.simulation.init_substeps):
                    self.collisions_phase.compute_pass(
                        command_encoder,
                        self.positions_buffer,
                        self.prev_positions_buffer,
                    )

                self.setup_velocities_phase.compute_pass(
                    self.device,
                    self.positions_buffer,
                    self.prev_positions_buffer,
                    command_encoder,
                    self.number_of_points,
                )

                self.loaded = True
            else:
                for _ in range(self.simulation.substeps):
                    self.gravity_phase.compute_pass(
                        command_encoder,
                        self.positions_buffer,
                        self.prev_positions_buffer,
                    )
                    self.collisions_phase.compute_pass(
                        command_encoder,
                        self.positions_buffer,
                        self.prev_positions_buffer,
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

            self.simulation.end_frame(image_mem)
            self.canvas.request_draw(draw_frame)

        return draw_frame
