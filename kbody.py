import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer
from quad_tree_v2 import QuadTreeV2

number_of_points = 5
point_resolution = 32
window_width = 1024
window_height = 1024
separation = 0.01
point_radius = separation
gravitational_constant = 0.001
substeps = 8
dt = 0.001
frame_count = 900
pole_distance = separation * 5
spawn_radius = 0.2
initial_spin = 0.003
pows = [2**i for i in range(32)]
pow_to_p = {p: i for i, p in enumerate(pows)}


class Simulation:

    def __init__(self):
        self.positions = (
            (np.random.rand(number_of_points, 2) * 2 - 1)
            * (1 - separation)
            * spawn_radius
        )
        self.positions = self.positions.astype(np.float32)
        self.colors = np.ones(shape=(number_of_points, 4), dtype=np.float32)
        self.prev_positions = np.zeros_like(self.positions)
        for _ in range(substeps):
            quad_tree = QuadTreeV2(self.positions, separation * 2, pole_distance, gravitational_constant)
            deltas = quad_tree.get_collisions()
            self.positions += deltas
        velocity = np.stack([self.positions[:, 1], -self.positions[:, 0]], axis=-1)
        velocity = velocity / np.linalg.norm(velocity, axis=-1, keepdims=True)
        self.prev_positions = self.positions + velocity * dt * initial_spin
        self.frame_count = frame_count

    def end_frame(self, image_array):
        ffmpeg_writer.add_frame(image_array)

    def recenter_view(self):
        mean_position = self.positions.mean(0, keepdims=True)
        self.positions -= mean_position
        self.prev_positions -= mean_position

    def start_frame(self):
        for _ in range(substeps):
            quad_tree = QuadTreeV2(self.positions, separation * 2, pole_distance, gravitational_constant)
            accels = quad_tree.get_gravity()
            self.positions, self.prev_positions = (
                2 * self.positions - self.prev_positions + dt**2 * accels,
                self.positions,
            )
            quad_tree = QuadTreeV2(self.positions, separation * 2, pole_distance, gravitational_constant)
            deltas = quad_tree.get_collisions()
            self.positions += deltas

        e_kin = 0
        for i in range(number_of_points):
            e_kin += (
                np.linalg.norm(self.positions[i] - self.prev_positions[i]) / dt
            ) ** 2 / 2

        scale = 100
        intensity = np.exp(
            -scale * (((self.positions - self.prev_positions) / dt) ** 2).sum(-1) / 2
        )
        direction = self.positions - self.prev_positions
        direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
        self.colors[:, :2] = (1 - intensity[:, None]) * (
            direction / 2 + 1 / 2
        ) + intensity[:, None] * np.array([1.0, 1.0])[None, :]

        self.recenter_view()
        if self.frame_count > 0:
            self.frame_count -= 1
            return self.positions, self.colors
        ffmpeg_writer.finish()
        return None


if __name__ == "__main__":
    simulation = Simulation()
    ffmpeg_writer = FfmpegWriter("out.webm", window_width, window_height)
    wgpu_renderer = WgpuRenderer(
        window_width,
        window_height,
        simulation,
        point_resolution,
        point_radius,
        number_of_points,
    )
