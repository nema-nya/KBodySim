import numpy as np
from ffmpeg_writer import FfmpegWriter
from wgpu_renderer import WgpuRenderer
import dataclasses

number_of_points = 4
point_resolution = 32
window_width = 1024
window_height = 1024
separation = 0.01
point_radius = separation
gravitational_constant = 0.001
substeps = 8
dt = 0.001
frame_count = 900
pole_distance = 1
spawn_radius = 0.2
initial_spin = 0.001
pows = [2**i for i in range(32)]
pow_to_p = {p: i for i, p in enumerate(pows)}


@dataclasses.dataclass
class QuadTreeNode:
    bounding_box: tuple
    top_left: any = None
    top_right: any = None
    bottom_left: any = None
    bottom_right: any = None
    points: list = dataclasses.field(default_factory=list)
    mass: float = 0.0
    mass_center: any = dataclasses.field(
        default_factory=lambda: np.zeros(shape=(2,), dtype=float)
    )


class QuadTree:

    def __init__(self, bounding_box, separation, pole_distance):
        self.bounding_box = bounding_box
        self.separation = separation
        self.pole_distance = pole_distance
        self.root = QuadTreeNode(bounding_box)

    def add_point(self, point, position):
        node = self.root
        while True:
            node.mass_center = (node.mass_center * node.mass + position) / (
                node.mass + 1
            )
            node.mass += 1
            bb = node.bounding_box
            if bb[2] - bb[0] < self.separation or bb[3] - bb[1] < self.separation:
                node.points.append((point, position))
                break
            vert = (bb[2] + bb[0]) / 2
            hori = (bb[3] + bb[1]) / 2
            if position[0] < vert and position[1] < hori:
                if node.top_left is None:
                    node.top_left = QuadTreeNode((bb[0], bb[1], vert, hori))
                node = node.top_left
            elif position[0] >= vert and position[1] < hori:
                if node.top_right is None:
                    node.top_right = QuadTreeNode((vert, bb[1], bb[2], hori))
                node = node.top_right
            elif position[0] < vert and position[1] >= hori:
                if node.bottom_left is None:
                    node.bottom_left = QuadTreeNode((bb[0], hori, vert, bb[3]))
                node = node.bottom_left
            elif position[0] >= vert and position[1] >= hori:
                if node.bottom_right is None:
                    node.bottom_right = QuadTreeNode((vert, hori, bb[2], bb[3]))
                node = node.bottom_right

    def get_collisions(self, position):
        queue = [self.root]
        result = []
        while queue:
            node = queue.pop()
            bb = node.bounding_box
            if bb[0] >= position[0] + self.separation:
                continue
            elif bb[1] >= position[1] + self.separation:
                continue
            elif bb[2] < position[0] - self.separation:
                continue
            elif bb[3] < position[1] - self.separation:
                continue
            if node.top_left is not None:
                queue.append(node.top_left)
            if node.top_right is not None:
                queue.append(node.top_right)
            if node.bottom_left is not None:
                queue.append(node.bottom_left)
            if node.bottom_right is not None:
                queue.append(node.bottom_right)

            for point, point_position in node.points:
                if np.linalg.norm(position - point_position) < self.separation:
                    result.append(point)

        return result

    def get_gravity(self, position):
        accel = np.zeros(shape=(2,), dtype=float)
        queue = [self.root]
        while queue:
            node = queue.pop()
            bb = node.bounding_box
            r = node.mass_center - position
            r_len = np.linalg.norm(r)
            if bb[0] >= position[0] + self.pole_distance:
                accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            elif bb[1] >= position[1] + self.pole_distance:
                accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            elif bb[2] < position[0] - self.pole_distance:
                accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            elif bb[3] < position[1] - self.pole_distance:
                accel += r / r_len**3 * gravitational_constant * node.mass
                continue
            if node.top_left is not None:
                queue.append(node.top_left)
            if node.top_right is not None:
                queue.append(node.top_right)
            if node.bottom_left is not None:
                queue.append(node.bottom_left)
            if node.bottom_right is not None:
                queue.append(node.bottom_right)

            for _, point_position in node.points:
                r = point_position - position
                r_len = np.linalg.norm(r)
                if r_len < self.separation:
                    continue
                accel += r / r_len**3 * gravitational_constant
        return accel

    def get_potential(self, position):
        pot = 0
        queue = [self.root]
        while queue:
            node = queue.pop()
            bb = node.bounding_box
            r = node.mass_center - position
            r_len = np.linalg.norm(r)
            if bb[0] >= position[0] + self.pole_distance:
                pot += gravitational_constant * node.mass / r_len
                continue
            elif bb[1] >= position[1] + self.pole_distance:
                pot += gravitational_constant * node.mass / r_len
                continue
            elif bb[2] < position[0] - self.pole_distance:
                pot += gravitational_constant * node.mass / r_len
                continue
            elif bb[3] < position[1] - self.pole_distance:
                pot += gravitational_constant * node.mass / r_len
                continue
            if node.top_left is not None:
                queue.append(node.top_left)
            if node.top_right is not None:
                queue.append(node.top_right)
            if node.bottom_left is not None:
                queue.append(node.bottom_left)
            if node.bottom_right is not None:
                queue.append(node.bottom_right)

            for _, point_position in node.points:
                r = point_position - position
                r_len = np.linalg.norm(r)
                if r_len < self.separation:
                    continue
                pot += gravitational_constant / r_len
        return -pot


class QuadTreeV2:
    def _find_power_of_two(self, n):
        n -= 1
        n = n | (n >> 1)
        n = n | (n >> 2)
        n = n | (n >> 4)
        n = n | (n >> 8)
        n = n | (n >> 16)
        n = n | (n >> 32)
        n += 1
        return n

    def __init__(self, positions, separation, pole_distance):
        self.positions = positions
        self.bounding_box = (*positions.min(0), *positions.max(0))
        self.separation = separation
        self.pole_distance = pole_distance
        indicies = positions
        indicies = (indicies - indicies.min(0, keepdims=True))
        indicies = (indicies / separation).astype(np.uint32)

        stride_x = np.ceil(
            (self.bounding_box[2] - self.bounding_box[0]) / self.separation
        ).astype(np.uint32)
        stride_y = np.ceil(
            (self.bounding_box[3] - self.bounding_box[1]) / self.separation
        ).astype(np.uint32)
        stride = max(stride_x, stride_y, 2)

        points = np.arange(len(positions))
        point_buckets = indicies[:, 0] * stride + indicies[:, 1]
        points_perm = np.argsort(point_buckets)
        points = points[points_perm]
        indicies = indicies[points_perm]
        point_buckets = point_buckets[points_perm]

        mask_not_first = np.zeros(shape=(len(points),), dtype=bool)
        mask_not_first[1:] = point_buckets[:-1] == point_buckets[1:]
        mask_first = np.logical_not(mask_not_first)
        point_firsts = np.argwhere(mask_first).ravel()
        point_ends = np.ones_like(point_firsts) * len(points)
        point_ends[:-1] = point_firsts[1:]

        p = self._find_power_of_two(stride)
        assert p > 1
        levels = []
        node_buckets = []
        node_levels = []
        node_firsts = []
        node_ends = []
        tree_depth = 0
        while p > 0:
            tree_depth += 1
            buckets = indicies[:, 0] * stride + indicies[:, 1]
            levels.append(buckets)
            indicies = indicies // 2
            p = p // 2
            node_buckets.append(np.unique_values(buckets))
            print(f"at power {p} added {len(node_buckets[-1])} buckets {buckets}")

        for i in range(len(levels)):
            if i == 0:
                assert len(point_firsts) == len(
                    node_buckets[0]
                ), f"{len(point_firsts), len(node_buckets[0])}"
                node_firsts.append(point_firsts)
                node_ends.append(point_ends)
            else:
                node_firsts.append(np.zeros_like(node_buckets[i]))
                node_ends.append(np.zeros_like(node_buckets[i]))
            node_levels.append(
                np.ones(shape=(len(node_buckets[i]),), dtype=np.uint32)
                * (len(levels) - i - 1)
            )
            print(f"at level {len(levels) - 1 - i} - we have {len(node_levels[i])} nodes")

        node_levels = np.concat(node_levels)
        node_buckets = np.concat(node_buckets)
        node_firsts = np.concat(node_firsts)
        node_ends = np.concat(node_ends)

        node_buckets = node_buckets * (len(levels) + 1) + node_levels
        node_perm = np.argsort(node_buckets)

        node_levels = node_levels[node_perm]
        node_buckets = node_buckets[node_perm]
        node_firsts = node_firsts[node_perm]
        node_ends = node_ends[node_perm]

        node_levels = node_buckets % (len(levels) + 1)
        node_xys = node_buckets // (len(levels) + 1)
        node_ys = node_xys % stride
        node_xs = node_xys // stride
        print(list(zip(node_levels.tolist(), node_ys.tolist(), node_xs.tolist())))

        def node_to_child(x_off, y_off):
            children = ((node_xs * 2 + x_off) * stride + (node_ys * 2 + y_off)) * (
                len(levels) + 1
            ) + (node_levels + 1)
            mask_valid = np.isin(children, node_buckets)
            mask_level = node_levels != (len(levels) - 1)
            mask = mask_valid & mask_level
            child_buckets = children[mask]
            child_indicies = np.argwhere(np.isin(node_buckets, child_buckets)).ravel()
            children[mask] = child_indicies
            children = np.where(mask, children, np.zeros_like(children))
            return children

        node_top_lefts = node_to_child(0, 0)
        node_top_rights = node_to_child(1, 0)
        node_bottom_lefts = node_to_child(0, 1)
        node_bottom_rights = node_to_child(1, 1)

        node_bbs = np.zeros(shape=(len(node_buckets), 4))
        for i in range(4):
            node_bbs[:, i] = self.bounding_box[[2, 3, 0, 1][i]]

        node_masses = np.zeros(shape=(len(node_buckets),))
        node_mass_centers = np.zeros(shape=(len(node_buckets), 2))

        node_firsts_i = node_firsts.copy()
        while True:
            mask_level = node_levels == (tree_depth - 1)
            mask_running = node_firsts_i != node_ends
            mask = np.logical_and(mask_level, mask_running)
            if np.count_nonzero(mask) == 0:
                break
            current_firsts = node_firsts_i[mask]
            current_bbs = node_bbs[mask]
            current_masses = node_masses[mask]
            current_mass_centers = node_mass_centers[mask]
            current_positions = np.take_along_axis(
                positions, current_firsts[:, None], axis=0
            )
            current_bbs[:, 0] = np.where(
                current_positions[:, 0] < current_bbs[:, 0],
                current_positions[:, 0],
                current_bbs[:, 0],
            )
            current_bbs[:, 1] = np.where(
                current_positions[:, 1] < current_bbs[:, 1],
                current_positions[:, 1],
                current_bbs[:, 1],
            )
            current_bbs[:, 2] = np.where(
                current_positions[:, 0] >= current_bbs[:, 2],
                current_positions[:, 0],
                current_bbs[:, 2],
            )
            current_bbs[:, 3] = np.where(
                current_positions[:, 1] >= current_bbs[:, 3],
                current_positions[:, 1],
                current_bbs[:, 3],
            )
            current_mass_centers = (
                current_mass_centers * current_masses[:, None] + current_positions
            ) / (current_masses[:, None] + 1)
            current_masses += 1
            current_firsts += 1
            node_firsts_i[mask] = current_firsts
            node_bbs[mask] = current_bbs
            node_masses[mask] = current_masses
            node_mass_centers[mask] = current_mass_centers

        for i in range(tree_depth - 1):
            for children in [
                node_top_lefts,
                node_top_rights,
                node_bottom_lefts,
                node_bottom_rights,
            ]:
                mask_level = node_levels == (tree_depth - 2 - i)
                mask_children = children != 0
                mask = np.logical_and(mask_level, mask_children)
                if np.count_nonzero(mask) == 0:
                    continue
                current_children = children[mask]
                current_child_bbs = np.take_along_axis(
                    node_bbs, current_children[:, None], axis=0
                )
                current_child_masses = np.take_along_axis(
                    node_masses, current_children, axis=0
                )
                current_child_mass_centers = np.take_along_axis(
                    node_mass_centers, current_children[:, None], axis=0
                )
                current_bbs = node_bbs[mask]
                current_masses = node_masses[mask]
                current_mass_centers = node_mass_centers[mask]

                current_bbs[:, 0] = np.where(
                    current_child_bbs[:, 0] < current_bbs[:, 0],
                    current_child_bbs[:, 0],
                    current_bbs[:, 0],
                )
                current_bbs[:, 1] = np.where(
                    current_child_bbs[:, 1] < current_bbs[:, 1],
                    current_child_bbs[:, 1],
                    current_bbs[:, 1],
                )
                current_bbs[:, 2] = np.where(
                    current_child_bbs[:, 2] >= current_bbs[:, 2],
                    current_child_bbs[:, 2],
                    current_bbs[:, 2],
                )
                current_bbs[:, 3] = np.where(
                    current_child_bbs[:, 3] >= current_bbs[:, 3],
                    current_child_bbs[:, 3],
                    current_bbs[:, 3],
                )
                current_mass_centers = (
                    current_mass_centers * current_masses[:, None]
                    + current_child_mass_centers * current_child_masses[:, None]
                ) / (current_masses[:, None] + current_child_masses[:, None])
                current_masses += current_child_masses
                node_bbs[mask] = current_bbs
                node_masses[mask] = current_masses
                node_mass_centers[mask] = current_mass_centers

        self.node_bbs = node_bbs
        self.node_top_lefts = node_top_lefts
        self.node_top_rights = node_top_rights
        self.node_bottom_lefts = node_bottom_lefts
        self.node_bottom_rights = node_bottom_rights
        self.node_masses = node_masses
        self.node_mass_centers = node_mass_centers
        self.node_levels = node_levels
        self.depth = tree_depth
        self.node_firsts = node_firsts
        self.node_ends = node_ends

    def get_collisions(self):
        deltas = np.zeros_like(self.positions)
        stack = np.zeros(shape=(len(deltas), 4 * self.depth), dtype=np.uint32)
        stack_depth = np.ones(shape=(len(deltas),), dtype=np.uint32)
        collisions = {}
        misses = {}
        while True:
            mask_stack = stack_depth != 0
            if np.count_nonzero(mask_stack) == 0:
                break
            current_deltas = deltas[mask_stack]
            current_positions = self.positions[mask_stack]
            current_stack = stack[mask_stack]
            current_stack_depth = stack_depth[mask_stack]
            current_stack_depth -= 1
            current_nodes = np.take_along_axis(
                current_stack, current_stack_depth[:, None], axis=-1
            ).ravel()
            current_bbs = np.take_along_axis(
                self.node_bbs, current_nodes[:, None], axis=0
            )
            current_top_lefts = np.take_along_axis(
                self.node_top_lefts, current_nodes, axis=0
            )
            current_top_rights = np.take_along_axis(
                self.node_top_rights, current_nodes, axis=0
            )
            current_bottom_lefts = np.take_along_axis(
                self.node_bottom_lefts, current_nodes, axis=0
            )
            current_bottom_rights = np.take_along_axis(
                self.node_bottom_rights, current_nodes, axis=0
            )
            current_firsts = np.take_along_axis(self.node_firsts, current_nodes, axis=0)
            current_ends = np.take_along_axis(self.node_ends, current_nodes, axis=0)
            mask_left = current_bbs[:, 0] < current_positions[:, 0] + self.separation
            mask_right = current_bbs[:, 2] >= current_positions[:, 0] - self.separation
            mask_bottom = current_bbs[:, 1] < current_positions[:, 1] + self.separation
            mask_top = current_bbs[:, 3] >= current_positions[:, 1] - self.separation
            mask_branch = np.logical_and(
                np.logical_and(mask_left, mask_right),
                np.logical_and(mask_bottom, mask_top),
            )

            def get_node_points(n):
                res = []
                if self.node_top_lefts[n] != 0:
                    res += get_node_points(self.node_top_lefts[n])
                if self.node_top_rights[n] != 0:
                    res += get_node_points(self.node_top_rights[n])
                if self.node_bottom_lefts[n] != 0:
                    res += get_node_points(self.node_bottom_lefts[n])
                if self.node_bottom_rights[n] != 0:
                    res += get_node_points(self.node_bottom_rights[n])
                for p in range(self.node_firsts[n], self.node_ends[n]):
                    res.append(p)
                return res

            ix = np.arange(len(self.positions))
            ix = ix[mask_stack]
            for i in range(len(ix)):
                if not mask_branch[i]:
                    # print(f"point - {ix[i]} not checking node {current_stack[i, current_stack_depth[i]]}, {get_node_points(current_stack[i, current_stack_depth[i]])}")
                    for p in get_node_points(current_stack[i, current_stack_depth[i]]):
                        misses[(ix[i], p)] = (
                            f"skipped node {current_stack[i, current_stack_depth[i]]}"
                        )
            skip_depth = current_stack_depth.copy()
            mask_top_left = current_top_lefts != 0
            np.put_along_axis(
                current_stack,
                current_stack_depth[:, None],
                current_top_lefts[:, None],
                axis=-1,
            )
            current_stack_depth[mask_top_left] += 1

            mask_top_right = current_top_rights != 0
            np.put_along_axis(
                current_stack,
                current_stack_depth[:, None],
                current_top_rights[:, None],
                axis=-1,
            )
            current_stack_depth[mask_top_right] += 1

            mask_bottom_left = current_bottom_lefts != 0
            np.put_along_axis(
                current_stack,
                current_stack_depth[:, None],
                current_bottom_lefts[:, None],
                axis=-1,
            )
            current_stack_depth[mask_bottom_left] += 1

            mask_bottom_right = current_bottom_rights != 0
            np.put_along_axis(
                current_stack,
                current_stack_depth[:, None],
                current_bottom_rights[:, None],
                axis=-1,
            )
            current_stack_depth[mask_bottom_right] += 1

            current_firsts_i = current_firsts.copy()
            while True:
                mask_running = current_firsts_i < current_ends
                mask = np.logical_and(mask_running, mask_branch)
                if np.count_nonzero(mask) == 0:
                    break
                step_firsts = current_firsts_i[mask]
                step_query_positions = np.take_along_axis(
                    self.positions, step_firsts[:, None], axis=0
                )
                step_positions = current_positions[mask]
                step_overlap = step_query_positions - step_positions
                step_distances = np.linalg.norm(step_overlap, axis=-1)
                step_deltas = current_deltas[mask]
                mask_collision = step_distances < self.separation
                scale = np.where(
                    step_distances < 1e-5, np.ones_like(step_distances), step_distances
                )
                step_deltas -= np.where(
                    np.logical_and(step_distances >= 1e-5, mask_collision)[:, None],
                    step_overlap
                    / scale[:, None]
                    * (self.separation - step_distances)[:, None]
                    / 2,
                    np.zeros_like(step_overlap),
                )
                ix = np.arange(len(self.positions))
                ix = ix[mask_stack]
                ix = ix[mask]
                for i in range(len(ix)):
                    if mask_collision[i] and step_firsts[i] != ix[i]:
                        collisions[(ix[i].item(), step_firsts[i].item())] = (
                            step_distances[i].item()
                        )
                    elif not mask_collision[i]:
                        misses[(ix[i].item(), step_firsts[i].item())] = (
                            f"mask collision"
                        )
                    if (
                        not mask_collision[i]
                        or step_firsts[i] == ix[i]
                        or step_distances[i] > 0.016
                    ):
                        continue
                    # print(ix[i], step_firsts[i], step_distances[i], step_deltas[i])
                step_firsts += 1
                current_firsts_i[mask] = step_firsts
                current_deltas[mask] = step_deltas

            current_stack_depth = np.where(mask_branch, current_stack_depth, skip_depth)

            stack_depth[mask_stack] = current_stack_depth
            stack[mask_stack] = current_stack
            deltas[mask_stack] = current_deltas
        for k, v in collisions.items():
            k_ = (k[1], k[0])
            if k_ not in collisions:
                for i in range(len(self.node_levels)):
                    print(
                        i,
                        self.node_top_lefts[i],
                        self.node_top_rights[i],
                        self.node_bottom_lefts[i],
                        self.node_bottom_rights[i],
                        get_node_points(i),
                    )
                raise RuntimeError(f"{k,v} - {misses.get(k_, None)}")
        return deltas

    def get_gravity(self):
        pass


class Simulation:

    def __init__(self):
        self.positions = (
            (np.random.rand(number_of_points, 2) * 2 - 1)
            * (1 - separation)
            * spawn_radius
        )
        self.positions = self.positions.astype(np.float32)
        self.prev_positions = np.zeros_like(self.positions)
        for _ in range(substeps):
            quad_tree = QuadTreeV2(self.positions, separation * 2, pole_distance)
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
            quad_tree = QuadTree(
                (*self.positions.min(0), *self.positions.max(0)),
                separation * 2,
                pole_distance,
            )
            for i in range(number_of_points):
                quad_tree.add_point(i, self.positions[i])
            accel = np.zeros(shape=(number_of_points, 2), dtype=np.float32)
            for i in range(number_of_points):
                accel[i] = quad_tree.get_gravity(self.positions[i])
            self.positions, self.prev_positions = (
                2 * self.positions - self.prev_positions + dt**2 * accel,
                self.positions,
            )
            quad_tree = QuadTreeV2(self.positions, separation * 2, pole_distance)
            deltas = quad_tree.get_collisions()
            self.positions += deltas

        e_kin = 0
        for i in range(number_of_points):
            e_kin += (
                np.linalg.norm(self.positions[i] - self.prev_positions[i]) / dt
            ) ** 2 / 2

        self.recenter_view()
        if self.frame_count > 0:
            self.frame_count -= 1
            return self.positions
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
