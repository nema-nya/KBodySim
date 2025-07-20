import numpy as np


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

    def __init__(self, positions, separation, pole_distance, gravitational_constant):
        self.positions = positions
        self.bounding_box = (*positions.min(0), *positions.max(0))
        self.separation = separation
        self.pole_distance = pole_distance
        self.gravitational_constant = gravitational_constant
        indicies = positions
        indicies = indicies - indicies.min(0, keepdims=True)
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
                step_firsts += 1
                current_firsts_i[mask] = step_firsts
                current_deltas[mask] = step_deltas

            current_stack_depth = np.where(mask_branch, current_stack_depth, skip_depth)

            stack_depth[mask_stack] = current_stack_depth
            stack[mask_stack] = current_stack
            deltas[mask_stack] = current_deltas
        return deltas

    def get_gravity(self):
        accels = np.zeros_like(self.positions)
        stack = np.zeros(shape=(len(accels), 4 * self.depth), dtype=np.uint32)
        stack_depth = np.ones(shape=(len(accels),), dtype=np.uint32)
        while True:
            mask_stack = stack_depth != 0
            if np.count_nonzero(mask_stack) == 0:
                break
            current_accels = accels[mask_stack]
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
            current_masses = np.take_along_axis(self.node_masses, current_nodes, axis=0)
            current_mass_centers = np.take_along_axis(
                self.node_mass_centers, current_nodes[:, None], axis=0
            )
            current_firsts = np.take_along_axis(self.node_firsts, current_nodes, axis=0)
            current_ends = np.take_along_axis(self.node_ends, current_nodes, axis=0)
            mask_left = current_bbs[:, 0] < current_positions[:, 0] + self.pole_distance
            mask_right = (
                current_bbs[:, 2] >= current_positions[:, 0] - self.pole_distance
            )
            mask_bottom = (
                current_bbs[:, 1] < current_positions[:, 1] + self.pole_distance
            )
            mask_top = current_bbs[:, 3] >= current_positions[:, 1] - self.pole_distance
            mask_branch = np.logical_and(
                np.logical_and(mask_left, mask_right),
                np.logical_and(mask_bottom, mask_top),
            )

            r = current_mass_centers - current_positions
            r_len = np.linalg.norm(r, axis=-1)
            mask_gravity = r_len >= self.separation
            scale = np.where(r_len < 1e-5, np.ones_like(r_len), r_len)
            skip_accels = current_accels.copy()
            skip_accels += np.where(
                mask_gravity[:, None],
                r
                / scale[:, None] ** 3
                * self.gravitational_constant
                * current_masses[:, None],
                np.zeros_like(current_accels),
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
                r = step_query_positions - step_positions
                r_len = np.linalg.norm(r, axis=-1)
                step_accels = current_accels[mask]
                mask_gravity = r_len > self.separation
                scale = np.where(r_len <= self.separation, np.ones_like(r_len), r_len)
                step_accels += np.where(
                    mask_gravity[:, None],
                    r / scale[:, None] ** 3 * self.gravitational_constant,
                    np.zeros_like(step_accels),
                )
                step_firsts += 1
                current_firsts_i[mask] = step_firsts
                current_accels[mask] = step_accels

            current_stack_depth = np.where(mask_branch, current_stack_depth, skip_depth)
            current_accels = np.where(mask_branch[:, None], current_accels, skip_accels)

            stack_depth[mask_stack] = current_stack_depth
            stack[mask_stack] = current_stack
            accels[mask_stack] = current_accels
        return accels
