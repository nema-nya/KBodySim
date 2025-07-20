import dataclasses
import numpy as np

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
                accel += r / r_len**3 * self.gravitational_constant * node.mass
                continue
            elif bb[1] >= position[1] + self.pole_distance:
                accel += r / r_len**3 * self.gravitational_constant * node.mass
                continue
            elif bb[2] < position[0] - self.pole_distance:
                accel += r / r_len**3 * self.gravitational_constant * node.mass
                continue
            elif bb[3] < position[1] - self.pole_distance:
                accel += r / r_len**3 * self.gravitational_constant * node.mass
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
                accel += r / r_len**3 * self.gravitational_constant
        return accel

    def get_potential(self, position, gravitational_constant):
        pot = 0
        self.gravitational_constant = gravitational_constant
        queue = [self.root]
        while queue:
            node = queue.pop()
            bb = node.bounding_box
            r = node.mass_center - position
            r_len = np.linalg.norm(r)
            if bb[0] >= position[0] + self.pole_distance:
                pot += self.gravitational_constant * node.mass / r_len
                continue
            elif bb[1] >= position[1] + self.pole_distance:
                pot += self.gravitational_constant * node.mass / r_len
                continue
            elif bb[2] < position[0] - self.pole_distance:
                pot += self.gravitational_constant * node.mass / r_len
                continue
            elif bb[3] < position[1] - self.pole_distance:
                pot += self.gravitational_constant * node.mass / r_len
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
                pot += self.gravitational_constant / r_len
        return -pot
