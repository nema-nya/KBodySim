from .compute_point_tile_phase import ComputePointTilePhase
from .tile_sort_phase import TileSortPhase
from .compute_quad_tree_phase import ComputeQuadTreePhase
from .compute_impulses_phase import ComputeImpulsesPhase
from .apply_impulses_phase import ApplyImpulsesPhase


class CollisionsPhase:
    def __init__(self, device, number_of_points, simulation):
        self.device = device
        self.number_of_points = number_of_points
        self.simulation = simulation

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

        self.compute_impulses_phase = ComputeImpulsesPhase(
            self.number_of_points,
            self.device,
            self.simulation.separation,
        )

        self.apply_impulses_phase = ApplyImpulsesPhase(self.device)

    def compute_pass(self, command_encoder, positions_buffer, prev_positions_buffer):
        self.compute_point_tile_phase.compute_pass(
            self.device,
            positions_buffer,
            command_encoder,
            self.simulation.separation,
        )
        k = 2
        while k <= self.number_of_points:
            j = k // 2
            while j > 0:
                self.tile_sort_phase[(k, j)].compute_pass(
                    self.device,
                    positions_buffer,
                    prev_positions_buffer,
                    self.compute_point_tile_phase.tiles_buffer,
                    command_encoder,
                    self.number_of_points,
                    self.simulation.separation,
                    k,
                    j,
                )
                j = j // 2
            k = k * 2

        self.compute_quad_tree_phase.compute_pass(
            self.device,
            positions_buffer,
            command_encoder,
            self.number_of_points,
            self.simulation.separation,
        )

        self.compute_impulses_phase.compute_pass(
            self.device,
            positions_buffer,
            command_encoder,
            self.number_of_points,
            self.compute_quad_tree_phase.nodes_buffer,
        )

        self.apply_impulses_phase.compute_pass(
            self.device,
            positions_buffer,
            self.compute_impulses_phase.impulses_buffer,
            command_encoder,
            self.number_of_points,
            prev_positions_buffer,
        )
