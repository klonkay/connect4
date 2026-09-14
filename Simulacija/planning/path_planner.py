"""
planning/path_planner.py
Builds full pick and place waypoint sequences (each waypoint a UR 6-vector
[x, y, z, rx, ry, rz] in the robot base frame) from a chip's position/
normal and the tool/robot dimensions in config.py. All standoffs are
pulled from config, so re-measuring TOOL_* values is the only thing you
need to do to keep every generated path valid.
"""

import numpy as np
import config
from planning.geometry_utils import pose_from_position_normal


def _effective_travel_height():
    """The height actually used for lateral ('travel') legs -- whichever
    is larger of the tool's usual safe travel height, or enough to clear
    the board's known raised structure (config.BOARD_OBSTRUCTION_HEIGHT)
    plus config.ROBOT_BODY_CLEARANCE_MARGIN, a generous allowance since
    the arm's own body (not just the tool tip) needs room, and this
    module has no model of the arm's actual shape. This is a simple 'fly
    high enough over the whole board' heuristic, not full obstacle-aware
    whole-arm planning -- see config.ROBOT_BODY_CLEARANCE_MARGIN's
    docstring for the real limitation here. Centralizing it means
    bumping either config value raises every travel leg automatically.
    """
    return max(config.TOOL_SAFE_TRAVEL_HEIGHT,
               config.BOARD_OBSTRUCTION_HEIGHT + config.ROBOT_BODY_CLEARANCE_MARGIN)


def _effective_approach_height():
    """The height used for the vertical hover directly above a specific
    chip/slot before contact. This also can't be less than the board's
    known obstruction height (plus a small buffer) -- a misplaced chip
    can easily sit right next to a raised wall or divider, so the
    straight vertical descent onto it can clip that nearby structure
    even though the lateral travel leg was already safely above
    everything. Without this, TOOL_APPROACH_CLEARANCE alone only
    protects against the chip's own local geometry, not anything next
    to it.
    """
    return max(config.TOOL_APPROACH_CLEARANCE,
               config.BOARD_OBSTRUCTION_HEIGHT + 0.005)


def build_pick_path(chip_position_base, chip_normal_base):
    """
    chip_position_base : np.array xyz, base frame, on the chip's face surface
    chip_normal_base    : np.array xyz unit vector, base frame, pointing
                          away from the chip's face (out of the pile)

    Returns named 6-vector poses:
        travel   -- safe height above the chip AND the board's known
                    obstructions, used for fast lateral moves
        approach -- hover above the chip, clearing nearby board
                    obstructions too, not just the chip's own geometry
        contact  -- where the cup actually seals, accounting for compression
        retreat  -- straight back off along the normal after suctioning
    """
    n = chip_normal_base / np.linalg.norm(chip_normal_base)

    travel_point = chip_position_base + n * _effective_travel_height()
    approach_point = chip_position_base + n * _effective_approach_height()
    # Contact point is pulled slightly *into* the nominal surface point to
    # account for cup compression, so the cup actually seals rather than
    # hovering just above the chip.
    contact_point = chip_position_base - n * config.TOOL_CUP_COMPRESSION
    retreat_point = approach_point.copy()

    return {
        "travel":   pose_from_position_normal(travel_point, n),
        "approach": pose_from_position_normal(approach_point, n),
        "contact":  pose_from_position_normal(contact_point, n),
        "retreat":  pose_from_position_normal(retreat_point, n),
    }


def build_place_path(place_position_base, surface_normal_base):
    """
    Builds waypoints for releasing onto the destination surface.

    place_position_base : np.array xyz, robot base frame -- the exact
                           point on the surface to release the chip at
                           (already includes the board's calibrated
                           height via PlacementManager.next_slot)
    surface_normal_base : np.array xyz unit vector, robot base frame --
                           the surface's true orientation, from
                           PlacementManager.surface_normal_base. Straight
                           up for a flat board on a level table; tilted
                           if the board itself sits at an angle.
    """
    n = surface_normal_base / np.linalg.norm(surface_normal_base)
    place_point = np.asarray(place_position_base, dtype=float)

    travel_point = place_point + n * _effective_travel_height()
    approach_point = place_point + n * _effective_approach_height()
    # Release a little above full contact (half the pick-side compression
    # offset) so the chip is set down gently rather than pressed flat.
    release_point = place_point + n * (config.TOOL_CUP_COMPRESSION * 0.5)

    return {
        "travel":   pose_from_position_normal(travel_point, n),
        "approach": pose_from_position_normal(approach_point, n),
        "release":  pose_from_position_normal(release_point, n),
        "retreat":  pose_from_position_normal(approach_point, n),
    }
