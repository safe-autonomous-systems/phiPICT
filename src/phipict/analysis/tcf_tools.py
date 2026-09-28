# Original work Copyright 2025 Aleksandra Franz, Nils Thuerey
# Modified work Copyright 2026 Jannis Becktepe
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications:
# Reduced to the channel-flow unit conversion helpers.
# Moved into the phipict package, formatted, linted and typed.

"""Unit conversion helpers for turbulent channel flow (TCF)."""

from typing import TypeVar

import numpy as np
import torch

# Quantities that may be given as a scalar, a NumPy array or a torch tensor.
T = TypeVar("T", float, np.ndarray, torch.Tensor)


def Re_cl_to_wall(Re_cl: float) -> float:
    """Convert a centerline Reynolds number to a wall (friction) Reynolds number.

    Parameters
    ----------
    Re_cl : float
        Centerline Reynolds number.

    Returns
    -------
    float
        Wall Reynolds number.
    """
    return 0.116 * (Re_cl**0.88)


def Re_wall_to_cl(Re_wall: float) -> float:
    """Convert a wall (friction) Reynolds number to a centerline Reynolds number.

    Parameters
    ----------
    Re_wall : float
        Wall Reynolds number.

    Returns
    -------
    float
        Centerline Reynolds number.
    """
    return (Re_wall / 0.116) ** (1 / 0.88)


def t_to_ETT(t: T, u_wall: float, delta: float = 1) -> T:
    """Convert a time to eddy turnover times.

    Parameters
    ----------
    t : float, numpy.ndarray or torch.Tensor
        Time in simulation units.
    u_wall : float
        Wall friction velocity.
    delta : float, optional
        Channel half-height. Default is 1.

    Returns
    -------
    float, numpy.ndarray or torch.Tensor
        Time in eddy turnover times.
    """
    return t * u_wall / delta


def ETT_to_t(ETT: T, u_wall: float, delta: float = 1) -> T:
    """Convert eddy turnover times to a time.

    Parameters
    ----------
    ETT : float, numpy.ndarray or torch.Tensor
        Time in eddy turnover times.
    u_wall : float
        Wall friction velocity.
    delta : float, optional
        Channel half-height. Default is 1.

    Returns
    -------
    float, numpy.ndarray or torch.Tensor
        Time in simulation units.
    """
    return ETT * delta / u_wall


def t_star(visc: float, u_wall: float) -> float:
    """Compute the viscous time scale.

    Parameters
    ----------
    visc : float
        Kinematic viscosity.
    u_wall : float
        Wall friction velocity.

    Returns
    -------
    float
        Viscous time scale ``visc / u_wall**2``.
    """
    return visc / (u_wall**2)


def t_to_t_wall(t: T, visc: float, u_wall: float) -> T:
    """Convert a time to wall units.

    Parameters
    ----------
    t : float, numpy.ndarray or torch.Tensor
        Time in simulation units.
    visc : float
        Kinematic viscosity.
    u_wall : float
        Wall friction velocity.

    Returns
    -------
    float, numpy.ndarray or torch.Tensor
        Time in wall units.
    """
    return t / t_star(visc, u_wall)


def t_wall_to_t(t_wall: T, visc: float, u_wall: float) -> T:
    """Convert a time in wall units to simulation units.

    Parameters
    ----------
    t_wall : float, numpy.ndarray or torch.Tensor
        Time in wall units.
    visc : float
        Kinematic viscosity.
    u_wall : float
        Wall friction velocity.

    Returns
    -------
    float, numpy.ndarray or torch.Tensor
        Time in simulation units.
    """
    return t_wall * t_star(visc, u_wall)


def vel_to_vel_wall(vel: T, u_wall: float, order: int = 1) -> T:
    """Convert a velocity (moment) to wall units.

    Parameters
    ----------
    vel : float, numpy.ndarray or torch.Tensor
        Velocity or velocity moment in simulation units.
    u_wall : float
        Wall friction velocity.
    order : int, optional
        Order of the moment, e.g. 2 for Reynolds stresses. Default is 1.

    Returns
    -------
    float, numpy.ndarray or torch.Tensor
        Velocity (moment) in wall units.
    """
    return vel * (1 / (u_wall**order))


def pos_to_pos_wall(pos: T, viscosity: float, u_wall: float) -> T:
    """Convert a position to wall units.

    Parameters
    ----------
    pos : float, numpy.ndarray or torch.Tensor
        Position in simulation units.
    viscosity : float
        Kinematic viscosity.
    u_wall : float
        Wall friction velocity.

    Returns
    -------
    float, numpy.ndarray or torch.Tensor
        Position in wall units.
    """
    return pos * ((1 / viscosity) * u_wall)
