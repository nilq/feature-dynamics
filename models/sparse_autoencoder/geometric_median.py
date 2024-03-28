"""Computing geometric median."""

import tqdm
import torch


def weighted_average(points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Calculate the weighted average of points.
    
    Args:
        points (torch.Tensor): Points to get median of.
        weights (torch.Tensor, optional): Optional weights.
    
    Returns:
        torch.Tensor: Weighted average.
    """
    normalized_weights = weights / weights.sum()
    return (points * normalized_weights.unsqueeze(-1)).sum(dim=0)

def geometric_median_objective(median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor):
    """Calculate the weighted sum of distances from the median to each point.

    Args:
        median (torch.Tensor): Current median estimate.
        points (torch.Tensor): Points to get median of.
        weights (torch.Tensor, optional): Optional weights.

    Returns:
        torch.Tensor: Geometric median objective.
    """
    distances = torch.linalg.norm(points - median.unsqueeze(0), dim=1)
    return (distances * weights).sum()


def geometric_median(points: torch.Tensor, weights: torch.Tensor | None = None, eps: float = 1e-6, max_iter: int = 100, fractional_improvement_tolerance: float =1e-20) -> torch.Tensor:
    """Get geometric median of points using Weiszfeld algorithm.

    Args:
        points (torch.Tensor): Points to get median of.
        weights (torch.Tensor | None, optional): Optional weights. Defaults to None.
        eps (float, optional): Low epsilon value. Defaults to 1e-6.
        max_iter (int, optional): Maximum iterations to find geometric median. Defaults to 100.
        fractional_improvement_tolerance (float, optional): Fractional improvement tolerance for early stopping. Defaults to 1e-20.

    Returns:
        torch.Tensor: Geometric median of points.
    """
    if weights is None:
        weights = torch.ones(points.size(0), device=points.device)

    median = weighted_average(points=points, weights=weights)
    objective_value = geometric_median_objective(median=median, points=points, weights=weights)

    for _ in tqdm.tqdm(range(max_iter), desc="Computing geometric median"):
        distances = torch.linalg.norm(points - median.unsqueeze(0), dim=1)
        adjusted_weights = weights / torch.clamp(distances, min=eps)
        updated_median = weighted_average(points, adjusted_weights)

        new_objective_value = geometric_median_objective(median=updated_median, points=points, weights=weights)

        if abs(objective_value - new_objective_value) <= fractional_improvement_tolerance * objective_value:
            break

        median, objective_value = updated_median, new_objective_value

    return median
