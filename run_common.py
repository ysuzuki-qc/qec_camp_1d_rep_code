from typing import Literal
from pydantic import BaseModel
from dataclasses import field


class Result(BaseModel):
    code_distance: int
    num_round: int
    initial_state: int
    num_shot: int
    failure_count: int
    seed: int = 0
    expected_uniform_error_rate: float = 0.


class ResultSet(BaseModel):
    method_noise_model: Literal["unknown", "uniform", "characterization"]
    method_sampling: Literal["unknown", "simulation", "experiment"]
    result_list: list[Result] = field(default_factory=list)
