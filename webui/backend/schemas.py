"""Pydantic request/response schemas for the WebUI REST API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class StartRunRequest(BaseModel):
    task: str = "co2rr"
    data_path: str = "data/data.csv"
    candidates_path: Optional[str] = "data/candidates.csv"
    target_product: Optional[str] = None
    top_k: int = 10
    beta: float = 2.0
    beta_schedule: str = "fixed"
    beta_delta: float = 0.1
    acq_strategy: str = "ucb"
    qnei_mc_samples: int = 128
    kernel_type: str = "matern"
    h2_penalty_weight: float = 0.0
    skip_feature_selection: bool = False
    strict_training_schema: bool = False
    pre_fill_before_choice: bool = False
    seed: Optional[int] = None
    device: str = "auto"
    iterations: int = 10
    interactive: bool = True
    kabo_mode: bool = False
    lambda_p: float = 1.0
    lambda_k: float = 1.0
    lambda_v: float = 0.0
    expert_prior_file: Optional[str] = None
    diversity_weight: float = 0.5
    pe_budget: int = 0
    generate_candidates_n: int = 1000
    prefer_file_candidates: bool = False
    discrete_strategy: str = "acq"


class AnswerRequest(BaseModel):
    prompt_id: Optional[int] = None
    action: Optional[str] = Field(
        default=None,
        description="submit|manual|tie|exit|raw (depending on prompt kind)",
    )
    rank: Optional[int] = None
    values: Optional[dict[str, Any]] = None
    yields: Optional[dict[str, Any]] = None
    oob_confirmations: Optional[int] = 0
    value: Optional[str] = None  # for raw_input


class SaveFileRequest(BaseModel):
    content: str


class SaveJsonFileRequest(BaseModel):
    content: dict | list


class StatusResponse(BaseModel):
    status: str
    run_id: Optional[str] = None
    pending_prompt: Optional[dict] = None
    error: Optional[str] = None
