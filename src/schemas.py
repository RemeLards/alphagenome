from pydantic import BaseModel, Field
from typing import Any


# ─── Request ────────────────────────────────────────────────────────────────

class IntervalSchema(BaseModel):
    chromosome: str
    start: int
    end: int
    strand: str = "."


class VariantSchema(BaseModel):
    chromosome: str
    position: int
    reference_bases: str = Field(min_length=1, max_length=1)
    alternate_bases: str = Field(min_length=1, max_length=1)


class PredictVariantRequest(BaseModel):
    interval: IntervalSchema
    variant: VariantSchema
    ontology_terms: list[str] = ["UBERON:0001157"]
    requested_outputs: list[str] = ["RNA_SEQ"]


class PredictVariantBatchRequest(BaseModel):
    intervals: list[IntervalSchema]
    variants: list[VariantSchema]
    ontology_terms: list[str] = ["UBERON:0001157"]
    requested_outputs: list[str] = ["RNA_SEQ"]
    batch_size: int | None = Field(default=None, ge=1)
    max_workers: int | None = Field(default=None, ge=1)


class PredictIntervalRequest(BaseModel):
    interval: IntervalSchema
    ontology_terms: list[str] = ["UBERON:0001157"]
    requested_outputs: list[str] = ["RNA_SEQ"]


class PredictIntervalBatchRequest(BaseModel):
    intervals: list[IntervalSchema]
    ontology_terms: list[str] = ["UBERON:0001157"]
    requested_outputs: list[str] = ["RNA_SEQ"]
    batch_size: int | None = Field(default=None, ge=1)
    max_workers: int | None = Field(default=None, ge=1)


class PredictSequenceBatchRequest(BaseModel):
    sequences: list[str]
    ontology_terms: list[str] = ["UBERON:0001157"]
    requested_outputs: list[str] = ["RNA_SEQ"]
    batch_size: int | None = Field(default=None, ge=1)
    max_workers: int | None = Field(default=None, ge=1)


# ─── Response ───────────────────────────────────────────────────────────────

class TrackDataSchema(BaseModel):
    values: list[list[float]]
    names: list[str]
    resolution: int = 1
    interval: IntervalSchema | None = None


class VariantOutputSchema(BaseModel):
    reference: dict[str, TrackDataSchema | None]
    alternate: dict[str, TrackDataSchema | None]


class PredictIntervalResponse(BaseModel):
    outputs: dict[str, TrackDataSchema | None]


class PredictIntervalBatchResponse(BaseModel):
    outputs: list[dict[str, TrackDataSchema | None]]


class PredictVariantResponse(BaseModel):
    variant_outputs: list[VariantOutputSchema]


class MetricsSchema(BaseModel):
    runtime: float
    batch_size: int


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_name: str
