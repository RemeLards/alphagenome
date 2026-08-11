from time import time

from fastapi import APIRouter, HTTPException

from src.schemas import (
    PredictVariantRequest,
    PredictVariantBatchRequest,
    PredictIntervalRequest,
    PredictIntervalBatchRequest,
    PredictSequenceBatchRequest,
    PredictVariantResponse,
    PredictIntervalResponse,
    PredictIntervalBatchResponse,
    MetricsSchema,
    HealthResponse,
)

from src.model import alphagenome_model

from src.config import settings

router = APIRouter(prefix="/v1", tags=["predict"])


def _validate_interval_length(interval):
    if settings.WINDOW_SWEEP:
        return
    if abs(interval.start - interval.end) != settings.SEQUENCE_LEN:
        raise HTTPException(
            status_code=422,
            detail="interval length differs from sequence_len",
        )


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=alphagenome_model.is_loaded,
        model_name=alphagenome_model.model_name,
    )


@router.post("/predict/variant", response_model=PredictVariantResponse)
def predict_variant(req: PredictVariantRequest):
    _validate_interval_length(req.interval)

    try:
        result = alphagenome_model.predict_variant(
            interval=req.interval,
            variant=req.variant,
            ontology_terms=req.ontology_terms,
            requested_outputs=req.requested_outputs,
        )
        return PredictVariantResponse(variant_outputs=[result])
    except Exception as e:
        print(f"Erro = {e} -> {type(e).__name__}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/variants", response_model=PredictVariantResponse)
def predict_variants_batch(req: PredictVariantBatchRequest):
    if len(req.intervals) != len(req.variants):
        raise HTTPException(
            status_code=422,
            detail="intervals and variants must have the same length",
        )
    for interval in req.intervals:
        _validate_interval_length(interval)
    try:
        results = alphagenome_model.predict_variants_batch(
            intervals=req.intervals,
            variants=req.variants,
            ontology_terms=req.ontology_terms,
            requested_outputs=req.requested_outputs,
            max_workers=req.max_workers,
            batch_size=req.batch_size,
        )
        return PredictVariantResponse(variant_outputs=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/interval", response_model=PredictIntervalResponse)
def predict_interval(req: PredictIntervalRequest):
    _validate_interval_length(req.interval)
    try:
        outputs = alphagenome_model.predict_interval(
            interval=req.interval,
            ontology_terms=req.ontology_terms,
            requested_outputs=req.requested_outputs,
        )
        return PredictIntervalResponse(outputs=outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/intervals", response_model=PredictIntervalBatchResponse)
def predict_intervals_batch(req: PredictIntervalBatchRequest):
    for interval in req.intervals:
        _validate_interval_length(interval)
    try:
        outputs = alphagenome_model.predict_intervals_batch(
            intervals=req.intervals,
            ontology_terms=req.ontology_terms,
            requested_outputs=req.requested_outputs,
            max_workers=req.max_workers,
            batch_size=req.batch_size,
        )
        return PredictIntervalBatchResponse(outputs=outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/sequences", response_model=PredictIntervalBatchResponse)
def predict_sequences_batch(req: PredictSequenceBatchRequest):
    sequence_lengths = {len(sequence) for sequence in req.sequences}
    if len(sequence_lengths) > 1:
        raise HTTPException(
            status_code=422,
            detail="sequences must have the same length",
        )
    if not settings.WINDOW_SWEEP and sequence_lengths:
        sequence_len = next(iter(sequence_lengths))
        if sequence_len != settings.SEQUENCE_LEN:
            raise HTTPException(
                status_code=422,
                detail="sequence length differs from sequence_len",
            )
    try:
        outputs = alphagenome_model.predict_sequences_batch(
            sequences=req.sequences,
            ontology_terms=req.ontology_terms,
            requested_outputs=req.requested_outputs,
            max_workers=req.max_workers,
            batch_size=req.batch_size,
        )
        return PredictIntervalBatchResponse(outputs=outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
