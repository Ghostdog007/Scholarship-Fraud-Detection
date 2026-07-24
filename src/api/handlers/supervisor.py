"""
Supervisor feedback endpoints — ADR-005.

POST /v3/supervisor/confirm-fraud
POST /v3/supervisor/mark-false-positive
GET  /v3/supervisor/patterns
POST /v3/supervisor/patterns/confirm
POST /v3/supervisor/patterns/promote
"""
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    ConfirmFraudRequest,
    FalsePositiveRequest,
    ClearLabelRequest,
    ConfirmPatternRequest,
    PromotePatternRequest,
    ConfirmBatchRequest,
    PatternTestRequest,
    PatternIngestRequest,
    DeletePatternRequest,
    RetrainPatternsRequest,
)

log    = structlog.get_logger()
router = APIRouter()

_VALID_RETRAIN_MODES = {"incremental", "full_retrain"}


def _dispatch_retrain(mode: str, cycle: str, smoke_test: bool) -> str:
    """Queue the chosen retrain job and return its job_id. 'incremental' fine-
    tunes the existing checkpoint (RGCN frozen unless >=50 confirmed fraud) —
    it does NOT read synthetic_exposure_graph_v3.pt, so a promoted pattern's
    real topology sits unused until a full retrain runs. 'full_retrain' reruns
    main_v3.py end-to-end, which trains hybrid_graphmcm_v3 Stage 1 LOE against
    that topology-exposure file directly — the only path that actually teaches
    the RGCN the ring's structure rather than just its feature values."""
    import uuid
    job_id = f"job_retrain_{uuid.uuid4().hex[:8]}"
    if mode == "full_retrain":
        from src.api.tasks import run_full_pipeline_task
        run_full_pipeline_task.apply_async(kwargs=dict(smoke_test=smoke_test), task_id=job_id)
    else:
        from src.api.tasks import run_incremental_task
        run_incremental_task.apply_async(kwargs=dict(cycle=cycle, smoke_test=smoke_test), task_id=job_id)
    return job_id


@router.post("/confirm-fraud")
def confirm_fraud(req: ConfirmFraudRequest):
    from src.confirmed_fraud_store import add_confirmed, load_confirmed
    try:
        add_confirmed(
            app_id=req.application_id,
            fraud_type=req.fraud_type,
            confirmed_by=req.confirmed_by,
            notes=req.notes,
            cycle=req.cycle,
        )
        n_confirmed = len(load_confirmed())
        log.info(
            "supervisor.confirm_fraud",
            app_id=req.application_id,
            fraud_type=req.fraud_type,
            n_confirmed=n_confirmed,
        )
        return {
            "status": "ok",
            "application_id": req.application_id,
            "fraud_type": req.fraud_type,
            "n_confirmed": n_confirmed,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/mark-false-positive")
def mark_false_positive(req: FalsePositiveRequest):
    from src.confirmed_fraud_store import add_false_positive, load_false_positive_ids
    try:
        add_false_positive(
            app_id=req.application_id,
            confirmed_by=req.confirmed_by,
            notes=req.notes,
        )
        n_fp = len(load_false_positive_ids())
        log.info(
            "supervisor.mark_false_positive",
            app_id=req.application_id,
            n_false_positives=n_fp,
        )
        return {
            "status": "ok",
            "application_id": req.application_id,
            "n_false_positives": n_fp,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/clear-label")
def clear_label(req: ClearLabelRequest):
    """Undo a supervisor label — removes the application from the confirmed and
    false-positive stores so its state resets. Enables re-demoing the detection
    loop (label a topology → retrain → detected → clear → repeat) and correcting
    mis-clicks. 404 if the application had no label to clear."""
    from src.confirmed_fraud_store import remove_label
    result = remove_label(req.application_id)
    if not (result["removed_confirmed"] or result["removed_false_positive"]):
        raise HTTPException(
            status_code=404,
            detail=f"No label to clear for '{req.application_id}' (not in confirmed or false-positive store)",
        )
    log.info("supervisor.clear_label", app_id=req.application_id, **result)
    return {"status": "ok", "application_id": req.application_id, **result}


@router.post("/confirm-batch")
def confirm_batch(req: ConfirmBatchRequest):
    """Record supervisor verdicts for a reviewer-selected batch, then optionally
    dispatch ONE incremental fine-tune over the (now label-augmented) data.

    Each item is either a confirmed-fraud (needs fraud_type — one of
    confirmed_fraud_store.VALID_FRAUD_TYPES) or a false-positive. Per-item errors
    (unknown fraud_type, id not in features, unknown verdict) are collected and
    reported without aborting the whole batch; already-labelled ids are recorded
    as 'skipped'. Confirmed examples flow into the SAME confirmed_fraud_store the
    incremental trainer reads (3x LightGBM weight, RGCN frozen) — this endpoint
    builds no ad-hoc training set.

    Training is dispatched ONLY when dispatch_retrain is true (the "retrain on
    selected" action). Recording labels alone never advances training — hard
    stop #5 (no automatic self-training rounds) is respected: the reviewer's
    explicit click is the gate.
    """
    from src.confirmed_fraud_store import (
        add_confirmed, add_false_positive, load_confirmed, load_false_positive_ids,
    )

    if not req.items:
        raise HTTPException(status_code=400, detail="items is empty — nothing to label")

    recorded, errors = [], []
    for item in req.items:
        aid = item.application_id
        try:
            if item.verdict == "confirmed_fraud":
                add_confirmed(
                    app_id=aid, fraud_type=item.fraud_type,
                    confirmed_by=req.confirmed_by, notes=item.notes, cycle=req.cycle,
                )
            elif item.verdict == "false_positive":
                add_false_positive(app_id=aid, confirmed_by=req.confirmed_by, notes=item.notes)
            else:
                errors.append({"application_id": aid,
                               "error": f"unknown verdict '{item.verdict}' "
                                        f"(expected confirmed_fraud | false_positive)"})
                continue
            recorded.append({"application_id": aid, "verdict": item.verdict})
        except ValueError as e:
            errors.append({"application_id": aid, "error": str(e)})

    n_confirmed = len(load_confirmed())
    n_fp        = len(load_false_positive_ids())

    job_id = None
    if req.dispatch_retrain and recorded:
        # Mirror the pattern-promotion dispatch: pin the Celery task_id to our
        # job_id so GET /v3/training/jobs/{job_id} can poll it. Incremental
        # orchestrator retrains on existing (now label-augmented) data — no
        # dataset_path needed.
        import uuid
        job_id = f"job_retrain_{uuid.uuid4().hex[:8]}"
        from src.api.tasks import run_incremental_task
        run_incremental_task.apply_async(
            kwargs=dict(cycle=req.cycle or "supervisor_batch", smoke_test=req.smoke_test),
            task_id=job_id,
        )

    log.info(
        "supervisor.confirm_batch",
        n_recorded=len(recorded), n_errors=len(errors),
        n_confirmed=n_confirmed, n_false_positives=n_fp,
        dispatched=bool(job_id),
    )
    return {
        "status": "ok",
        "n_recorded": len(recorded),
        "recorded": recorded,
        "errors": errors,
        "n_confirmed": n_confirmed,
        "n_false_positives": n_fp,
        "retrain_dispatched": bool(job_id),
        "job_id": job_id,
    }


@router.post("/pattern/test")
def pattern_test(req: PatternTestRequest):
    """Read-only: score a supervisor-uploaded fraud ring against the CURRENT
    model. Merges the ring, rebuilds features + the identity graph (so the
    members' real shared-IP/name edges exist), scores just the ring, then
    restores every canonical file. Answers 'does the model already catch it?'.
    Synchronous and can take ~1 min (a full feature+graph rebuild), like
    /v3/monitoring/evaluate-dataset.
    """
    from src.pattern_ingest_v3 import test_pattern
    try:
        result = test_pattern(req.dataset_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    log.info("supervisor.pattern_test", dataset_path=req.dataset_path, n_members=result["n_members"])
    return result


@router.post("/pattern/ingest")
def pattern_ingest(req: PatternIngestRequest):
    """Permanently ingest a supervisor ring as a RELATIONAL pattern: merge it
    into the dataset, extract its real intra-ring subgraph (all 5 relations),
    append it as a new cluster to the topology-exposure set, and record it in
    both confirmed stores (graph pattern + tabular rows). Optionally dispatch the
    human-gated incremental fine-tune afterwards (dispatch_retrain — the gate;
    recording alone never trains, hard stop #5). Synchronous merge+rebuild (~1 min)
    then an async retrain job.
    """
    from src.pattern_ingest_v3 import ingest_pattern
    try:
        result = ingest_pattern(
            csv_path=req.dataset_path, fraud_type=req.fraud_type,
            confirmed_by=req.confirmed_by, notes=req.notes, cycle=req.cycle,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 3: a pattern ingest is a permanent merge — land the ring's raw rows
    # in Postgres as a merged 'pattern' batch. (loe_patterns + confirmed_fraud
    # rows are already dual-written by the stores.) Best-effort.
    try:
        from src.db.ingest import merge_batch, stage_raw_csv
        _name = Path(req.dataset_path).stem
        stage_raw_csv(req.dataset_path, name=_name, kind="pattern", source="pattern_csv")
        merge_batch(_name)
    except Exception as e:  # noqa: BLE001
        log.warning("supervisor.pattern_ingest.pg_failed", error=str(e))

    job_id = None
    if req.dispatch_retrain:
        import uuid
        job_id = f"job_retrain_{uuid.uuid4().hex[:8]}"
        from src.api.tasks import run_incremental_task
        run_incremental_task.apply_async(
            kwargs=dict(cycle=req.cycle or "pattern_ingest", smoke_test=req.smoke_test),
            task_id=job_id,
        )
    result["retrain_dispatched"] = bool(job_id)
    result["job_id"] = job_id
    log.info("supervisor.pattern_ingest", pattern_id=result["pattern_id"],
             n_members=result["n_members"], dispatched=bool(job_id))
    return result


@router.get("/patterns")
def list_patterns():
    from src.confirmed_fraud_graph_store import list_pending, count_pending
    pending = list_pending()
    return {
        "pending_count": len(pending),
        "patterns": pending
    }


@router.get("/patterns/all")
def list_all_patterns():
    """Every pattern flagged across ALL sessions (all states), newest first —
    the persistent 'flagged directory' the console surfaces so a reviewer can
    verify whether a cluster was handled before. Read-only."""
    from src.confirmed_fraud_graph_store import list_all
    patterns = list_all()
    return {"count": len(patterns), "patterns": patterns}


@router.post("/patterns/delete")
def delete_patterns(req: DeletePatternRequest):
    """Hard-delete flagged patterns from the store (flagged-history cleanup).
    Removes the RECORD only — it does NOT retrain or remove any exposure cluster
    a PROMOTED pattern already wrote. `removed_promoted` lists deleted ids that
    were promoted, so the UI can warn that their training effect persists until a
    rebuild. 404 only if NONE of the ids existed."""
    from src.confirmed_fraud_graph_store import remove_patterns
    if not req.pattern_ids:
        raise HTTPException(status_code=400, detail="pattern_ids is empty — nothing to delete")
    result = remove_patterns(req.pattern_ids)
    if not result["removed"]:
        raise HTTPException(
            status_code=404,
            detail=f"No matching patterns to delete (not found: {result['not_found']})",
        )
    log.info("supervisor.delete_patterns", n_removed=len(result["removed"]),
             n_promoted=len(result["removed_promoted"]), n_not_found=len(result["not_found"]))
    return {"status": "ok", **result}


@router.get("/patterns/export/bulk")
def export_patterns_bulk():
    """Zip export of every pattern in the flagged-history store (all states,
    all sessions): manifest.csv + patterns/<pattern_id>.json."""
    from fastapi.responses import Response
    from src.export_v3 import build_pattern_bulk_export
    result = build_pattern_bulk_export()
    if result is None:
        raise HTTPException(status_code=404, detail="No flagged patterns to export")
    filename, data = result
    log.info("supervisor.export_patterns_bulk", bytes=len(data))
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/patterns/export/selected")
def export_patterns_selected(ids: str = ""):
    """Zip export of a reviewer-chosen subset of the flagged-history store.
    `ids` is a comma-separated list of pattern_ids (from the console's
    multi-select)."""
    from fastapi.responses import Response
    from src.export_v3 import build_pattern_selected_export
    id_list = [s.strip() for s in ids.split(",") if s.strip()]
    if not id_list:
        raise HTTPException(status_code=400, detail="No pattern IDs supplied (?ids=a,b,c)")
    if len(id_list) > 200:
        raise HTTPException(status_code=400, detail=f"Too many IDs ({len(id_list)}); export at most 200 at once")
    result = build_pattern_selected_export(id_list)
    if result is None:
        raise HTTPException(status_code=404, detail="None of the requested pattern IDs were found")
    filename, data = result
    log.info("supervisor.export_patterns_selected", n=len(id_list), bytes=len(data))
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/patterns/coverage/{app_id}")
def pattern_coverage(app_id: str):
    """Soft, IP-only 'has this cluster already been flagged for LOE?' check.
    Matches app_id against the shares_ip neighbourhood of every non-rejected
    pattern's members (all sessions). A heuristic the UI shows as a warning — the
    reviewer confirms via the 3D identity ring. Never mutates anything."""
    from src.confirmed_fraud_graph_store import ip_coverage_for_app
    return ip_coverage_for_app(app_id)


@router.post("/patterns/confirm")
def confirm_pattern(req: ConfirmPatternRequest):
    from src.confirmed_fraud_graph_store import add_confirmed_pattern
    try:
        pid = add_confirmed_pattern(
            app_id=req.application_id,
            fraud_type=req.fraud_type,
            subgraph=req.subgraph,
            confirmed_by=req.confirmed_by,
            notes=req.notes
        )
        log.info("supervisor.confirm_pattern", pattern_id=pid, app_id=req.application_id)
        return {"status": "ok", "pattern_id": pid}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/patterns/promote")
def promote_patterns(req: PromotePatternRequest):
    """Promote CONFIRMED/SELECTED patterns (appends their real topology to
    synthetic_exposure_graph_v3.pt) then dispatch the chosen retrain job.
    mode="incremental" (default) fine-tunes the existing checkpoint; mode=
    "full_retrain" reruns the full pipeline so the RGCN actually trains
    against the newly-appended topology (see _dispatch_retrain)."""
    if req.mode not in _VALID_RETRAIN_MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {sorted(_VALID_RETRAIN_MODES)}, got '{req.mode}'")

    from src.confirmed_fraud_graph_store import select, promote
    try:
        select(req.pattern_ids)
        promoted = promote(req.pattern_ids)
        log.info("supervisor.promote_patterns", n_promoted=len(promoted), pattern_ids=req.pattern_ids, mode=req.mode)

        job_id = _dispatch_retrain(req.mode, cycle="pattern_promotion", smoke_test=req.smoke_test)

        return {
            "status": "ok",
            "message": f"Promoted {len(promoted)} patterns. {req.mode} retrain dispatched.",
            "mode": req.mode,
            "job_id": job_id,
            "promoted_pattern_ids": [p["pattern_id"] for p in promoted]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/retrain")
def retrain_patterns(req: RetrainPatternsRequest):
    """Dispatch a retrain over the flagged-history store directly — covers
    patterns already PROMOTED (topology already appended; this just (re)runs
    training) as well as any still CONFIRMED/SELECTED (promoted first). Empty
    pattern_ids = every non-REJECTED pattern currently in the store (the
    console's 'select all' action). This is the general-purpose retrain
    trigger for the flagged-history panel, independent of the pending-queue
    promote button."""
    if req.mode not in _VALID_RETRAIN_MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {sorted(_VALID_RETRAIN_MODES)}, got '{req.mode}'")

    from src.confirmed_fraud_graph_store import list_all, select, promote

    all_patterns = list_all()
    if req.pattern_ids:
        wanted = set(req.pattern_ids)
        target = [p for p in all_patterns if p["pattern_id"] in wanted]
        not_found = sorted(wanted - {p["pattern_id"] for p in target})
    else:
        target = [p for p in all_patterns if p.get("state") != "REJECTED"]
        not_found = []

    if not target:
        raise HTTPException(status_code=404, detail=f"No matching non-rejected patterns to retrain on (not found: {not_found})")

    to_promote = [p["pattern_id"] for p in target if p.get("state") in ("CONFIRMED", "SELECTED")]
    newly_promoted: list[dict] = []
    if to_promote:
        select(to_promote)
        newly_promoted = promote(to_promote)

    job_id = _dispatch_retrain(req.mode, cycle="pattern_retrain", smoke_test=req.smoke_test)

    log.info("supervisor.retrain_patterns", mode=req.mode, n_target=len(target),
             n_newly_promoted=len(newly_promoted), job_id=job_id)

    return {
        "status": "ok",
        "message": f"{req.mode} retrain dispatched over {len(target)} pattern(s) "
                   f"({len(newly_promoted)} newly promoted).",
        "mode": req.mode,
        "job_id": job_id,
        "n_target": len(target),
        "target_pattern_ids": [p["pattern_id"] for p in target],
        "newly_promoted_pattern_ids": [p["pattern_id"] for p in newly_promoted],
        "not_found": not_found,
    }
