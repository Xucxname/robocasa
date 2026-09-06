#!/usr/bin/env python3
"""Stream raw Sonic episodes from SSH, render wrist views, and discard raw files.

Only a small batch of HDF5 files exists locally at once under ``/tmp``.  Each
batch is state-aligned, rendered, validated, and then removed.  The cleaned
LeRobot baseline remains unchanged; a resumable ``.partial`` output is promoted
only after all episodes and all three camera streams pass full validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/robocasa_numba_cache")

import robocasa.scripts.backfill_sonic_wrist_cameras as core


WORK_NAME = "camera_backfill_stream_work.json"


def _load_episode_sources(cleaned: Path) -> list[dict[str, Any]]:
    cleanup = core._json_load(cleaned / "meta" / "cleanup_report.json")
    replay = core._json_load(cleaned / "meta" / "raw_replay_alignment.json")
    episodes = core.read_jsonlines(cleaned / "meta" / "episodes.jsonl")
    mappings = [
        entry
        for entry in cleanup["episode_mapping"]
        if int(entry["output_episode_index"]) >= 0
    ]
    selected_entries = [
        entry
        for entry in replay["matches"]
        if entry.get("selection") == "keep_replay_passed"
    ]
    selected = {
        int(entry["lerobot_episode_index"]): entry for entry in selected_entries
    }
    output_indices = [int(entry["output_episode_index"]) for entry in mappings]
    source_indices = [int(entry["source_episode_index"]) for entry in mappings]
    raw_episodes = [str(entry["raw_episode"]) for entry in selected_entries]
    expected = list(range(len(episodes)))
    if len(mappings) != len(episodes) or sorted(output_indices) != expected:
        raise ValueError("cleanup mapping does not cover every cleaned episode")
    if len(output_indices) != len(set(output_indices)):
        raise ValueError("duplicate cleanup output episode index")
    if len(source_indices) != len(set(source_indices)):
        raise ValueError("duplicate cleanup source episode index")
    if len(selected_entries) != len(episodes) or len(selected) != len(episodes):
        raise ValueError("replay-passed source episode mapping is not unique/complete")
    if set(source_indices) != set(selected):
        raise ValueError("cleanup and replay-passed source episode sets differ")
    if len(raw_episodes) != len(set(raw_episodes)):
        raise ValueError("replay alignment reuses a raw episode")

    result = []
    for mapping in sorted(mappings, key=lambda entry: entry["output_episode_index"]):
        output_index = int(mapping["output_episode_index"])
        source_index = int(mapping["source_episode_index"])
        raw_episode = str(selected[source_index]["raw_episode"])
        if re.fullmatch(r"ep_[0-9]+_[0-9]+", raw_episode) is None:
            raise ValueError(f"unsafe raw episode name: {raw_episode!r}")
        result.append(
            {
                "output_episode_index": output_index,
                "source_episode_index": source_index,
                "raw_episode": raw_episode,
            }
        )
    return result


def _stream_invariant(
    cleaned: Path,
    source: Path,
    partial: Path,
    ssh_host: str,
    identity_file: Path,
    remote_episodes_root: str,
    baseline_manifest: dict[str, dict[str, Any]],
    episode_sources: Sequence[dict[str, Any]],
    thresholds: core.AlignmentThresholds,
) -> dict[str, Any]:
    return {
        "implementation_version": 1,
        "streaming_script_sha256": core._sha256_file(Path(__file__).resolve()),
        "core_script_sha256": core._sha256_file(Path(core.__file__).resolve()),
        "cleaned_dataset": str(cleaned),
        "source_dataset": str(source),
        "partial_dataset": str(partial),
        "ssh_host": ssh_host,
        "identity_file": str(identity_file),
        "remote_episodes_root": remote_episodes_root,
        "baseline_manifest": baseline_manifest,
        "source_info": core._file_fingerprint(source / "meta" / "info.json"),
        "cleanup_report": core._file_fingerprint(
            cleaned / "meta" / "cleanup_report.json"
        ),
        "raw_replay_alignment": core._file_fingerprint(
            cleaned / "meta" / "raw_replay_alignment.json"
        ),
        "thresholds": asdict(thresholds),
        "cameras": core._camera_metadata(),
        "encoding": core.ENCODING_CONFIG,
        "episode_sources": list(episode_sources),
    }


def _load_or_create_work(partial: Path, invariant: dict[str, Any]) -> dict[str, Any]:
    path = partial / "meta" / WORK_NAME
    if path.is_file():
        work = core._json_load(path)
        if work.get("invariant") != invariant:
            raise RuntimeError(
                "streaming partial belongs to different source/code/alignment inputs"
            )
        if not isinstance(work.get("episodes"), dict):
            raise RuntimeError("streaming work file has invalid episode records")
        if not isinstance(work.get("completed_videos"), dict):
            raise RuntimeError("streaming work file has invalid video records")
        return work
    work = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "invariant": invariant,
        "episodes": {},
        "completed_videos": {},
    }
    core._json_write(path, work)
    return work


def _save_work(partial: Path, work: dict[str, Any]) -> None:
    work["updated_at"] = datetime.now().astimezone().isoformat()
    core._json_write(partial / "meta" / WORK_NAME, work)


def _fetch_raw_hdf5(
    ssh_host: str,
    identity_file: Path,
    remote_root: str,
    raw_episode: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    if temporary.exists():
        temporary.unlink()
    remote_path = f"{remote_root.rstrip('/')}/{raw_episode}/ep_demo.hdf5"
    print(f"[fetch] {raw_episode}", flush=True)
    subprocess.run(
        [
            "scp",
            "-q",
            "-i",
            str(identity_file),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            f"{ssh_host}:{remote_path}",
            str(temporary),
        ],
        check=True,
    )
    if temporary.stat().st_size == 0:
        raise ValueError(f"downloaded empty raw HDF5: {raw_episode}")
    temporary.replace(destination)
    print(
        f"[fetch] {raw_episode} complete ({destination.stat().st_size / 2**20:.1f} MiB)",
        flush=True,
    )


def _episode_signature(
    job: core.EpisodeJob, alignment_report: dict[str, Any]
) -> dict[str, Any]:
    return {
        "output_episode_index": job.output_episode_index,
        "source_episode_index": job.source_episode_index,
        "raw_episode": job.raw_episode,
        "source_frames": job.source_frames,
        "output_frames": job.output_frames,
        "raw_hdf5": {
            "bytes": job.raw_hdf5_bytes,
            "sha256": job.raw_hdf5_sha256,
        },
        "source_parquet": {
            "bytes": job.source_parquet_bytes,
            "sha256": job.source_parquet_sha256,
        },
        "cleaned_parquet": {
            "bytes": job.cleaned_parquet_bytes,
            "sha256": job.cleaned_parquet_sha256,
        },
        "retained_source_indices_sha256": alignment_report[
            "retained_source_indices_sha256"
        ],
        "raw_render_indices_sha256": alignment_report[
            "raw_render_indices_sha256"
        ],
    }


def _trusted_hashes_for_job(
    job: core.EpisodeJob,
    partial: Path,
    work: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    info = core.load_info(partial)
    trusted = []
    for _camera_name, video_key in core.CAMERAS:
        path = core.get_video_path(partial, info, job.output_episode_index, video_key)
        relative = str(path.relative_to(partial))
        record = work["completed_videos"].get(relative)
        if isinstance(record, dict) and isinstance(record.get("sha256"), str):
            trusted.append((relative, record["sha256"]))
    return tuple(trusted)


def _episode_is_complete(
    source_record: dict[str, Any],
    partial: Path,
    work: dict[str, Any],
) -> bool:
    episode_key = str(source_record["output_episode_index"])
    record = work["episodes"].get(episode_key)
    if not isinstance(record, dict):
        return False
    if not isinstance(record.get("alignment_report"), dict):
        return False
    render_report = record.get("render_report")
    if not isinstance(render_report, dict):
        return False
    for details in render_report.get("videos", {}).values():
        path = partial / details["path"]
        try:
            actual = core._validate_video(
                path,
                int(details["frames"]),
                int(details["width"]),
                int(details["height"]),
                int(round(float(details["fps"]))),
            )
        except Exception:
            return False
        if actual["sha256"] != details["sha256"]:
            return False
    return len(render_report.get("videos", {})) == 2


def _abort_pool(pool: ProcessPoolExecutor, futures) -> None:
    for future in futures:
        future.cancel()
    for process in list(pool._processes.values()):
        if process.is_alive():
            process.terminate()
    pool.shutdown(wait=True, cancel_futures=True)


def _render_batch(
    jobs: Sequence[core.EpisodeJob],
    partial: Path,
    work: dict[str, Any],
    workers: int,
) -> None:
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context)
    futures = {}
    try:
        futures = {
            pool.submit(core._render_episode, asdict(job)): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            episode_key = str(job.output_episode_index)
            work["episodes"][episode_key]["render_report"] = result
            for details in result["videos"].values():
                work["completed_videos"][details["path"]] = {
                    "bytes": int(details["bytes"]),
                    "sha256": details["sha256"],
                }
            _save_work(partial, work)
            print(
                f"[render] output_episode={job.output_episode_index} "
                f"frames={job.output_frames} new_cameras={result['rendered_camera_count']}",
                flush=True,
            )
    except BaseException:
        _abort_pool(pool, futures)
        raise
    else:
        pool.shutdown(wait=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dataset", type=Path, required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--remote-episodes-root", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--temp-parent", type=Path, default=Path("/tmp"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cleaned = args.cleaned_dataset.expanduser().resolve()
    source = args.source_dataset.expanduser().resolve()
    output = args.output_dataset.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    identity_file = args.identity_file.expanduser().resolve()
    temp_parent = args.temp_parent.expanduser().resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    for required in (cleaned, source, temp_parent):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if not identity_file.is_file():
        raise FileNotFoundError(identity_file)
    for source_root in (cleaned, source):
        if core._paths_overlap(output, source_root) or core._paths_overlap(
            partial, source_root
        ):
            raise ValueError(f"output overlaps an input: {source_root}")
    if output.exists():
        raise FileExistsError(f"final output already exists: {output}")

    baseline_manifest = core._dataset_identity_manifest(cleaned)
    if not partial.exists():
        partial.parent.mkdir(parents=True, exist_ok=True)
        print(f"Copying baseline dataset to {partial}", flush=True)
        shutil.copytree(cleaned, partial, copy_function=shutil.copy2)
    else:
        print(f"Resuming partial dataset at {partial}", flush=True)
    core.validate_partial_baseline(cleaned, partial, baseline_manifest)

    thresholds = core.AlignmentThresholds()
    episode_sources = _load_episode_sources(cleaned)
    invariant = _stream_invariant(
        cleaned,
        source,
        partial,
        args.ssh_host,
        identity_file,
        args.remote_episodes_root,
        baseline_manifest,
        episode_sources,
        thresholds,
    )
    work = _load_or_create_work(partial, invariant)
    pending = [
        record
        for record in episode_sources
        if not _episode_is_complete(record, partial, work)
    ]
    print(
        f"Streaming backfill: total={len(episode_sources)} "
        f"complete={len(episode_sources) - len(pending)} pending={len(pending)}",
        flush=True,
    )

    for batch_start in range(0, len(pending), args.workers):
        batch = pending[batch_start : batch_start + args.workers]
        with tempfile.TemporaryDirectory(
            prefix="robocasa_drawer_raw_", dir=temp_parent
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for record in batch:
                raw_episode = record["raw_episode"]
                destination = temporary_root / raw_episode / "ep_demo.hdf5"
                _fetch_raw_hdf5(
                    args.ssh_host,
                    identity_file,
                    args.remote_episodes_root,
                    raw_episode,
                    destination,
                )

            jobs: list[core.EpisodeJob] = []
            for record in batch:
                output_index = int(record["output_episode_index"])
                one_jobs, one_reports = core.build_episode_jobs(
                    cleaned,
                    source,
                    temporary_root,
                    partial,
                    thresholds,
                    output_episode_indices={output_index},
                )
                if len(one_jobs) != 1 or len(one_reports) != 1:
                    raise RuntimeError(f"expected one aligned job for {output_index}")
                job = one_jobs[0]
                alignment_report = one_reports[0]
                signature = _episode_signature(job, alignment_report)
                episode_key = str(output_index)
                previous = work["episodes"].get(episode_key)
                if previous is not None and previous.get("signature") != signature:
                    raise RuntimeError(
                        f"episode {output_index} raw/alignment changed since prior attempt"
                    )
                work["episodes"][episode_key] = {
                    "signature": signature,
                    "alignment_report": alignment_report,
                    **(
                        {"render_report": previous["render_report"]}
                        if previous is not None and "render_report" in previous
                        else {}
                    ),
                }
                job = replace(
                    job,
                    trusted_video_sha256=_trusted_hashes_for_job(job, partial, work),
                )
                jobs.append(job)
            _save_work(partial, work)
            _render_batch(jobs, partial, work, args.workers)
        print(
            f"[cleanup] removed temporary raw batch "
            f"{batch_start + 1}-{batch_start + len(batch)}",
            flush=True,
        )

    if len(work["episodes"]) != len(episode_sources):
        raise RuntimeError("not every episode has a streaming work record")
    alignment_reports = []
    render_reports = []
    for output_index in range(len(episode_sources)):
        record = work["episodes"][str(output_index)]
        if "alignment_report" not in record or "render_report" not in record:
            raise RuntimeError(f"episode {output_index} is incomplete")
        alignment_reports.append(record["alignment_report"])
        render_reports.append(record["render_report"])

    if core._dataset_identity_manifest(cleaned) != baseline_manifest:
        raise RuntimeError("baseline changed during streaming backfill")
    if core._file_fingerprint(source / "meta" / "info.json") != invariant["source_info"]:
        raise RuntimeError("source info.json changed during streaming backfill")
    core.validate_partial_baseline(cleaned, partial, baseline_manifest)
    remote_source = Path(
        f"{args.ssh_host}:{args.remote_episodes_root.rstrip('/')}"
    )
    report = core._update_metadata(
        partial,
        cleaned,
        source,
        remote_source,
        thresholds,
        alignment_reports,
        render_reports,
        baseline_manifest,
    )
    report["transfer"] = {
        "mode": "per-batch SSH temporary copy",
        "raw_hdf5_persisted_locally": False,
        "batch_size": args.workers,
        "temporary_parent": str(temp_parent),
        "remote_source": f"{args.ssh_host}:{args.remote_episodes_root.rstrip('/')}",
    }
    core._validate_preserved_baseline(partial, baseline_manifest)
    print("Running full LeRobot/parquet/video/statistics validation.", flush=True)
    validation = core.verify_dataset(
        partial,
        expected_episodes=int(report["result"]["total_episodes"]),
        expected_frames=int(report["result"]["total_frames"]),
        pose_modes=core.DEFAULT_POSE_MODES,
    )
    report["validation"] = validation
    report["validated_at"] = datetime.now().astimezone().isoformat()
    core._json_write(partial / "meta" / "camera_backfill_report.json", report)
    (partial / "meta" / WORK_NAME).unlink()
    partial.replace(output)
    print(
        f"COMPLETE output={output} episodes={report['result']['total_episodes']} "
        f"frames={report['result']['total_frames']} "
        f"videos={report['result']['total_videos']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
