from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from .config import XTBMDConfig


@dataclass
class MDRunResult:
    frames: list[Atoms]
    metadata: dict


class XTBMDRunner:
    def __init__(self, config: XTBMDConfig):
        self.config = config

    def generate(self, molecule: Atoms, run_dir: Path) -> MDRunResult:
        return self.run(molecule, run_dir)

    def run(self, molecule: Atoms, run_dir: Path) -> MDRunResult:
        run_dir.mkdir(parents=True, exist_ok=True)
        input_xyz = run_dir / "input.xyz"
        write(input_xyz.as_posix(), molecule)
        proc, cmd = self._run_md_attempt(
            run_dir=run_dir,
            input_xyz=input_xyz,
            md_input_name="md.inp",
            include_advanced_keywords=True,
            log_suffix="attempt1",
        )
        fallback_used = False
        accepted_nonzero = False
        if proc.returncode != 0:
            fallback_used = True
            proc, cmd = self._run_md_attempt(
                run_dir=run_dir,
                input_xyz=input_xyz,
                md_input_name="md_fallback.inp",
                include_advanced_keywords=False,
                log_suffix="attempt2",
            )
        if proc.returncode != 0:
            if self._can_accept_partial_md_success(run_dir=run_dir):
                accepted_nonzero = True
            else:
                raise RuntimeError(self._build_md_error_message(run_dir=run_dir, return_code=proc.returncode))
        traj_path = run_dir / "xtb.trj"
        if not traj_path.exists():
            raise FileNotFoundError(f"xTB trajectory not found: {traj_path}")
        frames = self._read_md_trajectory(traj_path)
        if not isinstance(frames, list):
            frames = [frames]
        metadata = {
            "cmd": cmd,
            "return_code": proc.returncode,
            "temperature_k": self.config.temperature_k,
            "time_ps": self.config.time_ps,
            "step_fs": self.config.step_fs,
            "dump_fs": self.config.dump_fs,
            "seed": self.config.seed,
            "n_frames": len(frames),
            "fallback_md_input_used": fallback_used,
            "accepted_nonzero_return_code": accepted_nonzero,
        }
        return MDRunResult(frames=frames, metadata=metadata)

    def _build_cmd(self, input_xyz: Path, md_input: Path) -> list[str]:
        cmd = [self.config.xtb_executable, input_xyz.name]
        if self.config.gfnff:
            cmd.append("--gfnff")
        cmd.extend(["--md", "--input", md_input.name, "--omd"])
        return cmd

    @staticmethod
    def _read_md_trajectory(traj_path: Path):
        try:
            return read(traj_path.as_posix(), index=":")
        except Exception:
            return read(traj_path.as_posix(), index=":", format="xyz")

    def _run_md_attempt(
        self,
        run_dir: Path,
        input_xyz: Path,
        md_input_name: str,
        include_advanced_keywords: bool,
        log_suffix: str,
    ) -> tuple[subprocess.CompletedProcess, list[str]]:
        md_input = run_dir / md_input_name
        md_input.write_text(self._render_md_input(include_advanced_keywords=include_advanced_keywords), encoding="utf-8")
        cmd = self._build_cmd(input_xyz=input_xyz, md_input=md_input)
        proc = subprocess.run(
            cmd,
            cwd=run_dir.as_posix(),
            capture_output=True,
            text=True,
            check=False,
        )
        (run_dir / f"xtb.stdout.{log_suffix}.log").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / f"xtb.stderr.{log_suffix}.log").write_text(proc.stderr or "", encoding="utf-8")
        if log_suffix == "attempt1":
            (run_dir / "xtb.stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (run_dir / "xtb.stderr.log").write_text(proc.stderr or "", encoding="utf-8")
        return proc, cmd

    def _render_md_input(self, include_advanced_keywords: bool = True) -> str:
        lines = [
            "$md",
            f"  temp={self.config.temperature_k}",
            f"  time={self.config.time_ps}",
            f"  dump={self.config.dump_fs}",
            f"  step={self.config.step_fs}",
            f"  seed={self.config.seed}",
        ]
        if include_advanced_keywords:
            lines.extend(
                [
                    f"  hmass={self.config.hmass}",
                    f"  shake={self.config.shake}",
                ]
            )
        else:
            lines.append("  shake=0")
        lines.extend(["$end", ""])
        return "\n".join(lines)

    @staticmethod
    def _build_md_error_message(run_dir: Path, return_code: int) -> str:
        stderr_tail = XTBMDRunner._tail_text(run_dir / "xtb.stderr.attempt2.log", 30)
        stdout_tail = XTBMDRunner._tail_text(run_dir / "xtb.stdout.attempt2.log", 30)
        return (
            f"xTB MD failed with return code {return_code}. "
            f"run_dir={run_dir}. "
            f"stderr_tail={stderr_tail} "
            f"stdout_tail={stdout_tail}"
        )

    @staticmethod
    def _tail_text(path: Path, n_lines: int) -> str:
        if not path.exists():
            return "<missing>"
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        tail = "\n".join(lines[-n_lines:])
        return tail if tail else "<empty>"

    def _can_accept_partial_md_success(self, run_dir: Path) -> bool:
        traj_path = run_dir / "xtb.trj"
        if not traj_path.exists():
            return False
        try:
            frames = self._read_md_trajectory(traj_path)
        except Exception:
            return False
        if isinstance(frames, list):
            return len(frames) > 0
        return len(frames) > 0
