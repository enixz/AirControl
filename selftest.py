#!/usr/bin/env python
"""AirControl 标准化自测入口。

把 CI（.github/workflows/ci.yml）里的三道标准检查整合为一条本地命令，支撑
「改进计划 → 编码 → 自测 → 达标即停 / 未达标再计划再编码再测试」的开发循环：

    python selftest.py              # 编译 + lint + 测试（默认，等价于 CI 核心检查）
    python selftest.py --cov        # 附带覆盖率（--cov=app --cov-report=term-missing）
    python selftest.py --lint-fix   # 先 `ruff check --fix` 自动修复再继续
    python selftest.py --only tests # 仅跑某一阶段：compile / lint / tests
    python selftest.py --skip lint  # 跳过某一阶段
    python selftest.py -k mouse     # 其余未知参数透传给 pytest，聚焦单点迭代
    python selftest.py tests/test_mouse_controller_edge.py -q   # 同上

退出码约定（便于脚本/循环判断）：
    0  —— 全部通过：本轮目标达标，可停止迭代。
    非 0 —— 存在失败：需要继续「再计划 → 再编码 → 再测试」。

三道检查与 CI 严格对齐，因此「本地绿 ⇒ CI 绿」。测试阶段自动设置
QT_QPA_PLATFORM=offscreen，使 PyQt6 在无显示环境（CI / 后台）下也能运行。
"""
import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows 中文控制台默认可能是 GBK，统一把输出改为 UTF-8（errors=replace 兜底），
# 避免本脚本/子进程的中文与符号触发 UnicodeEncodeError。
for _stream in ("stdout", "stderr"):
    _s = getattr(sys, _stream, None)
    if isinstance(_s, io.TextIOWrapper):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent
# 检查目标与 CI 保持一致（compileall / ruff 的作用范围），并把本脚本一并纳入自检。
TARGETS = ["app", "tests", "benchmark_gesture_ab.py", "build.py", "selftest.py"]
STAGES = ("compile", "lint", "tests")


def _print_banner(text):
    line = "=" * 64
    print(f"\n{line}\n{text}\n{line}")


def _run_stage(title, cmd, env):
    """运行一个阶段，打印命令、结果与耗时，返回 (ok: bool, seconds: float)。"""
    print(f"\n{'-' * 64}\n>> {title}\n   $ {' '.join(cmd)}", flush=True)
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    elapsed = time.perf_counter() - start
    ok = proc.returncode == 0
    status = "[ OK ]" if ok else f"[FAIL] (exit {proc.returncode})"
    print(f"   {status}  ·  {elapsed:.2f}s", flush=True)
    return ok, elapsed


def _build_commands(args):
    """根据参数构造各阶段命令，返回有序的 (stage, title, cmd) 列表。"""
    py = sys.executable
    commands = []

    if "compile" in args.stages:
        commands.append((
            "compile",
            "编译检查 (py_compile 全部源文件)",
            [py, "-m", "compileall", "-q", *TARGETS],
        ))

    if "lint" in args.stages:
        if args.lint_fix:
            commands.append((
                "lint",
                "代码风格自动修复 (ruff check --fix)",
                [py, "-m", "ruff", "check", "--fix", *TARGETS],
            ))
        else:
            commands.append((
                "lint",
                "代码风格检查 (ruff check)",
                [py, "-m", "ruff", "check", *TARGETS],
            ))

    if "tests" in args.stages:
        pytest_cmd = [py, "-m", "pytest"]
        if args.cov:
            pytest_cmd += ["--cov=app", "--cov-report=term-missing"]
        pytest_cmd += args.pytest_args  # 透传 -k / 文件名 / -q 等
        commands.append((
            "tests",
            "单元测试 (pytest)",
            pytest_cmd,
        ))

    return commands


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="selftest.py",
        description="AirControl 标准化自测：编译 + lint + 测试。未知参数透传给 pytest。",
        epilog="退出码 0=全部通过(达标可停)，非0=有失败(需继续迭代)。",
    )
    parser.add_argument(
        "--only", choices=STAGES, action="append", metavar="STAGE",
        help="仅运行指定阶段（可重复）：compile / lint / tests",
    )
    parser.add_argument(
        "--skip", choices=STAGES, action="append", metavar="STAGE",
        help="跳过指定阶段（可重复）",
    )
    parser.add_argument(
        "--cov", action="store_true",
        help="测试阶段附带覆盖率统计",
    )
    parser.add_argument(
        "--lint-fix", action="store_true",
        help="lint 阶段改用 ruff check --fix 自动修复",
    )
    parser.add_argument(
        "-x", "--fail-fast", action="store_true",
        help="任一阶段失败立即停止（默认跑完所有阶段再汇总）",
    )
    args, extra = parser.parse_known_args(argv)

    # 计算实际要跑的阶段集合。
    selected = set(args.only) if args.only else set(STAGES)
    if args.skip:
        selected -= set(args.skip)
    # 保持固定顺序：compile -> lint -> tests
    args.stages = [s for s in STAGES if s in selected]
    args.pytest_args = extra
    return args


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if not args.stages:
        print("没有要运行的阶段（--only/--skip 组合为空）。", file=sys.stderr)
        return 2

    # 测试阶段需要无显示环境的 Qt 平台插件。
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")

    _print_banner("AirControl 自测  ·  阶段: " + " -> ".join(args.stages))

    results = []  # (stage, ok, seconds)
    for stage, title, cmd in _build_commands(args):
        ok, elapsed = _run_stage(title, cmd, env)
        results.append((stage, ok, elapsed))
        if not ok and args.fail_fast:
            print("\n[fail-fast] 阶段失败，停止后续检查。", flush=True)
            break

    total = sum(e for _, _, e in results)
    all_ok = all(ok for _, ok, _ in results)

    _print_banner("自测结果汇总")
    for stage, ok, elapsed in results:
        mark = "[ OK ]" if ok else "[FAIL]"
        print(f"  {mark}  {stage:<8}  {elapsed:6.2f}s")
    print(f"\n  合计耗时: {total:.2f}s")
    if all_ok:
        print("  结论: 全部通过 —— 本轮目标达标，可停止迭代。")
    else:
        failed = [s for s, ok, _ in results if not ok]
        print(f"  结论: 失败阶段 = {', '.join(failed)} —— 请继续修复后重跑。")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
