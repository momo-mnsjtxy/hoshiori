"""奖励沙箱：在受限子进程里运行外部 ``reward(payload)``，只交回分数或错误。

调用方式：``run(path, payload, timeout=10, memory_mb=512)``。path 必须是一个 ``.py``
文件，文件内定义可调用的 ``reward(payload)``；payload 要能编码成 JSON。成功返回
``(float, None)``，失败返回 ``(None, 明确错误)``，外部代码的异常不会越过这个边界。

调用示例：
  from features import sandbox
  score, error = sandbox.run("score.py", {"prompt": "题", "response": "答"})

隔离边界是独立 Python 进程、独立进程组、隔离模式 ``-I``，并用 RLIMIT_AS / RLIMIT_CPU
限制地址空间和 CPU 时间；墙钟超时由父进程补一层，届时杀掉整个进程组。它不限制文件和网络访问。
"""

import json
import math
import numbers
import os
import resource
import runpy
import signal
import subprocess
import sys
from pathlib import Path

RESULT_LIMIT = 16 * 1024  # 子进程只准交回一小段结构化结果，避免错误文本拖大主进程
MEMORY_RESULT = b'{"kind":"memory"}'  # 内存耗尽后不再临时拼字符串，仍能把类型交回来


def run(path, payload, timeout=10, memory_mb=512) -> tuple:
    """运行一份外部奖励代码，返回 ``(分数, 错误)``，任何失败都不向调用方抛出。

    ``timeout`` 同时决定墙钟等待上限与子进程 CPU 上限；``memory_mb`` 限制整个子进程
    的虚拟地址空间。分数只接受有限实数，bool、字符串、NaN 和 Inf 都会被拒绝。
    """
    try:                                                                        # path 和 payload 都来自调用边界，只在这里检查一次
        target = Path(path)                                                     # 统一接字符串和 Path
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")          # 标准输入只走 JSON，不共享主进程对象
        seconds = float(timeout)                                                # 接受整数或小数秒
        memory = int(memory_mb)                                                 # MB 在这里统一换算
    except (TypeError, ValueError, OverflowError) as bad:                       # 输入不合法也不越过返回契约
        return None, f"奖励沙箱参数无效: {type(bad).__name__}: {bad}"
    if target.suffix.lower() != ".py":                                        # 只执行约定的单文件 Python 奖励
        return None, "奖励文件必须是 .py 文件"
    if not target.is_file():                                                   # 不把找不到文件变成子进程异常
        return None, f"奖励文件不存在: {target}"
    if not math.isfinite(seconds) or seconds <= 0:                              # 非正等待时间没有可执行含义
        return None, "奖励沙箱参数无效: timeout 必须是正数"
    if memory <= 0:                                                             # 非正内存上限同样在边界挡住
        return None, "奖励沙箱参数无效: memory_mb 必须是正整数"

    cpu = max(1, math.ceil(seconds))                                             # RLIMIT_CPU 只收整秒，向上取整
    process, result_fd, error = _start(target, cpu, memory)                      # 新进程先建好，外部代码尚未执行
    if error:
        return None, error                                                      # 解释器起不来时直接交代
    raw, error = _wait(process, result_fd, body, seconds)                        # 喂 payload 并等到结束或超时
    if error:
        return None, error                                                      # 超时和通信故障已经完成清场
    return _answer(raw, process.returncode, cpu, memory)                         # 最终协议统一翻成公开二元组


def _start(target: Path, cpu: int, memory: int) -> tuple:
    """启动一个隔离子进程，返回 ``(进程, 结果管道, 错误)``。"""
    read_fd, write_fd = os.pipe()                                               # 专用管道只传最终结果，外部 print 全丢弃
    command = [                                                                 # 子进程重新执行本文件，再由它载入奖励文件
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--child",
        str(target.resolve()),
        str(write_fd),
        str(cpu),
        str(memory * 1024 * 1024),
    ]
    try:
        process = subprocess.Popen(                                             # 新会话让超时时能连同后代一起杀掉
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(write_fd,),
            start_new_session=True,
        )
    except (OSError, ValueError) as bad:                                        # 解释器或管道起不来仍按返回契约交代
        os.close(read_fd)
        os.close(write_fd)
        return None, None, f"奖励子进程启动失败: {type(bad).__name__}: {bad}"
    os.close(write_fd)                                                          # 父进程不写结果，及时关掉自己的副本
    return process, read_fd, None                                               # 读端由等待动作负责关闭


def _wait(process: subprocess.Popen, result_fd: int, body: bytes, seconds: float) -> tuple:
    """把 payload 交给子进程并等待；结束后清掉整组，返回 ``(原始结果, 错误)``。"""
    try:
        try:
            process.communicate(input=body, timeout=seconds)                    # 墙钟时间到就由下面统一清理
        except subprocess.TimeoutExpired:
            _stop(process)                                                      # 忙循环或阻塞都杀整个进程组
            return None, f"奖励函数执行超时（上限 {seconds:g} 秒）"
        _stop(process)                                                          # 主子进程结束后也清掉它可能留下的后代
        return _read(result_fd), None                                           # 固定上限读取，外部结果撑不大主进程
    except (OSError, subprocess.SubprocessError) as bad:                         # 进程通信故障也遵守不外抛的接口
        _stop(process)
        return None, f"奖励子进程通信失败: {type(bad).__name__}: {bad}"
    finally:
        os.close(result_fd)                                                     # 每条返回路径都释放主进程的读端


def _answer(raw: bytes, returncode: int, cpu: int, memory: int) -> tuple:
    """把子进程的有界 JSON 结果翻成公开的 ``(分数, 错误)``。"""
    if not raw:                                                                 # 没有结构化结果时按退出信号判断资源原因
        if returncode == -signal.SIGXCPU:
            return None, f"奖励函数执行超时（CPU 上限 {cpu} 秒）"
        memory_signals = {signal.SIGKILL, signal.SIGSEGV, signal.SIGABRT}
        if hasattr(signal, "SIGBUS"):
            memory_signals.add(signal.SIGBUS)
        if returncode in {-one for one in memory_signals}:
            return None, f"奖励函数内存超限（上限 {memory} MB）"
        return None, f"奖励函数执行异常: 子进程退出码 {returncode}"
    try:
        result = json.loads(raw.decode("utf-8"))                                 # 专用管道里只认一份 JSON 结果
    except (UnicodeDecodeError, json.JSONDecodeError) as bad:
        return None, f"奖励函数执行异常: 子进程结果损坏（{type(bad).__name__}）"

    kind = result.get("kind")                                                   # 子进程已经把外部结果分好类
    if kind == "ok":
        return float(result["score"]), None                                     # 对外统一交有限浮点分数
    if kind == "not_number":
        return None, f"奖励函数返回值不是数字: {result.get('type', 'unknown')}"
    if kind == "non_finite":
        return None, "奖励函数返回值不是有限数"
    if kind == "memory":
        return None, f"奖励函数内存超限（上限 {memory} MB）"
    detail = result.get("detail") or "未知异常"                                 # 未定义 reward 也从这条清楚交代
    return None, f"奖励函数执行异常: {detail}"


def _stop(process: subprocess.Popen) -> None:
    """杀掉沙箱进程组并等它收尸；主进程已退出时仍清理可能留下的后代。"""
    try:
        os.killpg(process.pid, signal.SIGKILL)                                   # 新会话的 pid 同时就是进程组 id
    except ProcessLookupError:
        pass                                                                     # 整组已经自然退出
    if process.poll() is None:                                                   # killpg 失败前进程仍活着就再按对象杀一次
        process.kill()
    process.wait()                                                               # 不留僵尸进程


def _read(fd: int) -> bytes:
    """非阻塞读取子进程结果；即使外部代码遗留进程握着管道，也不会把主进程卡住。"""
    os.set_blocking(fd, False)                                                   # 主子进程已结束，不再等待迟到内容
    chunks = []                                                                  # 结果很小，分块只为兼容短读
    size = 0                                                                     # 硬上限保护主进程
    while size < RESULT_LIMIT:
        try:
            part = os.read(fd, RESULT_LIMIT - size)                              # 只读剩余配额
        except BlockingIOError:
            break                                                                # 没有可读内容就结束，不等外部后代
        if not part:
            break                                                                # 管道正常关闭
        chunks.append(part)                                                      # 收下这一块结构化结果
        size += len(part)                                                        # 下一轮只留剩余空间
    return b"".join(chunks)                                                      # 交给父进程解 JSON


def _child(path: Path, result_fd: int, cpu: int, memory: int) -> None:
    """子进程入口：先落资源上限，再载入目标并调用 ``reward(payload)``。"""
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))                     # 地址空间超过上限时分配会失败
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))                      # 先发 SIGXCPU，再用硬线确保停下
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))            # 只从主进程接一份 JSON 输入
        scope = runpy.run_path(str(path))                                        # 文件顶层和 reward 都只在这个进程执行
        reward = scope.get("reward")                                             # 接口名字固定，调用者不用猜
        if not callable(reward):
            _send(result_fd, {"kind": "exception", "detail": "未定义可调用的 reward(payload)"})
            return
        value = reward(payload)                                                  # 外部代码真正执行的唯一入口
        if isinstance(value, bool) or not isinstance(value, numbers.Number):     # bool 虽是 int 子类，但不是有效分数
            _send(result_fd, {"kind": "not_number", "type": type(value).__name__})
            return
        try:
            score = float(value)                                                 # numpy、Decimal 等数字统一转成 float
        except OverflowError:
            _send(result_fd, {"kind": "non_finite"})                            # 大到 float 装不下也不能进入损失
            return
        except (TypeError, ValueError):
            _send(result_fd, {"kind": "not_number", "type": type(value).__name__})  # 复数等不能落到实数轴
            return
        if not math.isfinite(score):                                             # NaN/Inf 不得进入训练损失
            _send(result_fd, {"kind": "non_finite"})
            return
        _send(result_fd, {"kind": "ok", "score": score})                     # 成功只交回一个有限分数
    except MemoryError:
        os.write(result_fd, MEMORY_RESULT)                                       # 内存耗尽时用预建常量回报，不再分配大对象
    except BaseException as bad:                                                 # 外部 SystemExit 等也不能越过进程边界
        try:
            detail = f"{type(bad).__name__}: {bad}"[:2000]                       # 错误够定位即可，长度要有上限
        except BaseException:
            detail = type(bad).__name__                                          # 异常自己的 __str__ 坏了也能报类型
        _send(result_fd, {"kind": "exception", "detail": detail})


def _send(fd: int, result: dict) -> None:
    """把一份有界 JSON 结果写给父进程。"""
    body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")  # 保持协议紧凑
    os.write(fd, body[:RESULT_LIMIT])                                             # 最后一层硬限制，绝不回传无限文本


if __name__ == "__main__" and len(sys.argv) == 6 and sys.argv[1] == "--child":
    _child(Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))  # 参数全由上面的 run 生成
