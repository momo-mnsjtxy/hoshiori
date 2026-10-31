"""命令 judge：拿一个答题后端跑评测套件，判分、落三份报告，给了基线就过回归门。

调用示例：
  ori judge sft.warp.yaml --suite zh-lite --answer-cmd 'echo {prompt}'
  ori judge sft.warp.yaml --suite format-lite --limit 3 --out out/r
  ori judge sft.warp.yaml --suite zh-lite --baseline out/base/report.json

--answer-cmd 里的 {prompt} 会换成题面，所以不绑死任何模型后端；
不给就退回回显占位，把题面当答案，报告里标 backend: echo。
检查只在这里做一次：配方在不在、--limit 是不是负数、套件认不认得出、基线在不在。
"""

import subprocess
from pathlib import Path

from features import gate, recipe, report, scorer, suite
from utils.fail import fail

TIMEOUT = 300  # 一道题的答案命令最多等几秒，卡死要能自己出来


def command(args) -> int:
    """跑套件、判分、落报告；给了基线就比完返回门的退出码。"""
    if not args.recipe:                                                       # 没给配方
        fail("input", "要给一份配方", hint="例: ori judge sft.warp.yaml --suite zh-lite")
    if args.limit < 0:                                                        # 限题数写负了
        fail("input", "--limit 不能是负数", hint="给 0 或正整数，0 表示不限")

    data = recipe.load(Path(args.recipe))                                     # 读配方，坏 yaml 在这里停
    recipe.check(data)                                                        # 先过校验
    names = args.suite or (data.get("eval") or {}).get("suites") or []        # 命令行优先，配方兜底
    if not names:                                                             # 两头都没给
        fail("input", "没给套件，配方里也没有 eval.suites", hint="例: --suite zh-lite")
    found = [(name, suite.locate(suite.folder(), name)) for name in names]    # 先全部找到，缺一套当场停
    baseline = Path(args.baseline) if args.baseline else None                 # 给没给基线
    if baseline:                                                              # 给了就先确认在不在
        gate.load(baseline)                                                   # 顺便验一下这份基线能不能读
    min_delta = gate_allowed(data, args.min_delta)                            # 允许掉多少分

    rows = []                                                                 # 逐题结果
    for _, path in found:                                                     # 一套一套跑
        one = suite.read(path)                                                # 坏套件在这里停
        tasks = one.tasks[: args.limit] if args.limit else one.tasks          # 只跑前 limit 题
        print(f"套件 {one.name}（{len(tasks)} 题）：{one.desc}")
        for task in tasks:                                                    # 一题一题来
            text = answer(task.prompt, args.answer_cmd or "")                 # 拿答案
            verdict = scorer.grade(task, text)                                # 判分
            rows.append({                                                     # 这一题的结果
                "suite": one.name,                                            # 哪一套
                "id": task.id,                                                # 哪一题
                "score": verdict.score,                                       # 得分
                "reason": verdict.reason,                                     # 为什么得这个分
                "answer": text.strip(),                                       # 答的是什么
            })
            print(f"  {task.id}  {verdict.score:.2f}  {verdict.reason}")

    payload = report.make(                                                    # 一份报告的数据，三份产物都由它派生
        model=data.get("name") or Path(args.recipe).stem,                     # 被评的名字
        backend=args.answer_cmd or "echo",                                    # echo 说明答案只是回显
        suite=", ".join(names),                                               # 跑了哪几套
        recipe=recipe.stamp(data),                                            # 配方哈希
        rows=rows,                                                            # 逐题结果
    )
    paths = report.save(payload, Path(args.out))                              # 落 markdown、json、csv
    print(f"\n总分 {payload['total']}（{payload['count']} 题）")
    print(f"报告 {paths['markdown']} · {paths['json']} · {paths['csv']}")
    if not baseline:                                                          # 没基线就到此为止
        return 0

    outcome = gate.check(payload, gate.load(baseline), min_delta)             # 跟基线比
    print(outcome.say())
    return outcome.exit_code                                                  # 掉了就非零


def gate_allowed(data: dict, given: float) -> float:
    """允许掉多少分。命令行给了就用命令行的，没给就用配方 eval.gate.min_delta，都没有算 0。"""
    if given is not None:  # 命令行优先
        return given
    return float(((data.get("eval") or {}).get("gate") or {}).get("min_delta", 0.0))


def answer(question: str, cmd: str) -> str:
    """拿一份答案。没给命令就把题面原样回显。"""
    if not cmd:                                                                                    # 没有后端，回显占位
        return question
    run = cmd.replace("{prompt}", question)                                                        # 占位换成题面
    try:
        done = subprocess.run(run, shell=True, capture_output=True, text=True, timeout=TIMEOUT)    # 走 shell，好拼管道
    except subprocess.TimeoutExpired:                                                              # 后端卡住了
        fail("input", f"答案命令 {TIMEOUT} 秒没跑完", where=run[:100])
    if done.returncode:                                                                            # 命令自己失败，别把空答案当成模型答错
        fail("input", f"答案命令退出码 {done.returncode}", where=run[:100], hint=done.stderr.strip()[:300])
    return done.stdout
