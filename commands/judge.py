"""命令 judge：拿一个答题后端跑评测套件，判分、落三份报告，给了基线就过回归门。

调用示例：
  ori judge sft.warp.yaml --suite zh-lite --answer-cmd 'echo {prompt}'
  ori judge sft.warp.yaml --suite format-lite --limit 3 --out out/r
  ori judge sft.warp.yaml --suite zh-lite --baseline out/base/report.json
  ori judge sft.warp.yaml --suite zh-lite --answer-cmd 'a {prompt}' --arena 'b {prompt}'
  ori judge sft.warp.yaml --suite zh-lite --answer-cmd 'a {prompt}' --submit https://example/board
  ori judge --import-lm-eval tasks/demo.yaml     # 只导题库，不跑评测

--answer-cmd 里的 {prompt} 会换成题面，所以不绑死任何模型后端；
不给就退回回显占位，把题面当答案，报告里标 backend: echo。
--arena 把 --answer-cmd 当 A 方、另一个命令当 B 方，逐题乱序对打并把排名写进 JSON；
--submit 在报告落盘后提交公开字段，断网时在报告旁留下 board-pending.json。
给了 --import-lm-eval 就只跑导入，不读配方、不评测。
检查只在这里开头做一次：选项能否同用、配方与套件能否读、基线是否有效。
"""

from pathlib import Path

from features import arena, board, gate, lm_eval, recipe, report, suite
from utils.fail import fail


def command(args) -> int:
    """跑套件、判分、落报告；给了基线就比完返回门的退出码。"""
    if args.import_lm_eval:                                                   # 给了就只导题
        if args.arena is not None or args.submit is not None:                 # 导入没有报告，也没有后端可供这两项使用
            fail("input", "--import-lm-eval 不能和 --arena / --submit 同用")
        return run_import(args.import_lm_eval)
    if not args.recipe:                                                       # 没给配方
        fail("input", "要给一份配方", hint="例: ori judge sft.warp.yaml --suite zh-lite")
    if args.limit < 0:                                                        # 限题数写负了
        fail("input", "--limit 不能是负数", hint="给 0 或正整数，0 表示不限")
    if args.arena is not None and not args.arena.strip():                     # 参数给了但命令是空白
        fail("input", "--arena 后面要给另一个答题命令")
    if args.arena is not None and not (args.answer_cmd or "").strip():        # A 方必须明确，避免拿回显误当模型
        fail("input", "--arena 要和 --answer-cmd 一起用", hint="--answer-cmd 是 A 方，--arena 是 B 方")

    data = recipe.load(Path(args.recipe))                                     # 读配方，坏 yaml 在这里停
    recipe.check(data)                                                        # 先过校验
    names = args.suite or (data.get("eval") or {}).get("suites") or []        # 命令行优先，配方兜底
    if not names:                                                             # 两头都没给
        fail("input", "没给套件，配方里也没有 eval.suites", hint="例: --suite zh-lite")
    found = [(name, suite.read(suite.locate(suite.folder(), name))) for name in names]  # 全部读好再启动后端
    baseline = Path(args.baseline) if args.baseline else None                 # 给没给基线
    if baseline:                                                              # 给了就先确认在不在
        gate.load(baseline)                                                   # 顺便验一下这份基线能不能读
    min_delta = gate_allowed(data, args.min_delta)                            # 允许掉多少分

    contest = None                                                            # 普通评测没有擂台附表
    if args.arena is not None:                                                # 两个后端走同一批题的成对擂台
        rows, contest = fight(found, args.answer_cmd, args.arena, args.limit)
    else:                                                                     # 一个后端走普通逐题评测
        rows = score(found, args.answer_cmd or "", args.limit)

    payload = report.make(                                                    # 一份报告的数据，三份产物都由它派生
        model=data.get("name") or Path(args.recipe).stem,                     # 被评的名字
        backend=args.answer_cmd or "echo",                                    # echo 说明答案只是回显
        suite=", ".join(names),                                               # 跑了哪几套
        recipe=recipe.stamp(data),                                            # 配方哈希
        rows=rows,                                                            # 逐题结果
    )
    if contest:                                                               # 擂台明细只在走擂台时追加
        payload["arena"] = contest                                           # JSON 保留两方答案、赛果与排名
    paths = report.save(payload, Path(args.out))                              # 落 markdown、json、csv
    print(f"\n总分 {payload['total']}（{payload['count']} 题）")
    print(f"报告 {paths['markdown']} · {paths['json']} · {paths['csv']}")
    if args.submit is not None:                                               # 用户明确要求才碰公开榜
        submitted = board.submit(paths["json"], args.submit)                  # 发不出会自动落待补文件
        if submitted["submitted"]:                                           # 服务已经接收
            print(f"公开榜 已提交 {args.submit}")
        else:                                                                 # 没网或没 URL，告诉用户去哪补
            print(f"公开榜 待补 {submitted['pending']}")
    if not baseline:                                                          # 没基线就到此为止
        return 0

    outcome = gate.check(payload, gate.load(baseline), min_delta)             # 跟基线比
    print(outcome.say())
    return outcome.exit_code                                                  # 掉了就非零


def score(found: list, backend_cmd: str, limit: int) -> list:
    """让一个后端依次跑完已装载套件，并把逐题结果打印出来。"""
    rows = []                                                                 # 所有套件最后合成一份报告
    for _, one in found:                                                      # 一套一套跑，顺序跟命令行一致
        result = lm_eval.run_backend(backend_cmd, one, limit)                 # 真调用后端并判完整套题
        print(f"套件 {one.name}（{result['count']} 题）：{one.desc}")
        for row in result["tasks"]:                                          # 一题一行即时给人看
            print(f"  {row['id']}  {row['score']:.2f}  {row['reason']}")
        rows.extend(result["tasks"])                                         # 汇进最终三格式报告
    return rows                                                              # 已按套件、题目顺序排好


def fight(found: list, first_backend: str, second_backend: str, limit: int) -> tuple:
    """让两个后端对打，打印逐题比分和最终排名，并返回 A 方报告行与擂台附表。"""
    suites = [one for _, one in found]                                        # 路径名已经完成检查，擂台只需套件对象
    contest = arena.run(first_backend, second_backend, suites, limit)         # 两方乱序答同一批题
    rows = []                                                                 # 普通报告主体沿用 A 方结果，保持旧格式可读
    current = None                                                           # 打印时套件切换才出一次标题
    for match in contest["matches"]:                                         # 每道题就是一场对局
        if match["suite"] != current:                                        # 进入下一套题
            current = match["suite"]                                        # 记下正在打印哪套
            count = sum(one["suite"] == current for one in contest["matches"])  # 这一套实际打几场
            print(f"擂台 {current}（{count} 题）")
        first, second = match["players"]                                     # players 固定是 A、B，不受调用顺序影响
        print(f"  {match['id']}  A {first['score']:.2f} · B {second['score']:.2f}  顺序 {'→'.join(match['order'])}")
        rows.append({                                                         # A 方仍可走原有报告与回归门
            "suite": match["suite"],                                        # 哪一套
            "id": match["id"],                                              # 哪一题
            "score": first["score"],                                        # A 方得分
            "reason": first["reason"],                                      # A 方判分理由
            "answer": first["answer"],                                      # A 方实际答案
        })
    print("\n擂台排名（Elo / BT）")                                          # 两套强弱指标并排给人核对
    for row in contest["ranking"]:                                           # 已按 Elo、BT 从强到弱排好
        print(f"  {row['rank']}. {row['side']} {row['elo']:.2f} / {row['bt']:.4f}  {row['backend']}")
    return rows, contest                                                     # 主报告与擂台附表同源


def run_import(paths: list) -> int:
    """把 lm-eval 任务导成套件，打印每份的题数和丢弃规则数。"""
    results = lm_eval.import_all(paths, suite.folder())                       # 落到内置套件目录
    for one in results:                                                       # 一份一行回话
        print(one.say())
    print(f"\n共导 {sum(one.tasks for one in results)} 题，写了 {len(results)} 份套件。")
    dropped = sum(len(one.dropped) for one in results)                        # 丢弃的规则总数
    if dropped:                                                               # 有丢弃就指到报告
        print(f"有 {dropped} 条规则认不出，都写在各自套件旁边的 .report.md 里。")
    return 0


def gate_allowed(data: dict, given: float) -> float:
    """允许掉多少分。命令行给了就用命令行的，没给就用配方 eval.gate.min_delta，都没有算 0。"""
    if given is not None:  # 命令行优先
        return given
    return float(((data.get("eval") or {}).get("gate") or {}).get("min_delta", 0.0))
