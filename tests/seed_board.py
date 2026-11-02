"""用已下载且固定 revision 的五个小模型生成公开榜真实启动数据。

调用：PATH=/path/to/llama.cpp/bin:$PATH python tests/seed_board.py --models .work/board-models --out .work/board-evaluation
下载目录每个模型含 config/tokenizer/model.safetensors/source.json。只量化和实际推理，绝不注入答案。
"""
import argparse                                              # 验收脚本参数不混入产品命令
import hashlib                                               # 榜单可核查的模型/题集配方摘要
import json                                                  # 结果与源版本记录
import sys                                                   # 直接运行 tests 文件时定位仓库
from datetime import datetime, timezone                      # 记录本次实际运行时刻
from pathlib import Path                                     # 全部产物由调用者指定位置
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 从工作树导入当前导出器
from features import board, forge                            # 使用实际产品的导出、判分与发布路径

NAMES = ('SmolLM2-135M-Instruct', 'SmolLM2-135M', 'SmolLM2-360M-Instruct', 'SmolLM2-360M', 'SmolLM-135M-Instruct')


def run(models: Path, out: Path) -> None:
    """逐个导出 q8_0 并实际评12题；第一模型同时生成 f16 对照自评。"""
    suite = Path(__file__).resolve().parents[1] / 'features/suites/en-smoke.yaml'
    inbox = out / 'inbox'                                    # board collect 只读取公开提交
    inbox.mkdir(parents=True, exist_ok=True)
    for number, name in enumerate(NAMES):
        source = models / name.lower()                      # 模型目录名与 revision 来源一起核对
        provenance = json.loads((source / 'source.json').read_text())
        target = out / name                                 # 大模型不进站点，只保留报告供发布
        target.mkdir(parents=True, exist_ok=True)
        recipe = {**provenance, 'quant': 'q8_0', 'suite_sha256': hashlib.sha256(suite.read_bytes()).hexdigest(), 'seed': 1, 'temperature': 0, 'max_tokens': forge.SELF_CHECK_TOKENS}
        digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        choice = forge.Choice(source=source, config=source/'config.json', out=target/'model.gguf', name=name, quant='q8_0', self_check=(number == 0), suites=[str(suite)])
        summary = forge.Export(choice).run()                 # 第一模型的 f16 与 q8_0 都由真实运行时推理
        result = forge.benchmark(choice.out, [str(suite)], name)
        result['model'] = provenance['repo']                 # 榜上给出完整模型仓库名，避免基座/指令版混淆
        result['recipe'] = 'sha256:' + digest
        result['provenance'] = recipe                        # 可核查模型 revision、题集摘要和解码参数
        result['time'] = datetime.now(timezone.utc).isoformat()
        (target/'benchmark.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        payload = {'model': result['model'], 'score': result['total'], 'recipe': result['recipe'], 'hardware': result['hardware'], 'time': result['time']}
        (inbox/f'board-{number}.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(name, result['total'], 'self_check', summary.get('self_check', {}).get('degradation'), flush=True)
    board.publish(inbox, out/'site')                         # 所有榜单行由同一个产品发布函数渲染


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.models, args.out)
