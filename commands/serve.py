"""命令 serve：把一个模型起成 OpenAI 兼容的服务。

调用示例：
  ori serve out/merged --port 8080

眼下先核对模型目录和端口，网关接上后同一个入口换后端（vllm / sglang / llama.cpp）不用改命令。
"""

from pathlib import Path


def command(args) -> int:
    """核对模型目录与端口。"""
    if not Path(args.model).exists():  # 目录不在就别起
        print(f"模型目录不在: {args.model}")
        return 2
    if not 0 < args.port < 65536:  # 端口范围
        print(f"端口越界: {args.port}")
        return 2
    print(f"要在 {args.model} 上起服务，监听 :{args.port}")
    print("这一步还没接网关")
    return 0
