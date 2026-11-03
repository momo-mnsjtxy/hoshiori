# 仓库落位与名称占位

2026-11-03 查过一次占用：`github.com/hoshiori` 是个零仓库的个人号（login `HoshiOri`，2026-03 建），GitHub 用户名不分大小写，那条路径拿不回来了。PyPI 的 `hoshiori` 与 npm 的 `@hoshiori` scope 都还空着。

落位因此拆成两层：**仓库先挂个人账号，组织等有第二个维护者再转**。

## 现状

- 仓库：`momo-mnsjtxy/hoshiori`，**私有**，已推送。发布日改成 public 一条命令。
- 组织 `hoshiori-labs`：还没建，建好后转移。

## 发布日把仓库放开

```bash
gh repo edit momo-mnsjtxy/hoshiori --visibility public --accept-visibility-change-consequences
```

## 组织化（可选，早晚要做）

组织只能在网页建，CLI 建不了：

1. 打开 https://github.com/account/organizations/new ，选 Free
2. 组织名填 `hoshiori-labs`（可用，已查过）
3. 建好后把仓库转过去：

```bash
gh api -X POST repos/momo-mnsjtxy/hoshiori/transfer -f new_owner=hoshiori-labs
```

转移后星标、issue、PR 都跟着走，旧的 URL 会 301 过去。**确保 gh 的 token 重新授权过 `admin:org`**（`gh auth refresh -h github.com -s admin:org`），否则转移这一步会 403。

## 占位包：越早越好

名字空着就有人抢。两个都先传空壳，等 0.1.0 再覆盖。

### PyPI

```bash
mkdir -p /tmp/squat && cd /tmp/squat
cat > pyproject.toml <<'EOF'
[project]
name = "hoshiori"
version = "0.0.0"
description = "占位，正式包见 github.com/momo-mnsjtxy/hoshiori"
requires-python = ">=3.10"
EOF
python -m pip install -q build twine
python -m build
twine upload dist/*        # 需要先在 pypi.org 建 API token
```

上传完到 https://pypi.org/project/hoshiori/ 确认，顺手把 description 改成正经的一句话。

### npm scope

```bash
npm login                                  # 用要占名的账号
npm org create hoshiori                    # 已存在会报错，那就直接发包占住
mkdir -p /tmp/npm-squat && cd /tmp/npm-squat
npm init -y --scope=@hoshiori -w loom      # 造 @hoshiori/loom 的空包
npm publish --access public -w @hoshiori/loom
```

## 已经踩过的坑

- 仓库历史里有 130MB 的 `.bigcorpus`，GitHub 单文件限 100MB，第一次推直接被 pre-receive 拒。已用 `git filter-branch` 从全部历史里删掉，`.git` 从 47MB 缩到 928K。以后别把语料提交进去。
- `hoshiori-dev` 也被占，别再试。
- 备选名里 GitHub 的 `tsumugi`、`kagari`、`shizuku` 全被占；PyPI 的 `kagari`、`shizuku` 还空着，但包名与仓库名不一致会很难看，不如就用 `hoshiori` 加组织后缀。
