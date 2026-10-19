# ResumeFilter

当前阶段只实现可断点续传的 PDF OCR。脚本会在本地把 PDF 逐页渲染为图片，并直接请求
SiliconFlow 的 `deepseek-ai/DeepSeek-OCR`，不会请求或依赖 OCR Gateway。

项目也支持按 `STANDARD.md` 对 OCR Markdown 进行模型筛选。筛选脚本随机抽样并逐份串行请求
OpenAI 兼容接口，保存随机种子、单份 JSON 和汇总 Markdown。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e "."
```

复制 `.env.example` 为 `.env` 并填写配置，程序会在当前工作目录自动加载该文件。也可以直接设置进程环境变量：

```powershell
$env:SILICONFLOW_API_KEY = "your-key"
$env:SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
```

默认从 `resumes/` 递归读取 PDF，并写入 `ocr_results/`：

```powershell
.\.venv\Scripts\python.exe .\ocr.py
```

也可以显式指定单个 PDF、输入目录和输出目录：

```powershell
.\.venv\Scripts\python.exe .\ocr.py .\incoming --output .\ocr_results
```

## 断点续传

- 每个成功页面立即原子写入 `ocr_results/.checkpoints/`。
- 只有所有页面成功后，才原子生成最终的 `<原文件名>.md`。
- `<原文件名>.ocr.json` 保存源文件 SHA-256、OCR 配置指纹和任务状态。
- 重跑时，源 PDF 和 OCR 配置没有变化的完整结果会跳过。
- 未完成文档会复用已有页面检查点，只请求缺失页面。
- 修改模型、提示词、渲染 DPI、图片格式或输出 token 上限后，配置指纹改变，旧结果不会被误判为当前结果。
- 页面失败事件会追加到 `ocr_results/failures.jsonl`；命令最终以非零状态退出，但继续处理后续 PDF。

处理单个 PDF 的页面默认最多并发 2 个请求。当前版本按 PDF 顺序处理，避免在大批量任务中瞬间放大
上游请求量。`Ctrl+C` 中断后可以直接运行同一命令继续。

## 随机筛选 OCR 简历

在 `.env` 中配置兼容接口（保留 `COMPATIABLE` 拼写以兼容现有配置）：

```powershell
OPENAI_COMPATIABLE_URL=https://api.deepseek.com
OPENAI_API_KEY=your-key
OPENAI_MODEL=deepseek-chat
```

`OPENAI_MODEL` 为必填项，筛选程序不会在缺失时静默选择默认模型。OCR 与筛选分别使用
`SILICONFLOW_OCR_MODEL` 和 `OPENAI_MODEL`，两者互不影响。

从 `ocr_results/` 的完整 Markdown 中随机抽取 5 份并串行筛选：

```powershell
.\.venv\Scripts\python.exe .\screen.py --count 5
```

命令会输出随机种子；可用 `--seed <整数>` 复现同一批抽样。结果默认写入带时间戳的
`screening_results/` 子目录。分页检查点不会进入抽样。

处理 `ocr_results/` 中的全部完整 OCR 结果：

```powershell
.\.venv\Scripts\python.exe .\screen.py --all --input .\ocr_results
```

`--all` 按稳定的文件名顺序串行处理。单份失败会记录到 `manifest.json` 和
`errors/`，并继续处理后续文件；分页检查点不会进入全量任务。程序启动时会立即打印本次
运行目录，例如 `screening_results/20261018-150000-123456`。

可以用 `Ctrl+C` 安全暂停，也可以主动限制本次新处理数量：

```powershell
.\.venv\Scripts\python.exe .\screen.py --all --max-items 100
```

每份成功结果都会立即原子写入 `results/`，进度则写入小型 `progress.json`。使用运行目录继续：

```powershell
.\.venv\Scripts\python.exe .\screen.py `
  --resume .\screening_results\20261018-150000-123456
```

恢复时会跳过已经成功的文件，并重试尚未完成或上次失败的文件。模型或 `STANDARD.md`
发生变化时拒绝继续旧任务，避免同一批结果使用不同标准。

筛选程序不再等待全量完成后生成大型汇总。可以在任务运行中、暂停后或完成后，从另一个终端
随时生成当前快照：

```powershell
.\.venv\Scripts\python.exe .\summarize.py `
  --run .\screening_results\20261018-150000-123456 `
  --top 50
```

该命令生成体积受控的 `summary.md`（进度、分布和最多 50 名优先候选人），以及包含所有
已完成候选人的 `results.csv`。详细证据始终保留在每份独立 JSON 中。

```text
screening_results/<运行时间>/
├── manifest.json       # 固定任务清单，不反复嵌入全部结果
├── progress.json       # 小型实时进度
├── results/0001.json   # 每份成功结果
├── errors/0002.json    # 当前或历史失败信息
├── summary.md          # summarize.py 按需生成
└── results.csv         # summarize.py 按需生成的完整平铺结果
```

也可以指定一份完整 OCR 结果单独筛选：

```powershell
.\.venv\Scripts\python.exe .\screen.py --file "候选人.md"
```

原始简历、OCR 和筛选结果可能包含个人信息，因此 `resumes/`、`ocr_results/` 与
`screening_results/` 默认不会被 Git 跟踪。

