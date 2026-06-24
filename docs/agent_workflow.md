# RoboCasa Agent 工程工作流

本文档定义一套适用于本仓库的 agent 代理工作流，覆盖：

- 需求分解
- 仓库结构分析
- 具体实现
- 功能评审
- 代码审查
- 最终验收

目标不是让一个代理直接从需求跳到代码，而是把工程过程拆成多个可审计阶段。每个阶段都有明确输入、输出、退出条件和禁止事项。

## 1. 适用范围

本工作流适合以下类型的任务：

- 新增 RoboCasa 任务、场景、机器人模型或 wrapper。
- 修改数据采集、回放、评测、benchmark 或 policy learning 相关代码。
- 接入 Unitree G1、SONIC、GR00T、robomimic 等外部流程。
- 修复复杂 bug，尤其是涉及环境步进、控制器、数据格式、渲染或 demo 脚本的 bug。

不建议用于非常小的改动，例如拼写修正、README 小段落更新、单行参数调整。小改动可直接走普通 PR。

## 2. 仓库背景约束

RoboCasa 是一个用于训练和评测通用机器人的大规模仿真框架。当前 README 明确说明它支持大量任务、厨房场景、3D 物体、演示数据和 benchmark。工程改动需要避免破坏这些既有能力。

仓库的主要结构应按以下边界处理：

| 区域 | 责任 |
|---|---|
| `robocasa/environments/` | 环境与任务逻辑 |
| `robocasa/models/` | 机器人、场景、物体模型 |
| `robocasa/wrappers/` | 环境包装与数据接口 |
| `robocasa/demos/` | demo 和交互入口 |
| `robocasa/scripts/` | CLI 工具与批处理脚本 |
| `robocasa/utils/` | 通用工具与算法胶水代码 |
| `tests/` | 自动化测试 |
| `docs/` | 文档 |

关键原则：

- 优先复用现有 RoboCasa / robosuite 抽象。
- 不绕过 `env.step`、wrapper、controller、dataset pipeline 等既有约定，除非需求文档中明确说明原因。
- 大型资产和数据集不应随普通功能 PR 提交。
- 实现代理只做已批准范围内的最小变更。

## 3. 角色设计

### 3.1 需求统筹与项目总控代理

建议模型：`GPT-5.5 Thinking / x-high`

职责：

1. 把自然语言需求转成可执行需求文档。
2. 判断是否需要拆成多个 PR。
3. 生成任务分解、验收标准和风险清单。
4. 控制实现代理的范围。
5. 做最终合并建议。

禁止：

- 在需求不清楚时直接要求实现代理改代码。
- 接受没有测试证据的实现结果。
- 把架构决策交给实现代理临场发挥。

输出：

- `requirements.md`
- `implementation_plan.md`
- `final_validation.md`

### 3.2 仓库结构与约定分析代理

建议模型：`GPT-5.5 Thinking / high`

职责：

1. 定位相关模块和调用链。
2. 找到应复用的接口、wrapper、controller 或脚本。
3. 给出最小变更面。
4. 标出不应修改的区域。

输出：

- `repo_context.md`
- `change_surface.md`

### 3.3 具体实现代理

建议模型：`GPT-5.3 Spark / medium-high`

职责：

1. 严格按实现计划改代码。
2. 每次只做一个明确任务。
3. 补充最小测试或 smoke test。
4. 记录实际改动、未完成项和测试结果。

禁止：

- 为了方便绕过现有抽象。
- 未经批准重构大模块。
- 引入重型依赖。
- 修改无关资产、数据集、二进制文件。

输出：

- 代码 diff
- `implementation_notes.md`
- `test_log.md`

### 3.4 功能评审代理

建议模型：`GPT-5.5 Thinking / high`

职责：

1. 按需求验收功能，而不是只看代码风格。
2. 复现最小 demo 或测试命令。
3. 检查失败路径和边界条件。
4. 给出通过、条件通过或不通过结论。

输出：

- `functional_review.md`

### 3.5 代码审查代理

建议模型：`GPT-5.5 Thinking / high`

职责：

1. 审查 PR diff。
2. 检查架构一致性。
3. 检查可维护性、错误处理、测试覆盖。
4. 重点检查是否绕过 RoboCasa / robosuite 既有约定。

输出：

- `code_review.md`
- PR review comments

## 4. 阶段门禁

| Gate | 名称 | 负责人 | 通过条件 |
|---|---|---|---|
| G0 | 需求冻结 | 总控代理 | 需求、非目标、验收标准明确 |
| G1 | 方案批准 | 总控代理 | 变更面、风险、测试计划明确 |
| G2 | 实现完成 | 实现代理 | diff 完成，测试记录完整 |
| G3 | 功能验收 | 功能评审代理 | 行为满足需求 |
| G4 | 代码审查 | 代码审查代理 | 架构、质量、测试通过 |
| G5 | 最终合并建议 | 总控代理 | 所有阻塞项关闭 |

任何一个 gate 未通过，都必须返回前一阶段修正，不允许继续推进。

## 5. 推荐目录结构

```text
.agent-workflow/
  agents.json
  templates/
    00_requirement_intake.md
    01_repo_analysis.md
    02_implementation_plan.md
    03_implementation_notes.md
    04_functional_review.md
    05_code_review.md
    06_final_validation.md

.agent-runs/
  <feature-name>/
    requirements.md
    repo_context.md
    implementation_plan.md
    implementation_notes.md
    test_log.md
    functional_review.md
    code_review.md
    final_validation.md
```

`.agent-workflow/` 是模板和配置，应提交到仓库。`.agent-runs/` 是每次任务的运行产物，通常不提交，除非需要把完整工程记录放进 PR。

## 6. 使用方式

初始化一次任务：

```bash
python -m robocasa.scripts.agent_workflow init \
  --feature "add-unitree-g1-sonic-bridge" \
  --summary "接入 Unitree G1 的 SONIC 数据采集桥接流程"
```

查看角色配置：

```bash
python -m robocasa.scripts.agent_workflow agents
```

查看 gate 状态：

```bash
python -m robocasa.scripts.agent_workflow status \
  --run .agent-runs/add-unitree-g1-sonic-bridge
```

## 7. 标准执行顺序

### Step 1：需求统筹代理生成 `requirements.md`

输入：

- 用户原始需求。
- 相关 issue、PR、论文、设计文档。
- 当前仓库约束。

必须写清楚：

- 背景。
- 目标。
- 非目标。
- 用户可观察行为。
- 兼容性要求。
- 验收标准。
- 风险。

### Step 2：仓库结构代理生成 `repo_context.md`

必须回答：

- 应该改哪些文件。
- 不应该改哪些文件。
- 有哪些已有工具可复用。
- 哪些接口不能绕过。
- 需要增加哪些测试。

### Step 3：总控代理生成 `implementation_plan.md`

计划必须拆成可执行小任务。

每个任务包含：

- 修改文件。
- 具体动作。
- 预期 diff 类型。
- 验证命令。
- 回滚方式。

### Step 4：实现代理改代码

实现代理每轮只处理一个任务。

每轮结束必须更新：

- 改了什么。
- 没改什么。
- 跑了什么命令。
- 命令结果。
- 是否偏离计划。

### Step 5：功能评审代理验收

功能评审代理不重点看代码风格，而是验证功能是否符合需求。

必须给出：

- 通过项。
- 失败项。
- 无法验证项。
- 需要补充的测试。

### Step 6：代码审查代理审 PR

代码审查代理必须重点检查：

- 是否复用现有抽象。
- 是否绕过环境、wrapper、controller 或 dataset pipeline。
- 是否引入难维护的特殊路径。
- 是否影响已有 benchmark 或 demo。
- 是否有足够测试。

### Step 7：总控代理最终验收

总控代理汇总：

- 需求是否完成。
- 测试是否足够。
- 风险是否可接受。
- 是否建议合并。

## 8. RoboCasa 专用审查清单

### 环境与任务

- 新任务是否遵守现有 task 组织方式。
- reset、success check、reward、object placement 是否清晰。
- 是否会破坏已有 task registry。

### 控制与机器人

- 是否复用 robosuite controller / composite controller。
- 是否有充分理由绕过默认控制路径。
- action space、observation space 是否稳定。

### 数据采集与回放

- 是否复用已有 data collection wrapper。
- 是否复用已有 state replay / obs extraction pipeline。
- 数据格式是否兼容下游 policy learning。

### 评测与 benchmark

- horizon、seed、split、success metric 是否明确。
- 是否影响 leaderboard 或已有评测协议。

### 文档与 demo

- 新入口是否有最小使用示例。
- 是否说明依赖、资产、运行命令。

## 9. 默认验证命令

根据改动大小选择命令，不要求每次全部运行。

```bash
python -m compileall robocasa tests
python -m pytest tests -q
python -m robocasa.scripts.setup_macros
python -m robocasa.demos.demo_tasks
python -m robocasa.demos.demo_kitchen_scenes
python -m robocasa.demos.demo_objects
```

涉及 MuJoCo 渲染、macOS 或硬件设备时，测试记录必须说明机器环境和无法运行的原因。

## 10. 推荐 PR 描述结构

```md
## Summary
- 

## Scope
- 

## Non-goals
- 

## Implementation Notes
- 

## Validation
- [ ] python -m compileall robocasa tests
- [ ] python -m pytest tests -q
- [ ] relevant demo / script

## Agent Workflow Artifacts
- Requirements:
- Repo analysis:
- Functional review:
- Code review:

## Risks
- 
```

## 11. 推荐工作方式

建议采用如下模型分工：

- `GPT-5.5 Thinking / x-high`：总控、需求分解、最终验收。
- `GPT-5.5 Thinking / high`：仓库分析、功能评审、代码审查。
- `GPT-5.3 Spark / medium-high`：具体代码修改、补测试、修 lint。

这套分工的关键是：高推理模型负责判断和约束，快速实现模型负责执行。实现模型不能同时担任最终评审者。
