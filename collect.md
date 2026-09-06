本文档适用于本机 user@user-MS-7E07 上的 RoboCasa +
SonicG1 + PICO + GR00T/LeRobot 三相机数据采集流程。

- RoboCasa 根目录：/home/user/robocasa
- RoboSuite 根目录：/home/user/robosuite
- GR00T 根目录：/home/user/GR00T-WholeBodyControl
- 数据根目录：/home/user/robocasa/datasets
- conda 根目录：/home/user/miniforge3
- 65 个原子任务参数：见第 9 节
- 默认 viewer：机器人头部第一视角
- VLA 数据相机：头部、左腕、右腕三路相机
> 重要： 不要同时运行
gear_sonic/scripts/run_sim_loop.py。本流程中 RoboCasa 是唯一的
MuJoCo/DDS 仿真端，同时启动两个 sim loop 会造成 DDS 状态冲突。

1. 四个终端总览

终端
进程
职责
关键端口
1
RoboCasa collector
MuJoCo 仿真、DDS LowState、三路相机、raw/HDF5
5555、5580、5581
2
PICO manager
人体姿态、planner、录制控制状态
5556
3
C++ SONIC deploy
G1 全身策略、DDS LowCmd、机器人状态与配置
5557
4
LeRobot exporter
汇总图像、人体和机器人状态，生成 LeRobot 数据集
连接 5555/5556/5557/5580/5581

推荐启动顺序：

终端 1 RoboCasa -> 终端 2 PICO manager -> 终端 3 SONIC deploy -> 终端 4 exporter

终端 4 必须放在 deploy 输出 robot_config 之后启动。

1.1 四个终端快速启动命令（本地路径）

以下命令按终端 1 -> 终端 2 -> 终端 3 -> 终端 4 启动。默认示例为
PickPlaceCounterToStove。为了让同一模板覆盖当前全部 65 个原子任务，
默认使用按仓库排除规则均合法的 layout 4 / style 1。采集其他任务时，
从第 9 节复制 TASK_ENV 和 TASK_PROMPT；DATASET_NAME 由 TASK_ENV 自动生成。

终端 1：RoboCasa

```bash

source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate robocasa
cd ~/robocasa

python robocasa/scripts/collect_sonic_demos.py \
  --environment  \
  --layout 1 \
  --robot SonicG1 \
  --render-camera robot0_head_camera \
  --out /home/user/robocasa/datasets/sonic_raw \
  --vla-stream \
  --vla-camera-names \
    robot0_head_camera \
    robot0_left_wrist_camera \
    robot0_right_wrist_camera \
  --vla-camera-keys \
    ego_view \
    left_wrist \
    right_wrist
```

终端 2：PICO

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl

cd "$GROOT_WBC_ROOT"
source .venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

终端 3：policy

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl

cd "$GROOT_WBC_ROOT/gear_sonic_deploy"
source scripts/setup_env.sh

./deploy.sh \
  --input-type zmq_manager \
  --output-type zmq \
  sim
```

终端 4：LeRobot exporter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToStove
export TASK_PROMPT="Pick the item from the plate and place it in the pan on the stove."
export DATASET_NAME="robocasa_${TASK_ENV}_g1_3cam"

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

`--task-prompt` 现在只作为兜底值。终端 1 会在每次 RoboCasa reset 成功后，
把本次实际生成的 `ep_meta.lang` 通过 5581 发给 exporter；录制时再发送带
episode ID 的 `start` attempt。exporter 匹配成功后才进入
RECORDING，并在首帧前锁定该 instruction。因此 `LoadDishwasher` 本次若采样到 cup 和 bowl，
LeRobot 中保存的就是
`Pick up the cup and bowl from the counter, place them in the dishwasher, and close the dishwasher door.`，
无需再手工复制动态对象名称。开始录制时应看到 exporter 输出
`[EpisodeInstruction] start matched ...` 和
`[EpisodeInstruction] recording task from RoboCasa ...`。旧版 collector 没有
5581 通道时，在 exporter 增加 `--no-sync-robocasa-instruction`，即可直接使用
命令行的 `--task-prompt`。

采集结束后清洗数据：

```bash
python gear_sonic/scripts/process_dataset.py \
  --dataset-path "$DATA_ROOT/$DATASET_NAME" \
  --output-path "$DATA_ROOT/${DATASET_NAME}_cleaned" \
  --remove-discarded \
  --no-remove-stale-smpl
```

2. 启动前检查

确认以下条件：

- PICO 已连接 XRoboToolkit，人体追踪数据可用。
- 四个端口没有被旧进程占用。
- RoboCasa、RoboSuite 和 GR00T 的资产及虚拟环境已经安装。
- 交互采集使用图形桌面，没有设置 MUJOCO_GL=egl。
检查端口和残留进程：

ss -lptn | grep -E ':(5555|5556|5557|5580|5581)\b' || true

pgrep -af \
  'run_sim_loop.py|collect_sonic_demos.py|g1_deploy_onnx_ref|pico_manager_thread_server.py|run_data_exporter.py' \
  || true

如果有输出，先核对 PID 和进程归属。不要直接批量杀进程。

3. 终端 1：启动 RoboCasa 三相机采集器

以下命令仍以 CloseFridge 为联调示例；采集其他任务时使用第 9 节对应的
TASK_ENV 和 TASK_PROMPT，并保持通用 TASK_LAYOUT=4、TASK_STYLE=1。

cd /home/user/robocasa
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate robocasa

python robocasa/scripts/collect_sonic_demos.py \
  --environment CloseFridge \
  --layout 4 \
  --style 1 \
  --robot SonicG1 \
  --render-camera robot0_head_camera \
  --out /home/user/robocasa/datasets/sonic_raw \
  --vla-stream \
  --vla-camera-names \
    robot0_head_camera \
    robot0_left_wrist_camera \
    robot0_right_wrist_camera \
  --vla-camera-keys \
    ego_view \
    left_wrist \
    right_wrist

正常日志应包含：

[sonic] 200 Hz physics
[sonic-vla] publishing ego_view,left_wrist,right_wrist
[sonic] dataset dir: ...
[sonic] hotkeys: c=record k=save x=discard b=band

Viewer 视角切换

--render-camera 只控制交互窗口，不改变写入数据集的 VLA 相机。

第一视角：

--render-camera robot0_head_camera

机器人随身前方视角：

--render-camera robot0_frontview

真正写入数据集的相机由 --vla-camera-names 和
--vla-camera-keys 决定。当前脚本默认不垂直翻转图像。

4. 终端 2：启动 PICO manager

cd /home/user/GR00T-WholeBodyControl
source .venv_teleop/bin/activate

python gear_sonic/scripts/pico_manager_thread_server.py --manager

PICO 未连通时出现以下日志属于正常等待：

Waiting for body tracking data...

连通后应看到：

[Manager] ZMQ socket bound to port 5556
[Manager] StreamMode switch: ...

首次联调需要观察人体骨架时，可以使用：

python gear_sonic/scripts/pico_manager_thread_server.py \
  --manager \
  --vis_vr3pt \
  --vis_smpl

5. 终端 3：启动 C++ SONIC 全身控制器

cd /home/user/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh

./deploy.sh \
  --input-type zmq_manager \
  --output-type zmq \
  sim

操作要求：

1. 出现 Proceed with deployment? [Y/n] 时按回车或输入 Y。
2. 等待终端输出大小写完全一致的 Init Done。
3. 保持该进程运行。
显式设置 --output-type zmq，确保无 ROS exporter 能从端口 5557
收到 g1_debug 和 robot_config。

6. 终端 4：启动 GR00T/LeRobot exporter

先确认：

- 终端 3 已输出 Init Done。
- 终端 2 已绑定端口 5556。
- 终端 1 已发布三路相机。
启动三相机 exporter：

cd /home/user/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "Close the fridge door or doors." \
  --dataset-name robocasa_CloseFridge_g1_3cam \
  --root-output-dir /home/user/robocasa/datasets \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech

三相机模式必须保留 --record-wrist-cameras。

正常日志应包含：

[Sonic] Connected to ZMQ at localhost:5556
[Sonic] Subscribed to: pose, planner, manager_state
Recording to /home/user/robocasa/datasets/...

exporter 启动时会等待端口 5557 的 robot_config。如果一直等待，优先检查
终端 3 和 --output-type zmq。

7. 开始控制与录制

7.1 校准和进入 POSE

1. 人保持校准姿势：直立、双脚并拢、上臂贴身、前臂向前弯曲约 90 度。
2. PICO 同时按 A+B+X+Y，启动策略并执行首次完整校准。
3. 机器人收到真实 SONIC 命令并稳定后，在终端 1 按 b 解除 startup band。
4. 人体姿态先与机器人对齐，再按 A+X 进入 POSE 全身跟踪。
5. 使用本地键盘或 PICO 录制组合键开始 episode。
解除 band 前必须确认机器人已稳定。姿态未对齐时不要直接切入 POSE。

7.2 终端 1 本地按键

按键
动作
说明
b
切换 startup band
平衡后解除；需要保护时可重新挂起
c
开始录制
同步通知 RoboCasa 与 exporter
k
保存 episode
结束当前录制并保存两侧数据
x
丢弃 episode
丢弃当前录制并同步 exporter

7.3 PICO 操作

组合键
动作
注意事项
Left Grip + A
开始/停止并保存
再次按下结束当前 episode
Left Grip + B
丢弃当前 episode
仅在确认数据不可用时执行
Trigger
控制对应手抓握
左右手分别控制
A + X
PLANNER <-> POSE
切换前先对齐人体和机器人姿态
A+B+X+Y
启动/停止策略
进入 OFF 也可作为紧急停止

8. 正确停止流程

不要在录制中直接对四个终端执行 Ctrl+C。RoboCasa 可能保存本地
episode，但 exporter 不一定收到一致的结束状态。

正确顺序：
1. 仍在录制时，先按 k / Left Grip+A 保存，或按
x / Left Grip+B 丢弃。
2. 按 PICO A+B+X+Y 进入 OFF。紧急情况下可在 C++ 终端按大写 O。
3. 依次在终端 4、终端 3、终端 2、终端 1 按 Ctrl+C。
4. 检查 raw/HDF5 与 LeRobot 两个输出目录。

9. 全部 65 个原子任务

本节清单以当前仓库的 atomic task 注册表和生成的 Atomic Tasks 索引为准，
共 14 类 fixture、65 个任务。清单包含 PackDessert，不包含仍归类为
composite task 的 OrganizeMugsByHandle。

为便于一套命令覆盖全部任务，下面统一使用：

- TASK_LAYOUT=4
- TASK_STYLE=1

layout 4 / style 1 对当前 65 个任务的类级 EXCLUDE_LAYOUTS 和
EXCLUDE_STYLES 规则均合法，但这不等同于完成了 65 项逐任务的真实遥操作
验证。每个任务正式录制前，先做一次不录制的 reset，确认目标 fixture、
物体生成、机器人可达性和三路相机画面正常。

TASK_PROMPT 使用任务级通用描述。部分任务会在每次 reset 后随机改变物体、
左右方向、温度、rack 层级、burner 或导航目标；实际操作必须以终端 1
打印的 Instruction: 为准。不要把某一个随机分支的具体描述当成整批数据集
的固定 TASK_PROMPT。

原文中的“已采集”和“难”状态已保留在备注列；未标状态不代表已验证。

9.1 任务切换方法

终端 1 和终端 4 是两个独立 shell，变量必须分别设置。

终端 1：从下表选择 TASK_ENV，然后执行第 1.1 节的 RoboCasa 命令。

~~~bash
export TASK_ENV=CloseFridge
export TASK_LAYOUT=4
export TASK_STYLE=1
~~~

终端 4：从同一行复制 TASK_ENV 和 TASK_PROMPT。DATASET_NAME 统一由
TASK_ENV 生成，避免任务名不一致。

~~~bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseFridge
export TASK_PROMPT="Close the fridge door or doors."
export DATASET_NAME="robocasa_${TASK_ENV}_g1_3cam"

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
~~~

终端 2 PICO manager 和终端 3 SONIC deploy 与任务无关，不需要修改。

9.2 65 个任务参数表

Blender（3）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 1 | CloseBlenderLid | Close the blender lid by securely placing it on top. | 难（原文标记） |
| 2 | OpenBlenderLid | Open the blender by taking off the lid and placing it on the counter. | — |
| 3 | TurnOnBlender | Turn on the blender by pressing the power button. | — |

Coffee Machine（3）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 4 | CoffeeServeMug | Pick the mug from under the coffee machine dispenser and place it on the counter. | mug 的具体描述 |
| 5 | CoffeeSetupMug | Pick the mug from the counter and place it under the coffee machine dispenser. | 难（原文标记）；mug 的具体描述 |
| 6 | StartCoffeeMachine | Press the button on the coffee machine to serve coffee. | — |

Doors（12）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 7 | CloseCabinet | Close the cabinet door or doors. | 单门 / 双门 fixture |
| 8 | CloseDishwasher | Close the dishwasher door. | — |
| 9 | CloseFridge | Close the fridge door or doors. | 已采集；冰箱类型 |
| 10 | CloseMicrowave | Close the microwave door. | — |
| 11 | CloseOven | Close the oven door. | — |
| 12 | CloseToasterOvenDoor | Close the toaster oven door. | 已采集 |
| 13 | OpenCabinet | Open the cabinet door or doors. | 单门 / 双门 fixture |
| 14 | OpenDishwasher | Open the dishwasher door. | — |
| 15 | OpenFridge | Open the fridge door or doors. | 冰箱类型 |
| 16 | OpenMicrowave | Open the microwave door. | — |
| 17 | OpenOven | Open the oven door. | — |
| 18 | OpenToasterOvenDoor | Open the toaster oven door. | — |

Drawer（5）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 19 | CloseDrawer | Close the designated drawer. | left / right |
| 20 | CloseFridgeDrawer | Fully close the fridge drawer. | — |
| 21 | OpenDrawer | Open the designated drawer. | left / right |
| 22 | OpenFridgeDrawer | Fully open the fridge drawer. | — |
| 23 | SlideDishwasherRack | Fully slide the top dishwasher rack in the instructed direction. | in / out |

Electric Kettle（3）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 24 | CloseElectricKettleLid | Close the lid of the electric kettle. | — |
| 25 | OpenElectricKettleLid | Press the button to open the lid of the electric kettle. | — |
| 26 | TurnOnElectricKettle | Press down the lever to turn on the electric kettle. | — |

Microwave（2）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 27 | TurnOffMicrowave | Press the stop button on the microwave. | — |
| 28 | TurnOnMicrowave | Press the start button on the microwave. | — |

Navigation（1）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 29 | NavigateKitchen | Navigate to the target kitchen location. | 目标 fixture / location |

Oven（2）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 30 | PreheatOven | Preheat the oven by turning the temperature knob. | — |
| 31 | SlideOvenRack | Fully slide the designated oven rack in the instructed direction. | top / bottom、in / out；单层烤箱无层级 |

Pick and Place（21）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 32 | CheesyBread | Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish. | — |
| 33 | MakeIcedCoffee | Pick up an ice cube and place it in the glass of coffee. | — |
| 34 | PackDessert | Add the dessert to the tupperware that contains the food. | dessert / cooked food 类型 |
| 35 | PickPlaceCabinetToCounter | Pick the item from the cabinet and place it on the counter. | item |
| 36 | PickPlaceCounterToBlender | Pick the item from the counter and place it in the blender. | item |
| 37 | PickPlaceCounterToCabinet | Pick the item from the counter and place it in the cabinet. | item |
| 38 | PickPlaceCounterToDrawer | Pick the item from the counter and place it in the drawer. | item |
| 39 | PickPlaceCounterToMicrowave | Pick the item from the counter and place it in the microwave. | item |
| 40 | PickPlaceCounterToOven | Place the item on the designated rack of the oven. | item、top / bottom；单层烤箱无层级 |
| 41 | PickPlaceCounterToSink | Pick the item from the counter and place it in the sink. | item |
| 42 | PickPlaceCounterToStandMixer | Place the item in the stand mixer bowl. | item |
| 43 | PickPlaceCounterToStove | Pick the item from the plate and place it in the pan on the stove. | 已采集；item / cookware |
| 44 | PickPlaceCounterToToasterOven | Place the item on the designated rack or tray of the toaster oven. | item、rack / tray、可选 top / bottom |
| 45 | PickPlaceDrawerToCounter | Pick the item from the drawer and place it on the counter. | item |
| 46 | PickPlaceFridgeDrawerToShelf | Pick the item from the fridge drawer and place it on a fridge shelf. | item / 目标 shelf |
| 47 | PickPlaceFridgeShelfToDrawer | Pick the item from the fridge shelf and place it in the fridge drawer. | item / 来源 shelf |
| 48 | PickPlaceMicrowaveToCounter | Pick the item from the microwave and place it in or on the target container on the counter. | item / container |
| 49 | PickPlaceSinkToCounter | Pick the item from the sink and place it in or on the target container on the counter. | item / container |
| 50 | PickPlaceStoveToCounter | Pick the item from the cookware on the stove and place it in or on the target container on the counter. | item / cookware / container |
| 51 | PickPlaceToasterOvenToCounter | Pick the item from the toaster oven and place it on the plate on the counter. | item、rack / tray、可选 top / bottom |
| 52 | PickPlaceToasterToCounter | Place the toasted item on a plate. | toasted item |

Sink（4）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 53 | AdjustWaterTemperature | Adjust the running sink water to the opposite temperature while keeping the water on. | cold to hot / hot to cold |
| 54 | TurnOffSinkFaucet | Turn off the sink faucet. | — |
| 55 | TurnOnSinkFaucet | Turn on the sink faucet. | — |
| 56 | TurnSinkSpout | Turn the sink spout in the instructed direction. | left / right |

Stand Mixer（2）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 57 | CloseStandMixerHead | Close the stand mixer head. | — |
| 58 | OpenStandMixerHead | Open the stand mixer head. | 难（原文标记） |

Stove（3）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 59 | LowerHeat | Lower the heat of the kettle. | 目标 burner / knob |
| 60 | TurnOffStove | Turn off the designated burner of the stove. | burner location |
| 61 | TurnOnStove | Turn on the designated burner of the stove. | burner location |

Toaster（1）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 62 | TurnOnToaster | Push down the lever of the toaster to turn it on. | — |

Toaster Oven（3）

| # | TASK_ENV | TASK_PROMPT | 状态 / episode 变量 |
|---:|---|---|---|
| 63 | AdjustToasterOvenTemperature | Adjust the toaster oven temperature in the instructed direction. | increase / decrease |
| 64 | SlideToasterOvenRack | Fully slide the designated toaster oven rack or tray in the instructed direction. | rack / tray、可选 top / bottom、in / out |
| 65 | TurnOnToasterOven | Turn on the toaster oven by setting the timer. | — |

数量校验：3 + 3 + 12 + 5 + 3 + 2 + 1 + 2 + 21 + 4 + 2 + 3 + 1 + 3 = 65。

9.3 65 个任务的终端 4 exporter 完整命令

下面每个命令块都可以在一个新的终端 4 中单独复制执行。每项均显式设置
GR00T 根目录、数据根目录、TASK_ENV、TASK_PROMPT 和 DATASET_NAME，
并统一保留 50 Hz、三相机和关闭语音播报参数。

### 1. CloseBlenderLid 难
```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets/sonic_vla_exports/composite_tasks/source_unfiltered
export TASK_ENV=ShakePan
export DATASET_NAME=robocasa_ShakePan_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras  --no-text-to-speech
```

### 2. OpenBlenderLid 难

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenBlenderLid
export TASK_PROMPT="Open the blender by taking off the lid and placing it on the counter."
export DATASET_NAME=robocasa_OpenBlenderLid_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 3. TurnOnBlender 难

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOnBlender
export TASK_PROMPT="Turn on the blender by pressing the power button."
export DATASET_NAME=robocasa_TurnOnBlender_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 4. CoffeeServeMug 难

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PortionFruits
export TASK_PROMPT="Place one apple and one peach from the bowl on each plate."
export DATASET_NAME=robocasa_PortionFruits_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 5. CoffeeSetupMug 难

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CoffeeSetupMug
export TASK_PROMPT="Pick the mug from the counter and place it under the coffee machine dispenser."
export DATASET_NAME=robocasa_CoffeeSetupMug_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 6. StartCoffeeMachine 难

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=StartCoffeeMachine
export TASK_PROMPT="Press the button on the coffee machine to serve coffee."
export DATASET_NAME=robocasa_StartCoffeeMachine_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 7. CloseCabinet 难

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseCabinet
export TASK_PROMPT="Close the cabinet door or doors."
export DATASET_NAME=robocasa_CloseCabinet_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 8. CloseDishwasher 难

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseDishwasher
export TASK_PROMPT="Close the dishwasher door."
export DATASET_NAME=robocasa_CloseDishwasher_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 9. CloseFridge 已采集

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseFridge
export TASK_PROMPT="Close the fridge door or doors."
export DATASET_NAME=robocasa_CloseFridge_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 10. CloseMicrowave 太高了

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseMicrowave
export TASK_PROMPT="Close the microwave door."
export DATASET_NAME=robocasa_CloseMicrowave_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 11. CloseOven

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseOven
export TASK_PROMPT="Close the oven door."
export DATASET_NAME=robocasa_CloseOven_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 12. CloseToasterOvenDoor 已采集

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseToasterOvenDoor
export TASK_PROMPT="Close the toaster oven door."
export DATASET_NAME=robocasa_CloseToasterOvenDoor_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 13. OpenCabinet 太高了

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenCabinet
export TASK_PROMPT="Open the cabinet door or doors."
export DATASET_NAME=robocasa_OpenCabinet_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 14. OpenDishwasher 已采集 需要筛除部分洗碗机，有些打不开

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenDishwasher
export TASK_PROMPT="Open the dishwasher door."
export DATASET_NAME=robocasa_OpenDishwasher_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 15. OpenFridge  已采集

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenFridge
export TASK_PROMPT="Open the fridge door or doors."
export DATASET_NAME=robocasa_OpenFridge_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 16. OpenMicrowave 太高了

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenMicrowave
export TASK_PROMPT="Open the microwave door."
export DATASET_NAME=robocasa_OpenMicrowave_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 17. OpenOven

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenOven
export TASK_PROMPT="Open the oven door."
export DATASET_NAME=robocasa_OpenOven_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 18. OpenToasterOvenDoor

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenToasterOvenDoor
export TASK_PROMPT="Open the toaster oven door."
export DATASET_NAME=robocasa_OpenToasterOvenDoor_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 19. CloseDrawer

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseDrawer
export TASK_PROMPT="Close the designated drawer."
export DATASET_NAME=robocasa_CloseDrawer_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 20. CloseFridgeDrawer

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseFridgeDrawer
export TASK_PROMPT="Fully close the fridge drawer."
export DATASET_NAME=robocasa_CloseFridgeDrawer_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 21. OpenDrawer

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenDrawer
export TASK_PROMPT="Open the designated drawer."
export DATASET_NAME=robocasa_OpenDrawer_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 22. OpenFridgeDrawer

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenFridgeDrawer
export TASK_PROMPT="Fully open the fridge drawer."
export DATASET_NAME=robocasa_OpenFridgeDrawer_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 23. SlideDishwasherRack

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=SlideDishwasherRack
export TASK_PROMPT="Fully slide the top dishwasher rack in the instructed direction."
export DATASET_NAME=robocasa_SlideDishwasherRack_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 24. CloseElectricKettleLid

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseElectricKettleLid
export TASK_PROMPT="Close the lid of the electric kettle."
export DATASET_NAME=robocasa_CloseElectricKettleLid_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 25. OpenElectricKettleLid

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenElectricKettleLid
export TASK_PROMPT="Press the button to open the lid of the electric kettle."
export DATASET_NAME=robocasa_OpenElectricKettleLid_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 26. TurnOnElectricKettle

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOnElectricKettle
export TASK_PROMPT="Press down the lever to turn on the electric kettle."
export DATASET_NAME=robocasa_TurnOnElectricKettle_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 27. TurnOffMicrowave

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOffMicrowave
export TASK_PROMPT="Press the stop button on the microwave."
export DATASET_NAME=robocasa_TurnOffMicrowave_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 28. TurnOnMicrowave

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOnMicrowave
export TASK_PROMPT="Press the start button on the microwave."
export DATASET_NAME=robocasa_TurnOnMicrowave_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 29. NavigateKitchen

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=NavigateKitchen
export TASK_PROMPT="Navigate to the target kitchen location."
export DATASET_NAME=robocasa_NavigateKitchen_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 30. PreheatOven

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PreheatOven
export TASK_PROMPT="Preheat the oven by turning the temperature knob."
export DATASET_NAME=robocasa_PreheatOven_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 31. SlideOvenRack

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=SlideOvenRack
export TASK_PROMPT="Fully slide the designated oven rack in the instructed direction."
export DATASET_NAME=robocasa_SlideOvenRack_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 32. CheesyBread

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CheesyBread
export TASK_PROMPT="Pick up the wedge of cheese and place it on the slice of bread to prepare a simple cheese on bread dish."
export DATASET_NAME=robocasa_CheesyBread_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 33. MakeIcedCoffee

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=MakeIcedCoffee
export TASK_PROMPT="Pick up an ice cube and place it in the glass of coffee."
export DATASET_NAME=robocasa_MakeIcedCoffee_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 34. PackDessert

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PackDessert
export TASK_PROMPT="Add the dessert to the tupperware that contains the food."
export DATASET_NAME=robocasa_PackDessert_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 35. PickPlaceCabinetToCounter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCabinetToCounter
export TASK_PROMPT="Pick the item from the cabinet and place it on the counter."
export DATASET_NAME=robocasa_PickPlaceCabinetToCounter_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 36. PickPlaceCounterToBlender

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToBlender
export TASK_PROMPT="Pick the item from the counter and place it in the blender."
export DATASET_NAME=robocasa_PickPlaceCounterToBlender_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 37. PickPlaceCounterToCabinet

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToCabinet
export TASK_PROMPT="Pick the item from the counter and place it in the cabinet."
export DATASET_NAME=robocasa_PickPlaceCounterToCabinet_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 38. PickPlaceCounterToDrawer

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToDrawer
export TASK_PROMPT="Pick the item from the counter and place it in the drawer."
export DATASET_NAME=robocasa_PickPlaceCounterToDrawer_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 39. PickPlaceCounterToMicrowave

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToMicrowave
export TASK_PROMPT="Pick the item from the counter and place it in the microwave."
export DATASET_NAME=robocasa_PickPlaceCounterToMicrowave_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 40. PickPlaceCounterToOven

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToOven
export TASK_PROMPT="Place the item on the designated rack of the oven."
export DATASET_NAME=robocasa_PickPlaceCounterToOven_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 41. PickPlaceCounterToSink

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToSink
export TASK_PROMPT="Pick the item from the counter and place it in the sink."
export DATASET_NAME=robocasa_PickPlaceCounterToSink_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 42. PickPlaceCounterToStandMixer

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToStandMixer
export TASK_PROMPT="Place the item in the stand mixer bowl."
export DATASET_NAME=robocasa_PickPlaceCounterToStandMixer_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 43. PickPlaceCounterToStove

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToStove
export TASK_PROMPT="Pick the item from the plate and place it in the pan on the stove."
export DATASET_NAME=robocasa_PickPlaceCounterToStove_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 44. PickPlaceCounterToToasterOven

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceCounterToToasterOven
export TASK_PROMPT="Place the item on the designated rack or tray of the toaster oven."
export DATASET_NAME=robocasa_PickPlaceCounterToToasterOven_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 45. PickPlaceDrawerToCounter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceDrawerToCounter
export TASK_PROMPT="Pick the item from the drawer and place it on the counter."
export DATASET_NAME=robocasa_PickPlaceDrawerToCounter_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 46. PickPlaceFridgeDrawerToShelf

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceFridgeDrawerToShelf
export TASK_PROMPT="Pick the item from the fridge drawer and place it on a fridge shelf."
export DATASET_NAME=robocasa_PickPlaceFridgeDrawerToShelf_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 47. PickPlaceFridgeShelfToDrawer

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceFridgeShelfToDrawer
export TASK_PROMPT="Pick the item from the fridge shelf and place it in the fridge drawer."
export DATASET_NAME=robocasa_PickPlaceFridgeShelfToDrawer_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 48. PickPlaceMicrowaveToCounter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceMicrowaveToCounter
export TASK_PROMPT="Pick the item from the microwave and place it in or on the target container on the counter."
export DATASET_NAME=robocasa_PickPlaceMicrowaveToCounter_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 49. PickPlaceSinkToCounter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceSinkToCounter
export TASK_PROMPT="Pick the item from the sink and place it in or on the target container on the counter."
export DATASET_NAME=robocasa_PickPlaceSinkToCounter_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 50. PickPlaceStoveToCounter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceStoveToCounter
export TASK_PROMPT="Pick the item from the cookware on the stove and place it in or on the target container on the counter."
export DATASET_NAME=robocasa_PickPlaceStoveToCounter_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 51. PickPlaceToasterOvenToCounter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceToasterOvenToCounter
export TASK_PROMPT="Pick the item from the toaster oven and place it on the plate on the counter."
export DATASET_NAME=robocasa_PickPlaceToasterOvenToCounter_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 52. PickPlaceToasterToCounter

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=PickPlaceToasterToCounter
export TASK_PROMPT="Place the toasted item on a plate."
export DATASET_NAME=robocasa_PickPlaceToasterToCounter_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 53. AdjustWaterTemperature

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=AdjustWaterTemperature
export TASK_PROMPT="Adjust the running sink water to the opposite temperature while keeping the water on."
export DATASET_NAME=robocasa_AdjustWaterTemperature_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 54. TurnOffSinkFaucet

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOffSinkFaucet
export TASK_PROMPT="Turn off the sink faucet."
export DATASET_NAME=robocasa_TurnOffSinkFaucet_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 55. TurnOnSinkFaucet

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOnSinkFaucet
export TASK_PROMPT="Turn on the sink faucet."
export DATASET_NAME=robocasa_TurnOnSinkFaucet_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 56. TurnSinkSpout

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnSinkSpout
export TASK_PROMPT="Turn the sink spout in the instructed direction."
export DATASET_NAME=robocasa_TurnSinkSpout_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 57. CloseStandMixerHead

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=CloseStandMixerHead
export TASK_PROMPT="Close the stand mixer head."
export DATASET_NAME=robocasa_CloseStandMixerHead_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 58. OpenStandMixerHead

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=OpenStandMixerHead
export TASK_PROMPT="Open the stand mixer head."
export DATASET_NAME=robocasa_OpenStandMixerHead_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 59. LowerHeat

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=LowerHeat
export TASK_PROMPT="Lower the heat of the kettle."
export DATASET_NAME=robocasa_LowerHeat_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 60. TurnOffStove

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOffStove
export TASK_PROMPT="Turn off the designated burner of the stove."
export DATASET_NAME=robocasa_TurnOffStove_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 61. TurnOnStove

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOnStove
export TASK_PROMPT="Turn on the designated burner of the stove."
export DATASET_NAME=robocasa_TurnOnStove_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 62. TurnOnToaster

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOnToaster
export TASK_PROMPT="Push down the lever of the toaster to turn it on."
export DATASET_NAME=robocasa_TurnOnToaster_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 63. AdjustToasterOvenTemperature

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=AdjustToasterOvenTemperature
export TASK_PROMPT="Adjust the toaster oven temperature in the instructed direction."
export DATASET_NAME=robocasa_AdjustToasterOvenTemperature_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 64. SlideToasterOvenRack

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=SlideToasterOvenRack
export TASK_PROMPT="Fully slide the designated toaster oven rack or tray in the instructed direction."
export DATASET_NAME=robocasa_SlideToasterOvenRack_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```

### 65. TurnOnToasterOven

```bash
export GROOT_WBC_ROOT=/home/user/GR00T-WholeBodyControl
export DATA_ROOT=/home/user/robocasa/datasets
export TASK_ENV=TurnOnToasterOven
export TASK_PROMPT="Turn on the toaster oven by setting the timer."
export DATASET_NAME=robocasa_TurnOnToasterOven_g1_3cam

cd "$GROOT_WBC_ROOT"
source .venv_data_collection/bin/activate

python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "$TASK_PROMPT" \
  --dataset-name "$DATASET_NAME" \
  --root-output-dir "$DATA_ROOT" \
  --data-collection-frequency 50 \
  --record-wrist-cameras \
  --no-text-to-speech
```
