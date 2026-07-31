# MaixCAM2 拼图碎片识别

本项目用于在 MaixCAM2 上识别黑色 A4 纸上的白色拼图碎片。A版v2.2.0使用跨帧`GRAPH → FOUR_FAST → FALLBACK`流水线，并通过UART4协议版本3发送会话心跳、完整A4毫米边界和1～4片拼装结果。普通UNKNOWN不再用目标矩形毫米尺寸作硬门；未得到可靠解时会显示并发送`PLAN BEST !`，结果flags bit0提醒F4该拼法可能不准确。当前现场宏为92%严格门、85%受约束容错门和10mm伪短边清理。龙门架通信与运动控制仍由下位机接入。

当前保留原稳定版，并新增两个独立自动黑纸ROI版本：A版使用四边形掩膜，B版使用透视展开。完整安装、按钮和现场调参说明见 [A/B操作手册](docs/maixcam2-auto-roi-ab-guide.md)。

| 版本 | 目录 | 发布ZIP |
|---|---|---|
| 稳定版 | `maixcam2_app/` | `maix-diansai_1-v1.0.0.zip` |
| A 四边形掩膜与规划、UART4协议 | `maixcam2_app_A_quad/` | `diansai_quad-v2.2.0.zip` |
| B 透视展开 | `maixcam2_app_B_warp/` | `diansai_warp-v1.1.0.zip` |

> A版`v2.2.0`使用设置V5保存纸张方向。竖放映射210×297mm，横放映射297×210mm，黄色视觉区域默认等于完整蓝色A4区域；230×230mm机械行程由F4按实际零点独立限位。正常界面只显示标定A4纸面；选择模式/材料后必须点击`START`，连续3帧后锁定一次轮廓。WHITE依次执行整边`GRAPH_AUTO`、普通UNKNOWN内部四片`FOURFAST`和FALLBACK，独立`FOUR`仍保留自己的尺寸规则。

## 当前功能

| 模块 | 功能 |
|---|---|
| 图像分割 | Otsu自动阈值或固定阈值、开闭运算、面积过滤 |
| 几何提取 | 3至5个主要顶点、中心、方向角、边长和内角 |
| 未知模式 | 按位置稳定编号为`U1`至`U4` |
| 已知模式 | 登记并匹配`K1`至`K4`，低置信度显示`UNKNOWN` |
| 固定ROI | 在固定相机画面中裁掉亮地面和无效纸面，只识别黑纸有效区 |
| 自动黑纸ROI | 单次识别完整A4四角并自动判断横竖；失败保留旧ROI，可手动切换PAPER V/H |
| 纸面专用显示 | 正常界面把已标定A4等比例展开到屏幕，内容区外为黑色，不显示龙门架和地面 |
| A/B视觉路径 | A版四边形掩膜；B版420×594展开并裁取420×460工作区 |
| 现场调参 | `ROI`、`MASK`、`RESULT`、`MEASURE`和`ADV`预览，量化SCALE/GAP/JITTER/RECT及轮廓诊断 |
| 触摸操作 | 正常界面使用`KNOWN`、`UNKNOWN`、`WHITE/SAVE`、`START`、`CAL`，CAL界面直接调参数 |
| 参数持久化 | 只有`GOOD 4/4`时允许保存，重启后自动恢复现场参数 |
| 分离保存门 | 1～4片可`LOCK ROI`；ADV分割参数仍需`GOOD 4/4` |
| 视觉毫米区域 | 屏幕直接调X/Y/W/H/SPLIT/PAPER；竖放默认210×297mm，横放默认297×210mm，F4另做230×230mm机械限位 |
| 已知拼装 | 下半区保存一次正确100×60mm布局，运行时最多24种全局匹配 |
| 未知拼装 | 1～4片WHITE/CARD显式选择；WHITE依次运行32边/90组合GRAPH、四片分段接缝Beam和96宽FALLBACK；CARD保留T形分段接缝和扑克牌评分 |
| 倾斜相机映射 | 完整A4四角按V=210×297mm或H=297×210mm建立Homography，像素轮廓反算后再规划回绘 |
| 完全待机与单次快照 | 未START时不分析碎片；START后连续3帧深复制一次坐标和轮廓，求解、成功和失败均使用该快照，再点START才重拍 |
| PC回放 | 使用与设备相同的OpenCV核心处理保存的实拍图 |

## PC测试

安装测试依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

运行全部测试：

```powershell
python -m pytest -v
python -m compileall maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tests tests_ab tools
```

生成A/B发布包：

```powershell
python tools\package_variants.py
```

## 实拍图回放

使用Otsu自动阈值：

```powershell
python tools/replay_image.py "input.jpg" --output "tmp\replay.jpg"
```

固定阈值并限制ROI：

```powershell
python tools/replay_image.py "input.jpg" --output "tmp\replay.jpg" --roi 20 30 600 420 --threshold 180
```

终端会输出每片碎片的编号、顶点、中心、角度、面积和完整性JSON。叠加图中绿色为完整轮廓，橙色为接触ROI边界的不完整轮廓，红点为多边形顶点，黄色十字为中心。

## MaixCAM2运行

1. 在MaixCAM2设置中升级到最新运行库。
2. 使用MaixVision连接设备，直接打开`maixcam2_app`目录作为工程，确认下面7个Python文件都显示在左侧文件树中：`main.py`、`config.py`、`puzzle_vision.py`、`template_store.py`、`touch_ui.py`、`settings_store.py`、`calibration_ui.py`。
3. 直接运行`main.py`。程序同时兼容PC包目录和MaixVision的`/tmp/maixpy_run`平铺部署方式。
4. 固定相机并调整焦距，使黑色A4有效区域尽量充满画面。
5. 运行入口，默认选择`UNKNOWN WHITE`并显示`PRESS START`；点击`START`后才开始检测。

屏幕按键：

| 按键 | 作用 |
|---|---|
| `UNKNOWN` | 选择1～4片未知拼装模式，点击后回到待机 |
| `KNOWN` | 选择已保存模板K1～K4匹配模式，点击后回到待机 |
| `WHITE` / `CARD` | UNKNOWN下切换几何首解或花纹择优，切换后回到待机 |
| `SAVE` | KNOWN且已经START后，保存红线下方四片正确100×60mm布局 |
| `START` | 按当前模式和材料开始检测；再次点击会清空旧快照并重拍 |
| `CAL` | 进入或退出现场调参界面；返回正常页后必须重新点击START |

首次登记已知碎片时，在红线下方按正确关系摆成100×60mm，相邻片保留1～2mm黑缝，依次点击`KNOWN → START → SAVE`。A版同时保存稳定形状编号和目标局部轮廓到 `/root/maixcam2_puzzle_A/known_templates.json`；随机摆放回上半区后，再次点击`START`，连续稳定3帧即显示每片目标位置与旋转增量。

## 现场调参

相机和黑纸固定后，先把4片白色碎片互相分开并完整放入有效纸面，再按下面流程校准：

1. 点击`CAL`进入调参界面。
2. 在`ROI`页用`ITEM`选择`LEFT/RIGHT/TOP/BOTTOM`，用`-`和`+`裁掉亮地面、龙门架静态边缘及上下舍弃区域。
3. 点击`MASK`查看二值图。正确画面应只有4片碎片独立变白，黑纸保持黑色。
4. 用`ITEM`切换`TH/MIN/OPEN/CLOSE`；`STEP`循环1、5、10，控制单次调整量。
5. 点击`RESULT`查看轮廓分类：绿色有效、橙色触边、红色过小、紫色过大。
6. 状态达到`GOOD 4/4`后点击`SAVE`保存现场参数，再点击`CAL`返回正常界面。

相机允许固定倾斜，但A4纸、标准片和碎片顶面应近似共面。锁定蓝框后，把真实100×60mm标准片分别放在机械有效区中心、左上、右上、右下、左下，在`MEASURE`页记录五次`RECT`；五处长短边误差都应不超过约1.5～2mm。若中心合格而四周明显超差，问题是镜头径向畸变，重新调TH或机械毫米范围不能修复，需要做相机内参/畸变校正；若碎片顶面明显高于纸面且相机倾角较大，还需减小倾角或按碎片顶面重新标定以降低视差。

| 参数 | 屏幕操作 | 作用 |
|---|---|---|
| `ROI`四边 | 1/5/10像素步长 | 排除黑纸以外的亮背景和无效长边 |
| `TH` | `AUTO`或0至255 | 自动Otsu与固定灰度阈值切换 |
| `MIN` | 按ROI面积比例调整 | 过滤小噪声；过大会漏掉远距离小碎片 |
| `OPEN` | 1/3/5/7 | 去除小白点；过大会侵蚀小碎片 |
| `CLOSE` | 1/3/5/7/9 | 填补白片黑孔；过大会连接相邻碎片 |

现场参数保存到`/root/maixcam2_puzzle/vision_settings.json`。未达到`GOOD 4/4`时保存按钮禁用；未保存就退出CAL会丢弃本次修改。`known_match_threshold`等非现场参数仍在`maixcam2_app/config.py`中维护。

## 当前限制

- 输入碎片必须互不接触、互不重叠，符合题目启动时的随机摆放条件。
- 拼好后完全接触的同色碎片会合并为一个外轮廓，保存已知布局时需保留1～2mm黑缝。
- 龙门架遮挡尚未处理，正式启动识别前需要将执行机构停到固定停车位。
- 扑克牌图案已接入边缘颜色与梯度连续性评分，但尚未用现场牌面实物验证采样距离和成功率。
- Homography只校正共面透视，不校正镜头径向畸变，也不能消除碎片高度造成的视差。
- PC算法已经过自动测试；MaixCAM2摄像头、触摸屏、模板写入和帧率仍需实机验证。
