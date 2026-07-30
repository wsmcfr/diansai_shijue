# MaixCAM2 A版FOUR_FAST独立活动预算设计

## 问题与证据

现场四片已经稳定锁定，GRAPH检查90组后无解；FOUR_FAST随后占满UNKNOWN统一8秒活动预算，在阶段完成前直接返回`solver_timeout`，因此本可在第229节点成功的FALLBACK没有得到执行机会。同一毫米顶点在PC上得到以下对照：

| 路径 | 结果 | 关键数据 |
|---|---|---|
| FOUR_FAST Beam 32 | 失败 | 2400单元，128个完整候选，123个尺寸拒绝 |
| FOURFAST Beam 96 | 成功但更慢 | 5220单元，填充93.3%，重叠1.1% |
| FALLBACK | 成功 | 第229节点，填充93.3%，重叠1.1% |

三片现场输入不进入FOURFAST，现有`GRAPH -> FALLBACK`在1.841秒累计活动计算、6.141秒墙钟内成功。优化必须保持这条路径和共享8秒预算不变。

## 方案比较

| 方案 | 优点 | 风险 | 结论 |
|---|---|---|---|
| 降低UNKNOWN全局8秒预算 | 实现简单 | 三片当前已使用1.841秒，过低会制造三片超时 | 不采用 |
| 降低FOURFAST工作单元上限 | 确定性强 | 同一单元在PC和MaixCAM2成本差异明显，不能稳定表达现场时间 | 仅保留2400硬保护，不作为本次主门 |
| 增加FOURFAST独立活动预算 | 只约束四片，按真实CPU时间适配设备，能保留FALLBACK时间 | 需要在跨帧任务与子生成器之间传递中止标志 | 采用 |

## 已确认设计

新增现场常量`UNKNOWN_FOUR_FAST_ACTIVE_BUDGET_SECONDS = 1.5`。它只在`len(pieces) == 4`且正在执行FOURFAST时累计`UnknownSolveJob.advance()`包围单个生成器工作单元的实际耗时；拍照、显示和帧间等待不计入。

数据流如下：

```text
1～3片：GRAPH -> FALLBACK

4片：GRAPH
       -> FOURFAST（最多1.5秒活动计算）
       -> 成功：立即返回
       -> 子预算到期：设置abort标志并输出阶段诊断
       -> FALLBACK（继续使用统一8秒预算的剩余部分）
```

`UnknownSolveJob`负责计时并在共享`progress`中设置`four_fast_abort_requested`。FOURFAST生成器只在工作单元边界读取该标志，记录`four_active_limit_reached=1`和`four_active_elapsed_ms`后正常返回无解；流水线沿用既有阶段事件边界，在下一帧进入FALLBACK。不得关闭整个UnknownSolveJob，也不得返回`solver_timeout`。

## 边界与诊断

- FOURFAST在1.5秒内得到合法规划时不受影响。
- GRAPH和FALLBACK耗时不计入FOURFAST子预算。
- 三片从不设置或读取FOURFAST中止标志。
- 全局8秒活动预算、30秒墙钟、24ms/64单元保持不变。
- 日志增加`time_limit`和`active_ms`，用于区分工作量上限与活动时间上限。
- `UNKNOWN_FOUR_FAST_ACTIVE_BUDGET_SECONDS`必须是有限正数；现场可在1.0～2.0秒之间调整。

## 测试范围

1. 真实FOURFAST生成器在假时钟超过子预算后正常返回阶段失败，并于下一帧启动FALLBACK。
2. 子预算到期不能令整个任务返回`solver_timeout`。
3. 三片即使配置极小FOURFAST子预算，仍保持`GRAPH -> FALLBACK`并成功。
4. FOURFAST在子预算内成功时保留原规划。
5. 调试日志显示`time_limit=1 active_ms=...`。
6. A版专项、全量、编译、发布ZIP和B版固定哈希继续通过。

