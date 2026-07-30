# MaixCAM2与STM32F4串口协议说明

> 协议版本：1  
> 对应视觉程序：`maixcam2_app_A_quad` v2.0.0  
> 本文只说明STM32F4如何接线、接收、解析、确认和去重，不提供或修改F4工程源码。

## 1. 接线与串口参数

| MaixCAM2 | 方向 | STM32F4 | 说明 |
|---|---|---|---|
| A21 / UART4_TX | Maix -> F4 | 所选USART/UART的RX | 发送心跳、A4和拼图结果 |
| A22 / UART4_RX | F4 -> Maix | 所选USART/UART的TX | 接收F4发回的ACK |
| GND | 双向参考 | GND | 必须共地 |

串口固定为`115200 bit/s、8数据位、1停止位、无校验、无流控`。两端均使用3.3V TTL电平，不能直接连接5V TTL或RS-232电平。F4具体使用哪个USART以及哪个TX/RX引脚由F4硬件决定，只要和A21/A22交叉连接即可。

## 2. 坐标与数值约定

- 所有坐标均以完整A4纸左上角为`(0,0)`，X向纸面右侧增加，Y向纸面下侧增加。
- 竖放A4范围为`X=0..210mm，Y=0..297mm`；横放为`X=0..297mm，Y=0..210mm`。
- 坐标和角度均使用`实际值 × 10`的定点整数。F4解析后除以10得到毫米或度。
- 拼图坐标与角度使用有符号`int16`；A4尺寸和A4四角使用无符号`uint16`。
- 旋转角从纸面上方观察：正数为顺时针，负数为逆时针，范围为`[-180.0°, 180.0°)`。
- 全部多字节整数均为小端序。F4不能直接把接收缓冲区强制转换为带填充的C结构体。

## 3. 通用帧

```text
AA 55 | VERSION | TYPE | FLAGS | SEQ_L SEQ_H | LEN_L LEN_H | PAYLOAD | CRC_L CRC_H
```

| 帧偏移 | 长度 | 字段 | 解析规则 |
|---:|---:|---|---|
| 0 | 2 | SOF | 固定`AA 55` |
| 2 | 1 | VERSION | 固定`01` |
| 3 | 1 | TYPE | 见消息类型表 |
| 4 | 1 | FLAGS | bit0=`ACK_REQUIRED`，bit1=`RETRY` |
| 5 | 2 | SEQ | `uint16`小端，0..65535后回绕 |
| 7 | 2 | LENGTH | `uint16`小端，允许0..64 |
| 9 | N | PAYLOAD | 由TYPE决定，N必须等于LENGTH |
| 9+N | 2 | CRC16 | `uint16`小端 |

总帧长为`11 + LENGTH`字节，最大75字节。CRC计算范围是偏移2开始的`VERSION`至`PAYLOAD`最后一字节，共`7 + LENGTH`字节；不包含`AA 55`和末尾CRC。

| TYPE | 名称 | 方向 | ACK要求 |
|---:|---|---|---|
| `0x01` | HEARTBEAT | Maix -> F4 | 是 |
| `0x10` | PAPER_FRAME | Maix -> F4 | 是 |
| `0x20` | PUZZLE_RESULT | Maix -> F4 | 是 |
| `0x80` | ACK | F4 -> Maix | 否 |

## 4. CRC16-CCITT-FALSE

参数固定为：`poly=0x1021`、`init=0xFFFF`、`refin=false`、`refout=false`、`xorout=0x0000`。字符串`123456789`的结果必须是`0x29B1`。

下面函数可放入F4工程的协议模块中；调用时传入帧偏移2的地址和`7 + LENGTH`长度。

```c
/* 计算CRC16-CCITT-FALSE；data指向VERSION，length覆盖到PAYLOAD末尾。 */
static uint16_t vision_crc16_ccitt_false(const uint8_t *data, uint16_t length)
{
    uint16_t crc = 0xFFFFU;
    uint16_t index;
    uint8_t bit;

    for (index = 0U; index < length; ++index) {
        crc ^= (uint16_t)data[index] << 8;
        for (bit = 0U; bit < 8U; ++bit) {
            if ((crc & 0x8000U) != 0U) {
                crc = (uint16_t)((crc << 1) ^ 0x1021U);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }
    return crc;
}

/* 从未对齐字节流读取小端uint16，避免直接转换结构体造成未对齐或填充问题。 */
static uint16_t vision_read_u16_le(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

/* 从未对齐字节流读取小端int16，保留负旋转角的补码。 */
static int16_t vision_read_i16_le(const uint8_t *data)
{
    return (int16_t)vision_read_u16_le(data);
}
```

## 5. HEARTBEAT载荷

固定6字节：

| 载荷偏移 | 长度 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 4 | `uint32` | `uptime_ms`，Maix运行毫秒数，自然回绕 |
| 4 | 1 | `uint8` | `app_state`：0待机，1处于CAL，2已按START |
| 5 | 1 | `uint8` | `last_error`，当前固定为0 |

Maix每500ms产生一个新心跳序号。F4只要帧版本、长度和CRC正确，就应立即ACK；Maix在1500ms内收到匹配ACK时显示`UART:OK`，否则显示`UART:OFFLINE`。

## 6. PAPER_FRAME载荷

固定22字节，在CAL普通页按`SEND A4`时发送。它表示蓝框对应的完整A4毫米平面，不是黄色区域，也不是相机像素坐标。

| 载荷偏移 | 长度 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 1 | `uint8` | `orientation`：0竖放，1横放 |
| 1 | 1 | `uint8` | `corner_count`：固定4 |
| 2 | 2 | `uint16` | `paper_width_x10` |
| 4 | 2 | `uint16` | `paper_height_x10` |
| 6 | 4 | `uint16 x2` | 左上TL `(x,y)` |
| 10 | 4 | `uint16 x2` | 右上TR `(x,y)` |
| 14 | 4 | `uint16 x2` | 右下BR `(x,y)` |
| 18 | 4 | `uint16 x2` | 左下BL `(x,y)` |

竖放四角固定为`(0,0)、(210,0)、(210,297)、(0,297)mm`；横放固定为`(0,0)、(297,0)、(297,210)、(0,210)mm`。F4收到后可更新当前纸面方向和软件限位，但不得把这些点当作相机像素或电机脉冲数。

## 7. PUZZLE_RESULT载荷

一帧包含当前成功规划的全部1～4片，不会逐片拆帧。载荷头固定4字节：

| 载荷偏移 | 长度 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 1 | `uint8` | `mode`：0 KNOWN，1 UNKNOWN |
| 1 | 1 | `uint8` | `orientation`：0竖放，1横放 |
| 2 | 1 | `uint8` | `piece_count`：1..4 |
| 3 | 1 | `uint8` | 保留，必须为0 |

其后每片固定11字节，第`i`片记录起点为`4 + i*11`，其中`i=0..piece_count-1`：

| 记录偏移 | 长度 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 1 | `uint8` | `piece_index`，必须从1连续递增 |
| 1 | 2 | `int16` | `source_x_x10`，移动前中心X |
| 3 | 2 | `int16` | `source_y_x10`，移动前中心Y |
| 5 | 2 | `int16` | `target_x_x10`，目标中心X |
| 7 | 2 | `int16` | `target_y_x10`，目标中心Y |
| 9 | 2 | `int16` | `rotation_deg_x10`，所需旋转增量 |

载荷总长必须严格等于`4 + piece_count*11`。F4应先把全部记录复制到“待执行方案”缓冲区并完成范围检查，整帧全部合法后再原子替换当前方案；不能解析一片就立即启动电机。

## 8. ACK与可靠重发

ACK载荷固定4字节：

| 载荷偏移 | 长度 | 类型 | 字段 |
|---:|---:|---|---|
| 0 | 1 | `uint8` | `acked_type`，原帧TYPE |
| 1 | 2 | `uint16` | `acked_sequence`，原帧SEQ，小端 |
| 3 | 1 | `uint8` | `status`：0接受，非0拒绝 |

F4发送ACK时使用`TYPE=0x80、FLAGS=0、LENGTH=4`。ACK帧自身的SEQ可以使用F4独立递增序号；Maix按ACK载荷中的`acked_type + acked_sequence`匹配原帧。

F4必须维护最近已接受的`(TYPE, SEQ)`去重表：

1. 第一次收到合法`PAPER_FRAME`或`PUZZLE_RESULT`：保存/执行一次，然后发送`status=0` ACK。
2. 再次收到相同`TYPE+SEQ`：只重发`status=0` ACK，不得再次执行电机动作。
3. 收到CRC、VERSION、LENGTH不合法的帧：丢弃且不ACK，因为无法证明引用字段可靠。
4. 收到CRC正确但载荷字段非法的帧：不执行，发送非0 ACK；Maix会保留同一逻辑消息继续重发，便于调试发现错误。
5. 新的PUZZLE_RESULT可替换旧待执行方案；F4执行前仍需自行检查电机软限位、急停和夹具状态。

Maix首次发送后，前三次使用250ms重发间隔，之后每1000ms重发。重发保持相同TYPE和SEQ，只把FLAGS增加`RETRY`位，因此整帧CRC会变化。

## 9. F4接收状态机

推荐UART中断或DMA循环接收写入环形缓冲，主循环解析，禁止在中断里等待整帧、计算电机轨迹或阻塞发送。

```text
while 环形缓冲有数据:
    1. 搜索连续 AA 55；帧头前噪声逐字节丢弃
    2. 少于9字节时等待下一批数据
    3. 读取VERSION、TYPE、FLAGS、SEQ、LENGTH
    4. VERSION != 1 或 LENGTH > 64：只丢弃当前AA，重新搜索
    5. 少于11 + LENGTH字节时等待下一批数据
    6. 计算CRC(偏移2, 长度7 + LENGTH)，与末尾小端CRC比较
    7. CRC错误：只丢弃当前AA，重新搜索，不能清空整个环形缓冲
    8. CRC正确：按TYPE和严格载荷长度解析
    9. 业务字段全部合法后执行去重判断、提交数据并发送ACK
   10. 从环形缓冲移除本帧11 + LENGTH字节，继续解析后续粘包
```

建议环形缓冲至少256字节。解析器必须能处理：一帧分多次到达、一次到达多帧、帧头前噪声、坏CRC帧后紧跟合法帧。不要使用“收到一个字节就清空超时缓存”的做法，否则高负载时容易丢断包。

F4业务处理框架可按以下顺序组织：

```c
/* 伪代码：frame已经完成帧头、版本、长度和CRC校验。 */
static void vision_handle_frame(const VisionFrame *frame)
{
    if (frame->type == VISION_TYPE_HEARTBEAT) {
        if (frame->length == 6U) {
            vision_send_ack(frame->type, frame->sequence, 0U);
        }
        return;
    }

    if (vision_is_duplicate(frame->type, frame->sequence)) {
        /* 重发帧只重新ACK，绝不能再次启动机械动作。 */
        vision_send_ack(frame->type, frame->sequence, 0U);
        return;
    }

    if (frame->type == VISION_TYPE_PAPER && vision_parse_paper(frame)) {
        vision_mark_accepted(frame->type, frame->sequence);
        vision_send_ack(frame->type, frame->sequence, 0U);
    } else if (frame->type == VISION_TYPE_RESULT && vision_parse_result(frame)) {
        vision_mark_accepted(frame->type, frame->sequence);
        vision_send_ack(frame->type, frame->sequence, 0U);
    } else {
        vision_send_ack(frame->type, frame->sequence, 1U);
    }
}
```

## 10. 完整十六进制示例

### 10.1 竖放PAPER_FRAME

SEQ=`0x0010`，FLAGS=`ACK_REQUIRED`，CRC已包含：

```text
AA 55 01 10 01 10 00 16 00
00 04 34 08 9A 0B
00 00 00 00 34 08 00 00 34 08 9A 0B 00 00 9A 0B
05 02
```

### 10.2 横放PAPER_FRAME

SEQ=`0x0011`：

```text
AA 55 01 10 01 11 00 16 00
01 04 9A 0B 34 08
00 00 00 00 9A 0B 00 00 9A 0B 34 08 00 00 34 08
C3 32
```

### 10.3 三片UNKNOWN结果

竖放、SEQ=`0x0020`，三片数据为：

| 片号 | 源中心mm | 目标中心mm | 旋转 |
|---:|---|---|---:|
| 1 | (40.6, 90.9) | (65.0, 205.0) | +30.0° |
| 2 | (86.3, 81.2) | (105.0, 205.0) | -45.0° |
| 3 | (150.8, 91.6) | (145.0, 205.0) | +90.0° |

完整帧：

```text
AA 55 01 20 01 20 00 25 00
01 00 03 00
01 96 01 8D 03 8A 02 02 08 2C 01
02 5F 03 2C 03 1A 04 02 08 3E FE
03 E4 05 94 03 AA 05 02 08 84 03
B1 0B
```

其中第二片旋转`-45.0° × 10 = -450`，int16小端补码为`3E FE`。

### 10.4 对三片结果的ACK

ACK自身SEQ=`0x0044`，载荷引用`TYPE=0x20、SEQ=0x0020、status=0`：

```text
AA 55 01 80 00 44 00 04 00 20 20 00 00 E6 A6
```

## 11. 联调步骤

1. 先只连接GND和交叉TX/RX，F4电机输出保持禁用。
2. 运行Maix程序，确认F4每约500ms收到HEARTBEAT；逻辑分析仪解码必须为115200 8N1。
3. F4发匹配ACK，Maix屏幕应在1500ms内从`UART:OFFLINE`变为`UART:OK`。
4. 暂停F4 ACK，确认Maix变为`UART:OFFLINE`但相机、触摸和识别继续运行。
5. 进入CAL并完成AUTO ROI，点击`SEND A4`，逐字节核对方向、22字节载荷和CRC。
6. 故意丢弃第一次A4 ACK，确认F4收到相同TYPE+SEQ且FLAGS带RETRY；F4只能保存一次。
7. 完成一次拼图规划，确认一帧内`piece_count`与全部11字节记录完整，源中心、目标中心和角度与屏幕一致。
8. 故意丢弃第一次结果ACK，确认重复帧不会使F4重复执行方案。
9. 最后才启用电机，先抬高夹具、低速、单片验证坐标和旋转正负，再运行全部碎片。

PC自动测试已覆盖编解码、CRC、断包、粘包、坏帧恢复、心跳、ACK和重发。UART4电气连接、F4中断/DMA实现、实际ACK时序和电机安全必须按以上步骤实机验证。
