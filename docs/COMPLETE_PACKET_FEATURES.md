# 完整的数据包特征列表

## 🔍 PCAP文件中可提取的所有特征

通过分析PCAP文件，我们可以提取以下特征：

---

## 📊 特征分类

### 1. IP层特征（10个）

| 特征名 | 类型 | 说明 | 重要性 |
|--------|------|------|--------|
| `src_ip_int` | int | 源IP地址（32位整数） | ⭐⭐⭐⭐⭐ |
| `dst_ip_int` | int | 目标IP地址（32位整数） | ⭐⭐⭐⭐⭐ |
| `src_ip_is_private` | int | 源IP是否为内网（0/1） | ⭐⭐⭐⭐ |
| `dst_ip_is_private` | int | 目标IP是否为内网（0/1） | ⭐⭐⭐⭐ |
| `ip_version` | int | IP版本（4/6） | ⭐⭐ |
| `ip_ihl` | int | IP头长度 | ⭐⭐ |
| `ip_tos` | int | 服务类型（Type of Service） | ⭐⭐⭐ |
| `ip_len` | int | IP包总长度 | ⭐⭐⭐⭐ |
| `ip_id` | int | IP标识符 | ⭐⭐ |
| `ip_flags` | int | IP标志（DF/MF） | ⭐⭐⭐ |
| `ip_frag` | int | 分片偏移 | ⭐⭐ |
| `ip_ttl` | int | 生存时间 | ⭐⭐⭐⭐ |

### 2. TCP层特征（12个）

| 特征名 | 类型 | 说明 | 重要性 |
|--------|------|------|--------|
| `tcp_sport` | int | TCP源端口 | ⭐⭐⭐⭐⭐ |
| `tcp_dport` | int | TCP目标端口 | ⭐⭐⭐⭐⭐ |
| `tcp_seq` | int | TCP序列号 | ⭐⭐⭐ |
| `tcp_ack` | int | TCP确认号 | ⭐⭐⭐ |
| `tcp_dataofs` | int | TCP数据偏移（头长度） | ⭐⭐ |
| `tcp_window` | int | TCP窗口大小 | ⭐⭐⭐⭐ |
| `tcp_flags_syn` | int | SYN标志 | ⭐⭐⭐⭐⭐ |
| `tcp_flags_ack` | int | ACK标志 | ⭐⭐⭐⭐⭐ |
| `tcp_flags_fin` | int | FIN标志 | ⭐⭐⭐⭐ |
| `tcp_flags_rst` | int | RST标志 | ⭐⭐⭐⭐ |
| `tcp_flags_psh` | int | PSH标志 | ⭐⭐⭐⭐ |
| `tcp_flags_urg` | int | URG标志 | ⭐⭐ |

### 3. UDP层特征（2个）

| 特征名 | 类型 | 说明 | 重要性 |
|--------|------|------|--------|
| `udp_sport` | int | UDP源端口 | ⭐⭐⭐⭐⭐ |
| `udp_dport` | int | UDP目标端口 | ⭐⭐⭐⭐⭐ |

### 4. ICMP层特征（2个）

| 特征名 | 类型 | 说明 | 重要性 |
|--------|------|------|--------|
| `icmp_type` | int | ICMP类型 | ⭐⭐⭐⭐ |
| `icmp_code` | int | ICMP代码 | ⭐⭐⭐ |

### 5. 通用特征（5个）

| 特征名 | 类型 | 说明 | 重要性 |
|--------|------|------|--------|
| `timestamp` | float | 包到达时间（相对时间） | ⭐⭐⭐⭐⭐ |
| `packet_length` | int | 整个包的长度 | ⭐⭐⭐⭐⭐ |
| `protocol` | int | 协议类型（1=TCP, 2=UDP, 3=ICMP） | ⭐⭐⭐⭐⭐ |
| `payload_size` | int | 应用层载荷大小 | ⭐⭐⭐⭐ |
| `header_size` | int | 头部大小 | ⭐⭐⭐ |

---

## 📋 最终特征列表

### 方案1: 核心特征（25个）⭐⭐⭐⭐⭐

**只包含最重要的特征，适合快速训练**

| 序号 | 特征名 | 说明 |
|------|--------|------|
| 1 | `timestamp` | 包到达时间 |
| 2 | `packet_length` | 包长度 |
| 3 | `protocol` | 协议类型 |
| 4 | `src_ip_int` | 源IP（整数） |
| 5 | `dst_ip_int` | 目标IP（整数） |
| 6 | `src_ip_is_private` | 源IP是否内网 |
| 7 | `dst_ip_is_private` | 目标IP是否内网 |
| 8 | `src_port` | 源端口 |
| 9 | `dst_port` | 目标端口 |
| 10 | `ip_ttl` | TTL |
| 11 | `ip_len` | IP包长度 |
| 12 | `ip_tos` | 服务类型 |
| 13 | `ip_flags` | IP标志 |
| 14 | `tcp_flags_syn` | SYN标志 |
| 15 | `tcp_flags_ack` | ACK标志 |
| 16 | `tcp_flags_fin` | FIN标志 |
| 17 | `tcp_flags_rst` | RST标志 |
| 18 | `tcp_flags_psh` | PSH标志 |
| 19 | `tcp_flags_urg` | URG标志 |
| 20 | `tcp_window` | TCP窗口 |
| 21 | `tcp_seq` | TCP序列号 |
| 22 | `tcp_ack` | TCP确认号 |
| 23 | `payload_size` | 载荷大小 |
| 24 | `icmp_type` | ICMP类型 |
| 25 | `icmp_code` | ICMP代码 |

### 方案2: 完整特征（31个）⭐⭐⭐⭐

**包含所有有用的特征，适合追求最高性能**

在方案1的基础上增加：

| 序号 | 特征名 | 说明 |
|------|--------|------|
| 26 | `ip_version` | IP版本 |
| 27 | `ip_ihl` | IP头长度 |
| 28 | `ip_id` | IP标识符 |
| 29 | `ip_frag` | 分片偏移 |
| 30 | `tcp_dataofs` | TCP头长度 |
| 31 | `header_size` | 总头部大小 |

---

## 🎯 推荐方案

### 推荐：方案1（25个核心特征）

**原因**:
1. ✅ 包含所有关键特征（IP地址、端口、标志等）
2. ✅ 特征数量适中（25个）
3. ✅ 训练速度快
4. ✅ 性能应该已经很好

**相比当前方案（14个特征）的改进**:
- ✅ 新增IP地址特征（4个）
- ✅ 新增IP层特征（3个：tos, flags, seq/ack）
- ✅ 新增TCP URG标志（1个）
- ✅ 新增ICMP特征（2个）
- ✅ 新增TCP序列号和确认号（2个）

**总计**: 14 → 25个特征（+11个）

---

## 💡 特征重要性分析

### 为什么这些特征重要？

#### 1. IP地址（src_ip_int, dst_ip_int）⭐⭐⭐⭐⭐

**用途**:
- 区分不同的流
- 检测扫描攻击（目标IP变化快）
- 检测DDoS攻击（源IP变化快）

#### 2. IP内网标志（src_ip_is_private, dst_ip_is_private）⭐⭐⭐⭐

**用途**:
- 攻击通常是外网→内网
- 正常流量通常是内网→外网或内网→内网

#### 3. IP服务类型（ip_tos）⭐⭐⭐

**用途**:
- 区分不同优先级的流量
- 某些攻击可能设置特殊的TOS值

#### 4. IP标志（ip_flags）⭐⭐⭐

**用途**:
- DF（Don't Fragment）标志
- MF（More Fragments）标志
- 某些攻击可能使用分片

#### 5. TCP序列号和确认号（tcp_seq, tcp_ack）⭐⭐⭐

**用途**:
- 检测TCP序列号异常
- 检测TCP劫持攻击
- 检测重传

**注意**: 序列号是32位整数，范围很大，需要归一化或取模

#### 6. TCP URG标志（tcp_flags_urg）⭐⭐

**用途**:
- 紧急数据标志
- 某些攻击可能滥用URG标志

#### 7. ICMP类型和代码（icmp_type, icmp_code）⭐⭐⭐⭐

**用途**:
- 检测ICMP Flood攻击
- 区分不同类型的ICMP消息（ping, unreachable, etc.）

---

## 🔧 特征编码方式

### IP地址编码

```python
def ip_to_int(ip_str):
    """将IP地址转换为32位整数"""
    try:
        parts = [int(x) for x in ip_str.split('.')]
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except:
        return 0

def is_private_ip(ip_str):
    """判断是否为内网IP"""
    try:
        parts = [int(x) for x in ip_str.split('.')]
        if parts[0] == 10:
            return 1
        if parts[0] == 172 and 16 <= parts[1] <= 31:
            return 1
        if parts[0] == 192 and parts[1] == 168:
            return 1
        return 0
    except:
        return 0
```

### IP标志编码

```python
def encode_ip_flags(flags):
    """编码IP标志"""
    # flags可能是: '', 'DF', 'MF', 'DF+MF'
    flags_str = str(flags)
    if 'DF' in flags_str and 'MF' in flags_str:
        return 3  # DF+MF
    elif 'DF' in flags_str:
        return 1  # DF only
    elif 'MF' in flags_str:
        return 2  # MF only
    else:
        return 0  # No flags
```

### TCP序列号编码

```python
def normalize_tcp_seq(seq):
    """归一化TCP序列号"""
    # 方式1: 取模（减小范围）
    return seq % 1000000
    
    # 方式2: 归一化到[0, 1]
    # return seq / 4294967295.0
```

---

## 📊 特征统计

### 当前方案 vs 改进方案

| 方案 | 特征数 | IP地址 | IP层 | TCP层 | UDP层 | ICMP层 |
|------|--------|--------|------|-------|-------|--------|
| **当前** | 14 | ❌ | 2 | 7 | 0 | 0 |
| **改进（核心）** | 25 | ✅ | 8 | 12 | 2 | 2 |
| **改进（完整）** | 31 | ✅ | 12 | 13 | 2 | 2 |

### 预期性能提升

| 方案 | 特征数 | 预期Macro F1 | 提升 |
|------|--------|-------------|------|
| 当前 | 14 | 87.96% | - |
| 改进（核心） | 25 | 91-94% | +3-6% |
| 改进（完整） | 31 | 92-95% | +4-7% |

---

## 🚀 实施建议

### 推荐：使用方案1（25个核心特征）

**原因**:
1. 包含所有关键特征
2. 特征数量适中
3. 训练速度快
4. 性能应该已经很好

### 如果性能仍不理想

**可以尝试方案2（31个完整特征）**

或者考虑：
- 添加序列级别的统计特征
- 使用流级别的聚合
- 增加模型容量

---

希望这个完整的特征列表对你有帮助！🎉

