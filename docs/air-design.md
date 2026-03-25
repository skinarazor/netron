# Netron 支持 Ascend AIR 文件设计说明

## 1. 背景与目标

本方案的目标是在当前 Netron 代码基础上，新增对昇腾 `AIR` 文件的支持，使 `.air` 文件能够像现有 `OM`、`ONNX`、`TFLite` 等格式一样，被 Netron 识别、解析并展示。

当前阶段目标是：

- 支持打开并展示当前已获得的旧版 `AIR` 样本
- 尽量复用现有 `OM` 的成熟实现路径
- 在属性展示上采用“不丢失优先”策略
- 以最小改动接入 Netron 现有架构
- 为后续适配新 `AIR` 样本保留扩展空间

当前验证样本包括：

- [resnet50_export.air](/home/zhangfan/github/netron/resnet50_export.air)
- [export.air](/home/zhangfan/github/netron/export.air)

## 2. 样本分析结论

基于当前样本，已经确认：

- 该文件不是 `OM` 容器
- 文件头不是 `IMOD` / `PICO`
- 不存在 `OM` 的容器头和 partition table 结构
- 可以直接按 `ge.proto.ModelDef` 解码
- 因此它更接近“直接序列化图定义”的格式，而不是 `OM` 的分区容器格式
- 不同来源的 `AIR` 在图归一化细节上存在差异
  - MindSpore 导出的样本使用 `graph.output[]`
  - Torch-AIR 导出的样本使用 `NetOutput` 节点表达图输出
  - Torch-AIR 的部分张量元信息主要位于 `Const.value.t.desc.attr`

由此得到实现判断：

- `AIR` 不能直接复用 [om.js](/home/zhangfan/github/netron/source/om.js) 的容器解析逻辑
- 但可以复用 `OM` 的 protobuf schema、图归一化方式、metadata 和大量节点/属性处理思路

## 3. 总体设计

### 3.1 设计原则

本次设计遵循以下原则：

- 接入 Netron 现有格式体系，不新建平行框架
- 尽量复用 `OM` 已验证的通用逻辑
- 与 Netron 现有展示行为保持一致
- 对未知属性不丢失，优先原始展示
- 先支持当前样本，后续再扩展变体兼容

### 3.2 总体接入方式

AIR 作为一个新的格式模块接入 Netron：

- 在 [view.js](/home/zhangfan/github/netron/source/view.js) 中注册 `./air`
- 在 [base.js](/home/zhangfan/github/netron/source/base.js) 中加入 `.air` 文件过滤支持
- 在 [electron-builder.json](/home/zhangfan/github/netron/publish/electron-builder.json) 中补充桌面文件关联
- 新增 [air.js](/home/zhangfan/github/netron/source/air.js) 作为 AIR 格式实现

接入后，`.air` 文件进入 Netron 的标准流程：

`ModelFactoryService -> match() -> open() -> Model -> Graph -> Node -> view.js 通用渲染`

## 4. 代码设计说明

### 4.1 文件识别设计

入口为 [air.js](/home/zhangfan/github/netron/source/air.js) 中的 `ModelFactory.match(context)`。

处理逻辑：

1. 读取文件前 4 字节
2. 若命中 `IMOD` / `PICO`，直接返回 `null`
3. 动态加载 [om-proto.js](/home/zhangfan/github/netron/source/om-proto.js)
4. 使用 `ge.proto.ModelDef.decode()` 尝试解码
5. 若解码成功且存在 `graph`，判定为 `AIR`

设计原因：

- 当前样本不是容器格式
- 直接 protobuf 解码是当前阶段最可靠的识别方式
- 先排除 `OM`，避免与现有 `OM` 支持冲突

### 4.2 打开与模型构建设计

`open(context)` 的职责是：

- 加载 [om-metadata.json](/home/zhangfan/github/netron/source/om-metadata.json)
- 构建 `air.Model`

`air.Model` 的职责：

- 设置模型格式名 `Ascend AIR`
- 读取模型版本
- 将 `graph[]` 转成 Netron `modules`

### 4.3 Graph 归一化设计

`air.Graph` 负责把原始 `graph.op` 归一化为 Netron 图结构。

当前处理规则：

- `Data`
  - 生成 graph input
  - 按 `OM` 风格处理，不保留为普通节点
- `Const`
  - 按 `OM` 方式处理
  - 提取为 initializer tensor
  - 不保留为普通节点
- `NetOutput`
  - 不作为普通节点显示
  - 当 `graph.output[]` 为空时，使用 `NetOutput.input[]` 作为 graph output
- 普通算子
  - 转成 `air.Node`
- `graph.output`
  - 转成 graph output

其中 `Const` 和 `Data` 的处理都已明确对齐 `OM`。原因是 Netron 的视图层对 `initializer` 有固定展示逻辑，如果同时保留 `Const` 节点，又让下游输入继续作为 initializer，就会在 UI 中形成“孤立的 Const 节点”；而如果同时保留 `Data` 的 graph input 和 `Data` 节点，则会在图中形成重复的输入展示。因此当前策略是：

- `Const` 只保留 initializer 语义
- `Data` 只保留 graph input 语义
- `NetOutput` 只保留 graph output 语义

### 4.4 Node 设计

`air.Node` 负责将单个算子转换成 Netron 标准节点对象，包含：

- `name`
- `type`
- `inputs`
- `outputs`
- `attributes`
- `controlDependencies`
- `chain`
- `device`

输入处理逻辑：

- 遍历 `op.input`
- 解析 `"node:idx"` 格式
- 若来源是常量张量，则通过 `tensors.get(identifier)` 将 initializer 挂到输入 value 上
- 若同名 value 被重复声明，则优先合并类型和 initializer 信息，而不是立即报错
- 若是控制依赖，则进入 `controlDependencies`

输出处理逻辑：

- 根据 `op.output_desc` 构造 `${nodeName}:${i}` 形式的 value

### 4.5 属性处理设计

属性展示采用“已知结构化、未知原始兜底”的策略。

已支持的属性类型包括：

- 标量：`i`、`f`、`b`、`s`、`dt`
- 张量/描述符：`t`、`td`
- 图和函数：`g`、`func`
- 列表：`list.*`
- 嵌套列表：`list_list_int`、`list_list_float`

关键设计点：

1. 不主动过滤内部属性  
   当前会保留如下属性：
   - `_input_name_key`
   - `_input_name_value`
   - `_output_name_key`
   - `_output_name_value`
   - `is_input_const`
   - `_opt_input`

2. 未知属性不丢失  
   对无法识别语义的属性，使用 `rawObject()` 递归保留原始结构，避免因缺少解析分支而直接丢值

3. 已知分支不改行为  
   对已经明确工作的类型分支保持原实现，仅增强未知情况兜底，不破坏现有已知展示效果

4. tensor descriptor 元信息可见  
   `Const.value.t.desc.attr` 会挂到 initializer tensor 的 `attributes` 上，因此这类信息会在 UI 的 `Tensor Properties` 面板中显示。

5. 格式枚举做可读化展示  
   对 `origin_format_for_int`、`format_for_int` 这类整数格式字段，优先展示可读格式名并保留原始整数，例如 `ND (2)`。

## 5. 复用 OM 的逻辑说明

AIR 当前不是复用 `OM` 的容器层，而是复用 `OM` 的中后段逻辑。

### 5.1 复用的逻辑

1. 复用 protobuf schema  
   复用 [om-proto.js](/home/zhangfan/github/netron/source/om-proto.js) 中的 `ge.proto.ModelDef` 定义。  
   这是 AIR 当前实现最核心的复用点。

2. 复用 Graph / Node 归一化思路  
   `air.js` 的 `Model / Graph / Node / Value / Tensor` 结构，整体参考 [om.js](/home/zhangfan/github/netron/source/om.js) 的组织方式，包括：
   - `values.map()` 的统一 value 管理
   - `tensors` 映射
   - 输入输出 argument/value 组织方式
   - 属性转 Netron 节点属性的模式

3. 复用 `Const -> initializer` 处理方式  
   `AIR` 当前已按 [om.js](/home/zhangfan/github/netron/source/om.js#L65) 的逻辑处理 `Const`：
   - 提取 tensor
   - 放入 `tensors`
   - 不保留 `Const` 节点
   - 下游输入使用 initializer

4. 复用 metadata  
   直接复用 [om-metadata.json](/home/zhangfan/github/netron/source/om-metadata.json) 提升算子展示质量。

5. 复用 `Data` 的 graph input 处理方向  
   `AIR` 当前已与 `OM` 一致，不再把 `Data` 作为普通节点保留，而是只保留为 graph input。

6. 复用类型映射设计  
   包括：
   - `dtype`
   - `TensorType`
   - `TensorShape`
   - descriptor 到 tensor type 的转换方式

### 5.2 没复用的逻辑

1. 不复用 `OM` 容器识别  
   `AIR` 不是 `IMOD/PICO`，不走 `om.Container.open()`。

2. 不复用 `OM` 分区读取  
   当前 AIR 样本不包含 `MODEL_DEF` / `WEIGHTS_DATA` 分区，不需要 `OM` 的 header + partition table 解析。

3. 不复用 `context.signature` 分支  
   AIR 当前没有 `IMOD/PICO` 这种签名分支语义。

## 6. 约束与限制

### 6.1 样本覆盖限制

当前实现基于两份样本设计和验证：  

- [resnet50_export.air](/home/zhangfan/github/netron/resnet50_export.air)
- [export.air](/home/zhangfan/github/netron/export.air)

因此当前可确认的是：

- 支持至少两类已验证的 `AIR` 导出形态
- 尚不能宣称覆盖所有 AIR 变体

### 6.2 格式结构限制

当前实现假设目标 AIR 可以直接按 `ge.proto.ModelDef` 解码。  
如果后续新样本不满足这一前提，则需要在 [air.js](/home/zhangfan/github/netron/source/air.js) 内增加新的识别和解析分支。

### 6.3 识别策略误判风险

当前识别逻辑本质上是：

- 排除 `IMOD/PICO`
- 尝试 protobuf decode
- 成功则认为是 AIR

这种方法在当前阶段可行，但存在理论上的误判风险：

- 某些其他二进制文件如果刚好也能被错误解码为 `ModelDef`
- 可能会被误识别为 AIR

后续如果样本增多，需要进一步收紧识别条件。

### 6.4 Const 展示限制

当前 AIR 已明确按 `OM` 的方式处理 `Const`：

- `Const` 仅作为 initializer 存在
- 不作为普通节点出现在图中

这不是遗漏，而是有意设计。原因在于 Netron 公共视图层对 initializer 采用“权重输入展示”而不是“普通图边展示”。

### 6.5 Data 处理已对齐 OM

当前 `Data` 的处理已经对齐 `OM`：

- 会生成 graph input
- 不会保留为普通节点

这样可以避免图中重复出现输入节点和孤立输入点，和 `OM` 的显示习惯保持一致。代价是 `Data` 自身属性不再作为普通节点属性单独显示。

### 6.6 属性展示不等于语义完全理解

当前属性策略是“尽量全部呈现”，不是“所有语义都已完全理解”。  
因此：

- 已知属性会结构化展示
- 未知属性会以原始结构兜底
- 但 raw 展示不代表我们已经完全理解这些属性的业务含义

### 6.7 少量属性仍有展示改写

当前仍存在少量非完全原样展示的属性：

- `device` 被提升为节点级字段
- `relu_flag=true` 会被转成链式 `ReLU`

这类信息原则上没有丢失，但展示形式并非完全原始。

### 6.8 公共视图层限制

AIR 的最终显示仍受 [view.js](/home/zhangfan/github/netron/source/view.js) 的公共渲染规则约束。尤其是：

- 带 `initializer` 的输入默认按权重处理
- 不绘制普通边

这意味着 AIR 模块本身不能单独决定所有视觉效果。若后续要改变 initializer 的展示方式，需要修改公共渲染层，而不仅是 `air.js`。

### 6.9 Metadata 覆盖限制

当前 AIR 复用 [om-metadata.json](/home/zhangfan/github/netron/source/om-metadata.json)。  
这对 GE/OM 体系共享算子有效，但如果 AIR 后续出现专有算子，则：

- 节点名可能仍然较原始
- 属性名可能不完整
- 需要继续补 metadata

### 6.10 tensor descriptor 展示限制

当前已经将 `Const.value.t.desc.attr` 暴露到 initializer tensor 的 `attributes` 中，但这些信息呈现在 `Tensor Properties` 面板，而不是普通节点属性面板。  
这意味着：

- 常量张量的格式和形状补充信息现在可见
- 但需要通过节点输入中的 tensor/weight 入口查看
- 不会直接出现在节点主属性列表中

### 6.11 权重来源限制

当前样本中的常量权重可在模型内部结构中直接提取。  
当前实现尚未覆盖例如：

- 外部权重文件
- 多文件组合模型
- 额外附件资源

如果未来出现这类 AIR 变体，需要补充读取逻辑。

### 6.12 测试限制

当前验证方式主要包括：

- 样本可打开
- 图可构建
- 节点和属性可展示
- `eslint` 通过

目前还没有多样本自动化回归测试集，因此验证覆盖度有限。

### 6.13 平台验证限制

当前主要验证的是 Linux 本地产物：

- [netron](/home/zhangfan/github/netron/dist/linux-unpacked/netron)

Windows 和 macOS 尚未单独做运行验证。逻辑层改动本身是跨平台的，但分发产物行为还没有逐个平台确认。

## 7. 当前涉及代码文件

本方案当前涉及以下文件：

- 新增 [air.js](/home/zhangfan/github/netron/source/air.js)
- 修改 [view.js](/home/zhangfan/github/netron/source/view.js)
- 修改 [base.js](/home/zhangfan/github/netron/source/base.js)
- 修改 [om-metadata.json](/home/zhangfan/github/netron/source/om-metadata.json)
- 修改 [electron-builder.json](/home/zhangfan/github/netron/publish/electron-builder.json)

职责分别为：

- `air.js`
  - AIR 文件识别、解码、图建模、属性处理
- `view.js`
  - 注册 AIR 格式入口
- `base.js`
  - 补充 `.air` 文件过滤
- `om-metadata.json`
  - 改善 AIR 节点和属性显示
- `electron-builder.json`
  - 补充桌面端文件关联

## 8. 验证方式

当前可直接使用本地构建产物验证：

- [netron](/home/zhangfan/github/netron/dist/linux-unpacked/netron)

运行方式：

```bash
./dist/linux-unpacked/netron resnet50_export.air
```

建议验证项：

- `Conv2D` 节点是否可见
- `MatMulV2`、`ReduceMeanD` 是否可见且命名合理
- 属性面板中内部属性是否完整展示
- 常量输入是否以 initializer/权重方式展示
- 图输入输出是否正确
- Torch-AIR 中 `NetOutput` 是否正确转成 graph output
- Torch-AIR 和 MindSpore AIR 中 tensor 属性面板是否能看到 `origin_format` / `origin_format_for_int` 等 descriptor 属性

## 9. 后续演进计划

后续拿到新的 `.air` 样本后，建议按以下路径扩展：

1. 比对文件头和基础特征
2. 判断是否仍然可用 `ge.proto.ModelDef` 解码
3. 如果可以：
   - 在当前 [air.js](/home/zhangfan/github/netron/source/air.js) 基础上继续增强
   - 补更多 metadata
4. 如果不可以：
   - 在 `air.js` 中增加新分支
   - 区分旧 AIR / 新 AIR
   - 不破坏当前样本支持
