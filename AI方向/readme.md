# 🌳 **AI 全方向知识树（总览）**

```
AI
├── 数学基础
│   ├── 线性代数
│   ├── 微积分
│   ├── 概率论与统计
│   ├── 离散数学
│   ├── 信息论
│   └── 最优化理论
│
├── 编程基础
│   ├── Python
│   ├── 数据结构与算法
│   ├── 软件工程能力
│   └── GPU / CUDA 基础
│
├── 机器学习（ML）
│   ├── 监督学习
│   ├── 无监督学习
│   ├── 半监督 / 自监督
│   ├── 特征工程
│   ├── 模型评估
│   └── 经典模型体系
│       ├── 线性模型
│       ├── 树模型（XGBoost，LightGBM）
│       ├── SVM
│       ├── 聚类（KMeans 等）
│       └── 降维（PCA / t-SNE）
│
├── 深度学习（DL）
│   ├── 神经网络基础
│   ├── CNN（图像）
│   ├── RNN / LSTM（序列）
│   ├── Transformer（核心）
│   ├── GNN（图神经网络）
│   ├── 强化学习（RL）
│   └── 多模态学习
│
├── 大模型（LLM）
│   ├── 模型架构（GPT, Llama, Mistral）
│   ├── 注意力机制（Self-Attention）
│   ├── 预训练（Pretrain）
│   ├── 微调（SFT）
│   ├── 奖励建模（RM）
│   ├── 强化学习对齐（RLHF / RLAIF）
│   ├── 功能调优（DPO, PPO, ORPO）
│   ├── 推理加速（Speculative, KV Cache）
│   └── 多模态大模型（Vision, Audio, Agents）
│
├── AI 代理（Agents）
│   ├── 工具调用（ToolUse）
│   ├── 规划（Planning）
│   ├── 长期记忆（Memory）
│   ├── RAG（检索增强生成）
│   ├── 工作流 AI（Workflow Agents）
│   └── 多 Agent 协作系统
│
├── AI 工程化（MLOps）
│   ├── 数据工程（ETL / DataOps）
│   ├── 模型训练流水线
│   ├── 模型部署（Docker / K8s / Triton）
│   ├── 模型监控
│   ├── 持续评估（CEval / Harness）
│   ├── A/B 测试
│   └── 模型安全性（越狱、注入、幻觉）
│
├── AI 系统（系统层面）
│   ├── 分布式训练（DDP / FSDP）
│   ├── 大规模参数存储（ZeRO）
│   ├── 模型并行（TP / PP）
│   ├── 推理框架（vLLM / TensorRT / ONNX）
│   ├── 数据并行与流水线训练
│   ├── Cache / KV 管理
│   └── 高性能计算（HPC）
│
└── AI 应用领域
    ├── NLP
    ├── CV（图像 / 视频）
    ├── 语音（ASR / TTS）
    ├── 多模态（VLM）
    ├── 游戏 / 强化学习
    ├── 推荐系统
    ├── 医疗 AI
    ├── 金融 AI
    └── 自动驾驶（感知 / 规划 / 控制）
```

---

# 🧩 **一、数学基础知识树**

### **1. 线性代数**

* 矩阵运算
* 特征值、特征向量
* 奇异值分解（SVD）
* 张量运算（大模型核心）

### **2. 微积分**

* 偏导数
* 梯度、链式法则
* 优化过程中的梯度下降

### **3. 概率论**

* 随机变量
* 分布（正态/泊松/伯努利）
* 期望、方差
* KL 散度、交叉熵（损失函数核心）

### **4. 最优化理论**

* 梯度下降
* Adam / Adagrad
* 凸优化
* 拉格朗日乘子法

---

# 🧠 **二、机器学习知识树**

### **1. 监督学习**

* 分类（logistic 回归，树模型）
* 回归（线性回归）

### **2. 无监督学习**

* 聚类（KMeans、DBSCAN）
* 降维（PCA、t-SNE）

### **3. 核心技能**

* 特征工程
* 模型验证（AUC/Recall）
* 交叉验证
* 数据清洗

---

# 🔥 **三、深度学习知识树**

### **1. 基础**

* 激活函数（ReLU / GELU）
* 损失函数（交叉熵）
* 反向传播

### **2. 主要模块**

* CNN（图像任务）
* RNN/LSTM（序列）
* Transformer（NLP / 多模态主流）

### **3. 高级方向**

* 注意力机制
* 图神经网络（GNN）
* 强化学习（RL）

---

# 🦾 **四、大模型（LLM）知识树**

### **1. 核心原理**

* Transformer
* Self Attention
* Masked Attention
* Rotary Embedding (RoPE)

### **2. 训练阶段**

* 预训练（大规模语料）
* 指令微调（SFT）
* 对齐（RLHF / RLAIF）

### **3. 推理优化**

* KV Cache
* TensorRT-LLM
* Speculative Decoding
* 量化（INT8/4/3/1）

### **4. 微调技术**

* LoRA / QLoRA
* Adapter
* Flash-Attention

### **5. 多模态方向**

* 文本 → 图像（Diffusion）
* 图像 → 文本（VLM）
* 文本 → 音频（TTS）
* 视频生成（Sora 类模型）

---

# 🤖 **五、AI Agent 知识树**

### **1. 核心模块**

* 规划（Planning）
* 工具调用（Tool Use）
* 任务分解（Decomposition）
* 环境感知（Observation）

### **2. 工作流 AI**

* 多 Agent 协作
* Agent 团队（HuggingGPT, AutoGen）
* 长期记忆（Memory）

### **3. RAG（检索增强）**

* 向量数据库
* 文档分块
* 检索 → 生成链条

---

# 🏗️ **六、AI 工程化（MLOps）知识树**

### **1. 数据工程**

* ETL、数据清洗
* 标注系统
* 数据质量监控

### **2. 模型训练工程**

* 分布式训练
* 断点续训
* 模型版本管理

### **3. 部署**

* REST / gRPC
* K8s + Istio
* TensorRT / vLLM / Triton
* OnnxRuntime

### **4. 监控**

* 数据漂移
* 模型漂移
* 安全性测试（越狱/注入）

---

# 🚗 **七、AI 应用方向**

### 如果你想深耕某一方向，下面是可选树：

* **视觉（CV）**：分类、检测、分割、跟踪、视频生成
* **NLP**：翻译、问答、检索、LLM
* **语音**：TTS、ASR、声音合成
* **推荐系统**：CTR 预估、Embedding
* **自动驾驶**：感知、规划、控制
* **游戏 AI**：RL、多智能体、世界模型
* **金融 AI**：风控、量化
* **医疗 AI**：辅助诊断、分割、病灶检测
