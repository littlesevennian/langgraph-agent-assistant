# LangGraph 多任务智能 Agent 助手系统

基于 LangGraph 构建的多任务智能 Agent 应用系统，通过状态管理、任务分类、条件路由、工具封装以及大模型调用，实现针对不同用户目标的智能任务处理。

系统支持用户根据需求选择不同任务方向，包括：

- 求职 / 简历优化建议
- Agent 项目复刻指导
- Agent 学习路线规划

通过 LangGraph 的 StateGraph 架构，将用户输入分析、任务路由、业务处理以及 LLM 生成过程进行模块化设计，实现一个可扩展的智能 Agent 工作流。

---

# 项目背景

传统的大语言模型应用通常采用：


用户输入
↓
LLM
↓
生成回答


这种方式存在以下问题：

- 无法根据不同任务进行流程控制
- 复杂任务难以拆分
- 业务逻辑和模型调用耦合严重
- 后续扩展新的能力成本较高

因此，本项目基于 LangGraph 构建 Agent 工作流，通过：

- State 状态管理
- 节点化任务处理
- 条件分支路由
- 工具函数封装

实现更加工程化、可扩展的智能 Agent 系统。

---

# 系统架构

整体执行流程：

                用户输入
                   |
                   ↓
          analyze_input 节点
                   |
      生成 task_type 和 analysis_result
                   |
                   ↓
            route_task 路由
                   |
    ┌──────────────┼──────────────┐
    ↓              ↓              ↓

resume_response project_response learning_response

    ↓              ↓              ↓

generate_resume generate_project generate_learning
_advice _advice _advice

    └──────────────┼──────────────┘
                   |
                   ↓

             call_llm()

                   |
                   ↓

          生成最终回答

                   |
                   ↓

                END

---

# 核心功能

## 1. 基于 StateGraph 的 Agent 工作流

使用 LangGraph 的 StateGraph 构建 Agent 执行流程。

通过 State 保存：

```python
class State(TypedDict):
    messages: Sequence[BaseMessage]
    user_input: str
    task_type: str
    analysis_result: str
    final_answer: str

其中：

字段	作用
user_input	保存用户原始输入
task_type	保存任务分类结果，用于路由
analysis_result	保存用户需求分析结果
final_answer	保存最终生成内容
messages	保存对话消息历史
2. 用户意图分析与任务路由

系统首先分析用户输入，根据关键词判断用户目标：

例如：

我想优化我的简历
        ↓
task_type = resume
我想复刻一个GitHub Agent项目
        ↓
task_type = project
我想学习Agent
        ↓
task_type = learning

随后通过条件边：

add_conditional_edges()

动态选择对应处理节点。

3. 工具函数模块化设计

将不同任务能力封装为独立工具：

generate_resume_advice()

generate_project_advice()

generate_learning_advice()

节点负责：

控制流程
读取 State
更新 State

工具负责：

构造 Prompt
调用具体能力

实现：

流程控制
    +
业务能力
    +
LLM调用

的职责分离。

4. LLM 调用封装

通过统一函数：

call_llm()

完成：

API Key 管理
模型初始化
参数配置
异常处理

模型采用：

Qwen3-Max

通过 DashScope OpenAI-compatible API 调用。

技术栈
开发语言
Python
Agent框架
LangGraph
LangChain Core
大模型
Qwen3-Max
DashScope OpenAI-compatible API
工程能力
State Management
Prompt Engineering
Workflow Orchestration
Environment Configuration
Exception Handling
项目结构
project_root/

├── main.py                 # 项目启动入口

├── langgraph.json           # LangGraph运行配置

├── src/
│   └── agent/
│       ├── graph.py         # Agent核心流程定义
│       └── __init__.py

├── .env.example             # 环境变量模板

├── pyproject.toml           # 项目依赖配置

└── README.md
环境配置
1. 创建虚拟环境

推荐使用 uv：

uv sync

或者：

pip install -r requirements.txt
2. 配置环境变量

复制：

.env.example

修改为：

.env

填写：

DASHSCOPE_API_KEY=your_api_key

DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

DASHSCOPE_MODEL=qwen3-max

DASHSCOPE_TEMPERATURE=0.7
运行项目

执行：

python main.py

输入示例：

我想学习Agent

系统执行：

用户输入

↓

任务分析

↓

Agent路由

↓

对应任务处理

↓

LLM生成回答
项目亮点
1. Agent工作流设计

通过 LangGraph 将复杂任务拆解为多个节点，实现：

可视化流程控制
状态传递
条件分支
2. 模块化架构

通过：

Node
+
Tool
+
LLM

分离设计，提高系统可维护性和扩展能力。

3. 可扩展任务体系

新增任务类型时，只需要：

增加新的 task_type
添加对应 response 节点
添加对应工具函数

无需修改整体 Agent 架构。

后续优化方向
接入真实 Tool Calling
增加多轮对话 Memory
支持更多任务类型
增加 Web UI 展示
增加日志系统和测试模块
项目总结

本项目实现了一个基于 LangGraph 的多任务智能 Agent 系统，通过状态管理、条件路由、工具封装以及大模型调用，实现从用户需求分析到任务执行的完整 Agent 工作流。

该项目重点实践了 Agent 应用开发中的核心技术：

Agent Workflow Design
State Management
Prompt Engineering
LLM Application Development