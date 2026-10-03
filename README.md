Step 1 (Generation): I have prepared the comprehensive README content for your CrewAI project based on your presentation slide deck.

# Build Agents with CrewAI - Day 5 Tutorial

Welcome to the learning guide and tutorial for building intelligent, role-playing AI agents using **CrewAI**. This guide covers everything from core components to advanced custom LLM wrappers and web search integration.

## Table of Contents

* What is CrewAI?
* CrewAI Core Components
* Setup & LLM Configuration
* Defining Agents and Tasks
* Assembling & Running a Crew
* Agent Reasoning
* Tools & Custom LLMs

---

## What is CrewAI?

CrewAI is a Python framework designed for orchestrating role-playing AI agents that collaborate seamlessly on complex tasks.

* **Role-Based Agents:** Each agent features a distinct role, goal, and backstory, making collaboration contextual and natural.
* **Task-Driven Pipeline:** Tasks clearly define what needs to be done, expected outputs, and responsible agents.
* **Flexible LLM Backend:** Plug in any model via LiteLLM, including Gemini, GPT-4, Claude, Groq, and local models.
* **Tool Ecosystem:** Equip agents with built-in tools (such as Exa or Serper) or custom `BaseTool` subclasses.

---

## CrewAI Core Components

Four primary classes form the foundation of any CrewAI project:

| Component | Description |
| --- | --- |
| **LLM** | Wraps any LiteLLM-compatible model using a model string and API key. |
| **Agent** | The worker bee defined by a role, goal, backstory, LLM, tools, and reasoning flags. |
| **Task** | The unit of work detailing description, expected output, assigned agent, and markdown flags. |
| **Crew** | Binds agents and tasks together; `crew.kickoff()` initiates execution. |

---

## Setup & LLM Configuration

```python
# 1. Install dependencies
pip install crewai crewai-tools exa_py google-genai

# 2. Set API keys
import os
os.environ['GEMINI_API_KEY'] = "your_gemini_api_key"
os.environ['EXA_API_KEY'] = "your_exa_api_key"

# 3. Create LLM wrappers
from crewai import LLM

gemini_flash = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.environ["GEMINI_API_KEY"]
)

gemini_lite = LLM(
    model="gemini/gemini-2.5-flash-lite",
    api_key=os.environ["GEMINI_API_KEY"]
)

```

---

## Defining Agents and Tasks

```python
from crewai import Agent, Task

analyst = Agent(
    role="Research Specialist",
    goal="Conduct detailed research on {topic}",
    backstory="Expert at web research.",
    llm=gemini_lite,
    reasoning=True,
    max_reasoning_attempts=3,
    verbose=True,
)

analysis_task = Task(
    description="Research and analyze recent trends and generate a detailed report on {topic}.",
    expected_output="A detailed executive report for C-suite readers.",
    agent=analyst,
    verbose=True,
)

```

---

## Assembling & Running a Crew

```python
from crewai import Crew

crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    verbose=True,
)

result = crew.kickoff(inputs={
    "topic": "Influencer marketing",
    "todays_date": "July 31, 2025",
})

print(result.raw)
print(result.usage_metrics)

```

---

## Agent Reasoning

Enabling `reasoning=True` prompts the agent to reflect and plan before taking action.

* **Receive Task:** The agent intakes the objective.
* **Reflect & Plan:** Generates an internal plan outlining necessary information and steps (similar to built-in Chain-of-Thought prompting).
* **Execute & Return:** Executes the plan and returns the final result.

---

## Tools & Custom LLMs

Agents can leverage neural search tools like `EXASearchTool` for semantic web searches. Advanced implementations can also subclass `crewai.LLM` to inject specialized features like Gemini's Google Search grounding.

---

Would you like me to create this as a downloadable file or proceed with any specific uploads?
