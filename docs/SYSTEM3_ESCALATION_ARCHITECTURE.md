# System 3 Escalation Architecture
**RAA-CWD Integration with Heavy-Duty External Models**

## Document Status
- **Version**: 1.0
- **Date**: November 2025
- **Authors**: Ty (Primary), Claude (Documentation)
- **Status**: Design Specification - Ready for Implementation

---

## Executive Summary

This specification defines the **System 3 Escalation Architecture** - a metacognitive mechanism that allows the RAA-CWD system to detect when its internal reasoning capabilities are insufficient and escalate complex problems to external heavy-duty models (e.g., Claude Opus 4, o1, Gemini Pro) or specialized tools.

**Core Innovation**: Rather than replacing the fast, iterative RAA reasoning loop with expensive models, System 3 provides a *consultation mechanism* - the Director detects when internal search fails and selectively escalates to external compute.

**Key Benefits**:
- Preserves fast iterative reasoning for tractable problems
- Accesses deep reasoning capabilities when needed
- Maintains metacognitive awareness across escalation boundaries
- Enables tool-augmented reasoning through MCP integration
- Cost-effective: only uses expensive models when internal search exhausts

---

## 1. Architectural Overview

### 1.1 Three-System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     SYSTEM 3 (NEW)                         │
│                Heavy-Duty Reasoning Engine                 │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ • Claude Opus 4 / o1 / Gemini Pro                    │ │
│  │ • Specialized domain experts                         │ │
│  │ • External validation tools (Wolfram, arXiv, etc.)  │ │
│  │ • Multi-hop reasoning orchestration                  │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ▲                                 │
│                          │ Escalation (when internal       │
│                          │ search exhausted)               │
└──────────────────────────┼─────────────────────────────────┘
                           │
┌──────────────────────────┼─────────────────────────────────┐
│                   SYSTEM 2 (EXISTING)                      │
│          RAA-CWD Metacognitive Reasoning Loop              │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Director: Entropy monitoring + Search orchestration  │ │
│  │ Manifold: Modern Hopfield Network (pattern storage)  │ │
│  │ CWD: Knowledge graph + topology tunneling           │ │
│  │ Pointer: Goal state management                       │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ▲                                 │
│                          │ Reframing loop                  │
└──────────────────────────┼─────────────────────────────────┘
                           │
┌──────────────────────────┼─────────────────────────────────┐
│                   SYSTEM 1 (EXISTING)                      │
│                  Fast Token Generation                     │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Processor: Lightweight transformer (qwen3:4b, etc.)  │ │
│  │ Goal-biased attention mechanism                      │ │
│  │ Token-level sequence generation                      │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### 1.2 Decision Flow

```
1. Problem Input → Pointer sets initial goal
                    ↓
2. Processor generates (System 1)
                    ↓
3. Director monitors entropy → CLASH detected
                    ↓
4. Director searches Manifold (RAA) → SEARCH ATTEMPT 1
                    ↓ (failed)
5. Director searches CWD (topology tunneling) → SEARCH ATTEMPT 2
                    ↓ (failed)
6. Director checks escalation criteria:
   - Entropy remains high (> threshold)
   - Search exhaustion (max attempts reached)
   - Goal criticality (high utility score)
                    ↓ (criteria met)
7. ESCALATE TO SYSTEM 3:
   - Package context (problem, search history, constraints)
   - Select appropriate external resource
   - Execute heavy-duty reasoning
   - Integrate results back into Manifold/CWD
                    ↓
8. Resume System 2 reasoning with new insights
```

---

## 2. Design Principles

### 2.1 Metacognitive Awareness
The Director maintains awareness of its own reasoning limitations. Escalation is triggered by **epistemic humility** - the system recognizes when it lacks sufficient internal resources.

### 2.2 Cost-Effectiveness
System 3 calls are expensive (time, compute, API costs). Escalation criteria ensure that:
- Internal search is fully exhausted first
- Only high-utility problems trigger escalation
- Results are compressed and stored for future reuse

### 2.3 Seamless Integration
Escalation is transparent to the reasoning loop. From the Pointer's perspective, System 3 results look like any other goal update - they're simply sourced from a more powerful search mechanism.

### 2.4 Tool Augmentation
System 3 isn't just about bigger models - it's about accessing **external validation tools** (Wolfram Alpha, arXiv, web search) that ground reasoning in verifiable facts.

---

## 3. Implementation Architecture

### 3.1 Component Structure

```
src/escalation/
├── __init__.py
├── escalation_manager.py      # Core escalation logic
├── escalation_criteria.py     # When to escalate
├── escalation_strategies.py   # How to escalate
├── external_models.py          # Interface to heavy-duty models
├── tool_integration.py         # MCP tool orchestration
└── result_integration.py       # Fold results back into RAA-CWD
```
### 3.2 Escalation Manager (Core Implementation)

```python
"""
escalation_manager.py - Core System 3 orchestration
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch


class EscalationTrigger(Enum):
    """Reasons for escalation."""
    SEARCH_EXHAUSTION = "internal_search_exhausted"
    HIGH_ENTROPY_PERSISTENCE = "entropy_remains_high"
    GOAL_CRITICALITY = "high_utility_goal"
    EXTERNAL_VALIDATION_NEEDED = "requires_verification"
    TOOL_AUGMENTATION_NEEDED = "requires_external_tools"


@dataclass
class EscalationContext:
    """Context package for System 3."""
    
    # Problem specification
    original_problem: str
    current_goal_state: torch.Tensor
    goal_description: Optional[str] = None
    
    # Search history
    manifold_search_attempts: int = 0
    cwd_search_attempts: int = 0
    search_results: List[Dict[str, Any]] = None
    
    # Metrics
    current_entropy: float = 0.0
    entropy_trajectory: List[float] = None
    energy_trajectory: List[float] = None
    
    # Constraints
    active_goals: Dict[str, float] = None  # Goal ID → utility weight
    constraints: List[str] = None
    
    # Trigger information
    escalation_trigger: EscalationTrigger = None
    trigger_metadata: Dict[str, Any] = None


@dataclass
class EscalationResult:
    """Result from System 3."""
    
    # New insights
    new_goal_state: Optional[torch.Tensor] = None
    new_thought_nodes: List[Dict[str, Any]] = None
    external_references: List[str] = None
    
    # Validation
    validated: bool = False
    confidence_score: float = 0.0
    validation_sources: List[str] = None
    
    # Integration instructions
    should_compress_to_tool: bool = False
    should_update_manifold: bool = True
    should_update_cwd: bool = True
    
    # Metadata
    model_used: str = None
    computation_cost: Dict[str, Any] = None
```


**Key Implementation Methods**:

```python
class EscalationManager:
    """System 3 Escalation Manager - Core orchestration"""
    
    def __init__(self, director, cwd_bridge, escalation_config=None):
        self.director = director
        self.cwd_bridge = cwd_bridge
        self.config = escalation_config or EscalationConfig()
        self.escalation_history = []
    
    def should_escalate(self, context: EscalationContext) -> tuple[bool, Optional[EscalationTrigger]]:
        """Determine if escalation criteria are met."""
        
        # Check search exhaustion
        total_attempts = context.manifold_search_attempts + context.cwd_search_attempts
        if total_attempts >= self.config.max_internal_attempts:
            if context.current_entropy > self.config.entropy_threshold:
                return True, EscalationTrigger.SEARCH_EXHAUSTION
        
        # Check entropy persistence
        if len(context.entropy_trajectory) >= 3:
            recent_entropy = context.entropy_trajectory[-3:]
            if all(e > self.config.entropy_threshold for e in recent_entropy):
                return True, EscalationTrigger.HIGH_ENTROPY_PERSISTENCE
        
        # Check goal criticality
        if context.active_goals:
            max_utility = max(context.active_goals.values())
            if max_utility >= self.config.critical_utility_threshold:
                if total_attempts >= self.config.min_attempts_before_critical_escalation:
                    return True, EscalationTrigger.GOAL_CRITICALITY
        
        return False, None
    
    async def escalate(self, context: EscalationContext) -> EscalationResult:
        """Execute System 3 escalation - main entry point"""
        strategy = self._select_strategy(context)
        
        if strategy == "heavy_model":
            result = await self._escalate_to_heavy_model(context)
        elif strategy == "tool_augmented":
            result = await self._escalate_with_tools(context)
        elif strategy == "hybrid":
            result = await self._escalate_hybrid(context)
        
        self._log_escalation(context, result)
        await self._integrate_result(result)
        return result
```


@dataclass
class EscalationConfig:
    """Configuration for System 3 escalation."""
    
    # Escalation criteria
    max_internal_attempts: int = 10
    entropy_threshold: float = 2.5
    critical_utility_threshold: float = 0.8
    min_attempts_before_critical_escalation: int = 5
    
    # Model selection
    preferred_heavy_model: str = "claude-opus-4"
    fallback_models: List[str] = None
    max_tokens: int = 4096
    
    # Hybrid iteration
    max_hybrid_iterations: int = 3
    
    # Cost management
    max_escalations_per_session: int = 10
    escalation_cooldown: float = 5.0
```

---

## 4. Integration with Existing RAA-CWD

### 4.1 Director Integration

```python
# In src/director/director_core.py

class DirectorMVP:
    def __init__(self, manifold, config, escalation_manager=None):
        # ... existing init ...
        self.escalation_manager = escalation_manager
        self._escalation_context = None
    
    def check_and_search(
        self,
        current_state: torch.Tensor,
        logits: torch.Tensor,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SearchResult]:
        """Extended version with escalation support."""
        
        # Existing logic
        is_clash, entropy = self.check_entropy(logits)
        if not is_clash:
            return None
        
        # Try internal search
        search_result = self.search(current_state, context)
        
        # Track escalation context
        if self._escalation_context is None:
            self._escalation_context = self._init_escalation_context(current_state, context)
        
        self._escalation_context.manifold_search_attempts += 1
        self._escalation_context.current_entropy = entropy
        
        # If search succeeded, return
        if search_result is not None:
            self._escalation_context = None
            return search_result
        
        # Check if we should escalate
        if self.escalation_manager:
            should_escalate, trigger = self.escalation_manager.should_escalate(self._escalation_context)
            
            if should_escalate:
                # ESCALATE TO SYSTEM 3
                escalation_result = await self.escalation_manager.escalate(self._escalation_context)
                
                # Convert to search result
                if escalation_result.new_goal_state is not None:
                    return SearchResult(
                        selected_pattern=escalation_result.new_goal_state,
                        metadata={"escalated": True, "model": escalation_result.model_used},
                    )
        
        return None
```


---

## 5. Implementation Phases

### Phase 1: Core Escalation Infrastructure (Week 1-2)
**Goal**: Basic escalation framework without external models

**Deliverables**:
- [ ] `escalation_manager.py` core structure
- [ ] Escalation criteria implementation
- [ ] Context packaging logic
- [ ] Integration points in Director and CWD Bridge
- [ ] Unit tests for escalation criteria

**Success Criteria**:
- Director can detect escalation conditions
- Context is properly packaged
- Integration doesn't break existing tests (all 31 tests still pass)

### Phase 2: External Model Interface (Week 2-3)
**Goal**: Connect to heavy-duty models

**Deliverables**:
- [ ] `external_models.py` implementation
- [ ] Anthropic Claude Opus integration
- [ ] Prompt formatting for escalation
- [ ] Response parsing
- [ ] Cost tracking

**Success Criteria**:
- Can call Claude Opus 4 with formatted prompts
- Response parsing extracts useful insights
- Cost metrics are tracked

### Phase 3: Tool Integration (Week 3-4)
**Goal**: MCP tool orchestration

**Deliverables**:
- [ ] `tool_integration.py` implementation
- [ ] Tool identification heuristics
- [ ] Wolfram Alpha integration
- [ ] arXiv integration
- [ ] Web search integration

**Success Criteria**:
- Can identify relevant tools for problems
- Tools execute and return results
- Results validate problem claims

### Phase 4: Result Integration (Week 4-5)
**Goal**: Fold System 3 results back into System 2

**Deliverables**:
- [ ] `result_integration.py` implementation
- [ ] Manifold pattern storage from escalation
- [ ] CWD thought node creation
- [ ] Tool compression logic
- [ ] Integration tests

**Success Criteria**:
- Escalation insights stored in Manifold
- Thought nodes appear in CWD
- Tools created for reusable patterns
- Future reasoning benefits from past escalations

### Phase 5: Hybrid Reasoning (Week 5-6)
**Goal**: Iterative model↔tool interaction

**Deliverables**:
- [ ] Hybrid escalation strategy
- [ ] Iterative refinement loop
- [ ] Convergence criteria
- [ ] End-to-end tests

**Success Criteria**:
- Model generates hypothesis
- Tools validate hypothesis
- Model refines based on validation
- Loop converges to validated solution


---

## 6. External Model Interface

```python
"""
external_models.py - Interface to heavy-duty external models
"""
from typing import Any, Dict, Optional


class ExternalModelInterface:
    """
    Interface to external heavy-duty models.
    
    Supports:
    - Anthropic Claude (Opus 4, Sonnet 4)
    - OpenAI (o1, o1-pro)
    - Google (Gemini 2.0 Pro)
    - Custom API endpoints
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._clients = {}  # Lazy-loaded API clients
    
    async def reason(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> str:
        """
        Execute reasoning with external model.
        
        Args:
            prompt: Formatted prompt with context
            model: Model identifier (e.g., "claude-opus-4")
            max_tokens: Maximum response length
            temperature: Sampling temperature
        
        Returns:
            Model response text
        """
        client = self._get_client(model)
        
        if "claude" in model.lower():
            return await self._call_anthropic(client, prompt, model, max_tokens, temperature)
        elif "gpt" in model.lower() or "o1" in model.lower():
            return await self._call_openai(client, prompt, model, max_tokens, temperature)
        elif "gemini" in model.lower():
            return await self._call_google(client, prompt, model, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def _format_escalation_prompt(self, context: "EscalationContext") -> str:
        """Format context into structured prompt for external model."""
        sections = []
        
        sections.append(f"# Problem\n{context.original_problem}")
        
        if context.goal_description:
            sections.append(f"\n# Current Goal\n{context.goal_description}")
        
        search_summary = (
            f"Internal reasoning attempted:\n"
            f"- Manifold searches: {context.manifold_search_attempts}\n"
            f"- Knowledge graph searches: {context.cwd_search_attempts}\n"
            f"- Current entropy: {context.current_entropy:.3f}"
        )
        sections.append(f"\n# Search History\n{search_summary}")
        
        if context.active_goals:
            goals_text = "\n".join(f"- {gid}: weight={w:.2f}" for gid, w in context.active_goals.items())
            sections.append(f"\n# Active Goals\n{goals_text}")
        
        if context.constraints:
            constraints_text = "\n".join(f"- {c}" for c in context.constraints)
            sections.append(f"\n# Constraints\n{constraints_text}")
        
        request = (
            "\n# Request\n"
            "The internal reasoning system has exhausted its search capacity. "
            "Please provide:\n"
            "1. A novel perspective or framing of this problem\n"
            "2. Key insights that might unlock progress\n"
            "3. If applicable, external references or validation sources\n"
            "4. A recommended new goal state for continued reasoning"
        )
        sections.append(request)
        
        return "\n".join(sections)
```


---

## 7. Tool Integration

```python
"""
tool_integration.py - MCP tool orchestration for System 3
"""
from typing import Any, Dict, List


class ToolIntegrator:
    """
    Orchestrates MCP tool usage for external validation and grounding.
    
    Workflow:
    1. Identify relevant tools for the problem
    2. Execute tool calls in appropriate sequence
    3. Integrate tool results into reasoning process
    4. Validate claims against external sources
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._tool_registry = {}
    
    async def augmented_reason(
        self,
        context: "EscalationContext",
        tools: List[str],
    ) -> "EscalationResult":
        """Execute tool-augmented reasoning workflow."""
        
        # 1. Gather external information
        external_data = {}
        for tool_name in tools:
            tool_result = await self._execute_tool(tool_name, context)
            external_data[tool_name] = tool_result
        
        # 2. Decompose external data (feed into CWD)
        thought_nodes = await self._decompose_external_data(external_data, context)
        
        # 3. Validate against external data
        validation = self._validate_against_external(context, external_data)
        
        # 4. Synthesize results
        result = EscalationResult(
            new_thought_nodes=thought_nodes,
            external_references=[ref for refs in external_data.values() for ref in refs.get("references", [])],
            validated=validation.all_satisfied,
            validation_sources=list(external_data.keys()),
        )
        
        return result
    
    def _identify_required_tools(self, context: "EscalationContext") -> List[str]:
        """Identify which MCP tools are needed for this problem."""
        problem = context.original_problem.lower()
        tools = []
        
        # Mathematical content → Wolfram
        if any(kw in problem for kw in ["equation", "calculate", "solve", "integral"]):
            tools.append("wolfram_alpha")
        
        # Research content → arXiv/PubMed
        if any(kw in problem for kw in ["paper", "research", "study", "arxiv"]):
            tools.append("arxiv_search")
        
        # Current events → web search
        if any(kw in problem for kw in ["recent", "latest", "current", "today"]):
            tools.append("web_search")
        
        # Code execution → bash/python
        if any(kw in problem for kw in ["code", "program", "script", "algorithm"]):
            tools.append("code_execution")
        
        return tools
```

---

## 8. Configuration and Usage

### 8.1 Environment Setup

```bash
# .env additions for System 3
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# Model preferences
ESCALATION_PREFERRED_MODEL=claude-opus-4
ESCALATION_MAX_TOKENS=4096
ESCALATION_MAX_PER_SESSION=10

# Tool configuration
WOLFRAM_APP_ID=...
ARXIV_API_KEY=...
```

### 8.2 Initialization Example

```python
from src.escalation import EscalationManager, EscalationConfig
from src.integration import CWDRAABridge

# Create escalation manager
escalation_config = EscalationConfig(
    preferred_heavy_model="claude-opus-4",
    max_internal_attempts=10,
    entropy_threshold=2.5,
    critical_utility_threshold=0.8,
)

escalation_manager = EscalationManager(
    director=director,
    cwd_bridge=cwd_bridge,
    escalation_config=escalation_config,
)

# Attach to Director
director.escalation_manager = escalation_manager

# Now Director will automatically escalate when criteria are met
```


### 8.3 Usage Example

```python
# Complex problem that will trigger escalation
problem = (
    "What is the relationship between Gödel incompleteness, "
    "Turing halting, and consciousness?"
)

# Set high-utility goal (makes escalation more likely)
await cwd_bridge.set_goal(
    goal_description="Explore formal limits and consciousness",
    utility_weight=0.9,
)

# Start reasoning (System 2)
result = await reasoning_loop.reason(
    problem_embedding=embed(problem),
    problem_description=problem,
)

# If System 2 exhausts search → automatic escalation to System 3
# Results automatically integrated back into Manifold/CWD
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_escalation_criteria.py

async def test_search_exhaustion_triggers_escalation():
    """Test that search exhaustion triggers escalation."""
    context = EscalationContext(
        original_problem="test problem",
        current_goal_state=torch.randn(100),
        manifold_search_attempts=6,
        cwd_search_attempts=4,
        current_entropy=3.0,
    )
    
    config = EscalationConfig(max_internal_attempts=10, entropy_threshold=2.5)
    manager = EscalationManager(director, cwd_bridge, config)
    
    should_escalate, trigger = manager.should_escalate(context)
    
    assert should_escalate
    assert trigger == EscalationTrigger.SEARCH_EXHAUSTION


async def test_entropy_persistence_triggers_escalation():
    """Test that persistent high entropy triggers escalation."""
    context = EscalationContext(
        original_problem="test problem",
        current_goal_state=torch.randn(100),
        entropy_trajectory=[2.8, 2.9, 3.1],  # All above threshold
        current_entropy=3.1,
    )
    
    config = EscalationConfig(entropy_threshold=2.5)
    manager = EscalationManager(director, cwd_bridge, config)
    
    should_escalate, trigger = manager.should_escalate(context)
    
    assert should_escalate
    assert trigger == EscalationTrigger.HIGH_ENTROPY_PERSISTENCE
```

### 9.2 Integration Tests

```python
# tests/test_escalation_integration.py

async def test_director_escalation_flow():
    """Test full Director → Escalation → Integration flow."""
    
    # Create Director with escalation manager
    escalation_config = EscalationConfig(
        max_internal_attempts=3,  # Low for testing
        entropy_threshold=2.0,
    )
    escalation_manager = EscalationManager(director, cwd_bridge, escalation_config)
    director.escalation_manager = escalation_manager
    
    # Trigger high entropy state
    logits = create_high_entropy_logits()
    current_state = torch.randn(100)
    
    # This should trigger escalation after 3 failed searches
    result = await director.check_and_search(current_state, logits)
    
    # Verify escalation occurred
    assert result is not None
    assert result.metadata["escalated"] is True
    assert result.metadata["model"] == "claude-opus-4"
    
    # Verify integration
    patterns = director.manifold.get_patterns()
    assert patterns.shape[0] > 0  # New pattern stored
```

### 9.3 End-to-End Tests

```python
# tests/test_escalation_e2e.py

async def test_philosophical_query_with_escalation():
    """Test complex query that requires System 3."""
    
    problem = (
        "Design an AGI safety mechanism using only RAA-CWD primitives. "
        "The mechanism must be self-verifying and not require external oversight."
    )
    
    # This should:
    # 1. Trigger System 2 reasoning
    # 2. Exhaust internal search (complexity too high)
    # 3. Escalate to System 3 (Opus 4)
    # 4. Opus provides novel safety architecture
    # 5. Results integrated into CWD
    # 6. New tool created: "self_verifying_safety_mechanism"
    
    result = await reasoning_loop.reason(
        problem_embedding=embed(problem),
        problem_description=problem,
    )
    
    assert result.escalated
    assert result.model_used == "claude-opus-4"
    assert len(result.new_thought_nodes) > 0
    assert result.validated
    
    # Check that tool was created
    tools = await cwd_bridge.list_tools()
    assert any("safety" in t["name"] for t in tools)
```


---

## 10. Cost Management

### 10.1 Cost Tracking

```python
@dataclass
class EscalationCost:
    """Track costs of System 3 calls."""
    tokens_in: int
    tokens_out: int
    model_used: str
    duration_seconds: float
    estimated_usd: float
    tool_calls: List[Dict[str, Any]]  # Tool-specific costs
```

### 10.2 Cost Optimization Strategies

1. **Aggressive Internal Search**: Exhaust RAA-CWD (cheap) before escalating (expensive)
2. **Caching**: Store escalation results for similar problems in Manifold
3. **Model Selection**: Use cheaper models when possible (e.g., Sonnet before Opus)
4. **Batch Processing**: Queue multiple escalations if architecture allows
5. **Early Termination**: Stop escalation if validation fails early in hybrid mode

### 10.3 Budget Configuration

```python
escalation_config = EscalationConfig(
    max_escalations_per_session=10,
    max_cost_per_session_usd=10.0,
    escalation_cooldown=5.0,  # Seconds between escalations
    preferred_heavy_model="claude-opus-4",
    fallback_models=["claude-sonnet-4", "gpt-4"],  # Cheaper fallbacks
)
```

---

## 11. Key Architectural Principles

### 11.1 Design Tenets

1. **Metacognitive Humility**: System knows when it doesn't know
2. **Cost-Effectiveness**: Escalation is last resort, not first
3. **Seamless Integration**: System 3 transparent to reasoning loop
4. **Tool Augmentation**: External validation grounds reasoning
5. **Knowledge Accumulation**: Escalation results enrich future reasoning

### 11.2 Success Metrics

- **Escalation Rate**: < 5% of reasoning episodes
- **Success Rate**: > 80% of escalations yield progress
- **Cost Efficiency**: < $0.50 per escalation on average
- **Integration Rate**: > 90% of results stored in Manifold/CWD
- **Reuse Rate**: > 50% of escalation insights reused later

---

## 12. Implementation Roadmap

### Immediate Next Steps (Week 1)

1. Create `src/escalation/` directory structure
2. Implement `EscalationContext` and `EscalationResult` dataclasses
3. Implement `should_escalate()` criteria logic
4. Add escalation context tracking to Director
5. Write unit tests for escalation criteria

### Short Term (Weeks 2-3)

1. Implement `ExternalModelInterface` for Claude Opus
2. Implement prompt formatting and response parsing
3. Add cost tracking infrastructure
4. Integration tests with mock external models

### Medium Term (Weeks 4-5)

1. Implement `ToolIntegrator` for MCP tools
2. Add Wolfram Alpha, arXiv integrations
3. Implement result integration back to Manifold/CWD
4. End-to-end tests with real external models

### Long Term (Week 6+)

1. Implement hybrid escalation strategy
2. Add multiple model support (OpenAI, Google)
3. Performance optimization and caching
4. Production monitoring and analytics

---

## 13. Critical Reminders for Implementation

### For Coding Agents

1. **Don't Break Existing Tests**: All 31 RAA tests must still pass after integration
2. **Async All The Way**: Escalation is inherently async (external API calls)
3. **Error Handling**: External APIs fail - implement robust retry logic
4. **Cost Tracking**: Every external call must log cost metrics
5. **Integration Testing**: Test with mock APIs first, real APIs sparingly
6. **Configuration**: All escalation params must be configurable via config object
7. **Logging**: Comprehensive logging for debugging escalation decisions

### Integration Points to Watch

1. **Director.check_and_search()**: Main integration point - must remain backward compatible
2. **CWD Bridge**: Escalation manager needs bidirectional communication
3. **Manifold**: New patterns from escalation must use same storage format
4. **Pointer**: Goal updates from escalation must work with existing Pointer API

---

## 14. Future Extensions

### Multi-Agent Escalation
Allow System 3 to spawn specialized agents for sub-problems:
- Math agent (Wolfram-augmented)
- Research agent (arXiv-augmented)  
- Code agent (execution-augmented)

### Learned Escalation Policies
Train small model to predict escalation benefit:
- Features: entropy trajectory, search attempts, problem embedding
- Target: did escalation lead to solution?
- Result: smarter triggering

### Hierarchical Escalation Tiers
- **Tier 1**: Local heavy model (self-hosted)
- **Tier 2**: Cloud heavy model (Opus 4)
- **Tier 3**: Multi-agent ensemble

---

## References

- Schmidhuber, J. - Compression Progress & Curiosity
- Hofstadter, D. - Strange Loops & Metacognition
- Kahneman, D. - Thinking, Fast and Slow (System 1/2/3 analogy)
- Anthropic Claude API Documentation
- MCP Protocol Specification

---

**Document Version**: 1.0  
**Status**: Ready for Implementation  
**Next Review**: After Phase 1 completion

