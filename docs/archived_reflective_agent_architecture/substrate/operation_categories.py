from enum import Enum


class OperationCategory(Enum):
    INFRASTRUCTURE = "infrastructure"  # Heartbeat, transactions. NEVER looping.
    INTROSPECTION = "introspection"  # check_cognitive_state, diagnose_*. Normal.
    EXPLORATION = "exploration"  # explore_for_utility, hypothesize. High-entropy.
    DELIBERATION = "deliberation"  # synthesize, constrain, deconstruct. Goal-work.
    ROUTINE = "routine"  # User-defined crystallized patterns.


OPERATION_CATEGORY_MAP: dict[str, OperationCategory] = {
    # Infrastructure (exempt from looping)
    "substrate_transaction": OperationCategory.INFRASTRUCTURE,
    "log_operation": OperationCategory.INFRASTRUCTURE,
    # Introspection (exempt from looping)
    "check_cognitive_state": OperationCategory.INTROSPECTION,
    "diagnose_pointer": OperationCategory.INTROSPECTION,
    "diagnose_antifragility": OperationCategory.INTROSPECTION,
    "get_active_goals": OperationCategory.INTROSPECTION,
    "visualize_thought": OperationCategory.INTROSPECTION,
    "recall_work": OperationCategory.INTROSPECTION,
    "get_known_archetypes": OperationCategory.INTROSPECTION,
    "compute_grok_depth": OperationCategory.INTROSPECTION,
    # Exploration (considered for looping, high-entropy expected)
    "explore_for_utility": OperationCategory.EXPLORATION,
    "hypothesize": OperationCategory.EXPLORATION,
    "consult_curiosity": OperationCategory.EXPLORATION,
    "consult_ruminator": OperationCategory.EXPLORATION,
    "consult_advisor": OperationCategory.EXPLORATION,
    "list_advisors": OperationCategory.EXPLORATION,
    "inspect_graph": OperationCategory.EXPLORATION,
    "inspect_knowledge_graph": OperationCategory.EXPLORATION,
    # Deliberation (primary looping candidates)
    "deconstruct": OperationCategory.DELIBERATION,
    "synthesize": OperationCategory.DELIBERATION,
    "constrain": OperationCategory.DELIBERATION,
    "revise": OperationCategory.DELIBERATION,
    "set_goal": OperationCategory.DELIBERATION,
    "compress_to_tool": OperationCategory.DELIBERATION,
    "evolve_formula": OperationCategory.DELIBERATION,
    "resolve_meta_paradox": OperationCategory.DELIBERATION,
    "teach_cognitive_state": OperationCategory.DELIBERATION,
    "propose_goal": OperationCategory.DELIBERATION,
    "orthogonal_dimensions_analyzer": OperationCategory.DELIBERATION,
    "create_advisor": OperationCategory.DELIBERATION,
    "delete_advisor": OperationCategory.DELIBERATION,
    "consult_compass": OperationCategory.DELIBERATION,
    "run_sleep_cycle": OperationCategory.DELIBERATION,
}


def get_category(operation: str) -> OperationCategory:
    return OPERATION_CATEGORY_MAP.get(operation, OperationCategory.DELIBERATION)


def is_exempt_from_looping(operation: str) -> bool:
    cat = get_category(operation)
    return cat in (
        OperationCategory.INFRASTRUCTURE,
        OperationCategory.INTROSPECTION,
        OperationCategory.ROUTINE,
    )
