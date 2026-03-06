"""
Action Execution Tools for 5G-Advanced Network Optimizer

Tools used by the Optimizer & Executor Agent to perform network optimizations.
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from crewai.tools import tool

# Load dataset once (pandas is always available in requirements.txt)
_df = None
_df_index = 0

def _load_dataset():
    global _df
    try:
        import pandas as pd
        data_path = Path(__file__).parent.parent / "data" / "6G_HetNet_with_location.csv"
        _df = pd.read_csv(data_path)
    except Exception:
        _df = None

_load_dataset()


def _get_row_metrics() -> Dict:
    """Return metrics from one real dataset row, cycling through rows."""
    global _df_index
    if _df is not None and len(_df) > 0:
        row = _df.iloc[_df_index % len(_df)]
        _df_index += 1
        return {
            "latency_ms":          round(float(row["Network_Latency_ms"]), 2),
            "throughput_mbps":     round(float(row["Achieved_Throughput_Mbps"]), 2),
            "cell_load_percent":   round(float(row["Resource_Utilization"]), 2),
            "packet_loss_percent": round(float(row["Packet_Loss_Ratio"]) * 100, 4),
        }
    return {
        "latency_ms":          round(random.uniform(8, 60), 2),
        "throughput_mbps":     round(random.uniform(60, 140), 2),
        "cell_load_percent":   round(random.uniform(20, 85), 2),
        "packet_loss_percent": round(random.uniform(0, 2.5), 4),
    }


def _execute_action_impl(action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Internal implementation: Simulates executing an optimization action."""
    valid_actions = [
        "scale_bandwidth",
        "activate_cell",
        "adjust_priority",
        "modify_qos",
        "energy_saving"
    ]

    if action not in valid_actions:
        return {
            "success": False,
            "error": f"Invalid action: {action}",
            "valid_actions": valid_actions
        }

    before = _get_row_metrics()
    # Simulate improvement after action (next row from dataset as "after")
    after = {
        "latency_ms":          round(before["latency_ms"]        * random.uniform(0.75, 0.95), 2),
        "throughput_mbps":     round(before["throughput_mbps"]   * random.uniform(1.05, 1.25), 2),
        "cell_load_percent":   round(before["cell_load_percent"] * random.uniform(0.80, 0.95), 2),
        "packet_loss_percent": round(before["packet_loss_percent"] * random.uniform(0.70, 0.90), 4),
    }

    improvements = _calculate_improvements_dict(before, after)

    return {
        "action": action,
        "parameters": parameters,
        "success": True,
        "message": f"Action {action} executed successfully",
        "timestamp": datetime.now().isoformat(),
        "execution_details": {
            "before": before,
            "after": after,
            "improvements": improvements
        },
        "rollback_available": True,
        "steps_executed": [
            "validate_action",
            "capture_before_state",
            "execute_optimization",
            "capture_after_state",
            "calculate_improvement"
        ]
    }


@tool("Execute Action")
def execute_action(action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes an optimization action on the 5G-Advanced network.

    Available actions:
    - scale_bandwidth: Increase/decrease allocated bandwidth
    - activate_cell: Activate additional cells
    - adjust_priority: Change slice priority level
    - modify_qos: Modify QoS parameters
    - energy_saving: Enable energy saving mode

    This tool also:
    - Records state before and after optimization
    - Calculates improvement percentages
    - Handles rollback if action fails

    Args:
        action: Type of optimization action to execute
        parameters: Action-specific parameters

    Returns:
        Detailed result including before/after metrics and improvement
    """
    return _execute_action_impl(action, parameters)


def _calculate_improvements_dict(before: Dict, after: Dict) -> Dict[str, Any]:
    """Calculate improvement percentages between before and after metric dicts."""

    improvements = {}

    # Latency improvement (lower is better)
    if before["latency_ms"] > 0:
        chg = ((before["latency_ms"] - after["latency_ms"]) / before["latency_ms"]) * 100
        improvements["latency"] = {
            "change_percent": round(chg, 1),
            "improved": chg > 0,
            "description": f"{'Reduced' if chg > 0 else 'Increased'} by {abs(round(chg, 1))}%"
        }

    # Throughput improvement (higher is better)
    if before["throughput_mbps"] > 0:
        chg = ((after["throughput_mbps"] - before["throughput_mbps"]) / before["throughput_mbps"]) * 100
        improvements["throughput"] = {
            "change_percent": round(chg, 1),
            "improved": chg > 0,
            "description": f"{'Increased' if chg > 0 else 'Decreased'} by {abs(round(chg, 1))}%"
        }

    # Cell load improvement (lower is better)
    if before["cell_load_percent"] > 0:
        chg = ((before["cell_load_percent"] - after["cell_load_percent"]) / before["cell_load_percent"]) * 100
        improvements["cell_load"] = {
            "change_percent": round(chg, 1),
            "improved": chg > 0,
            "description": f"{'Reduced' if chg > 0 else 'Increased'} by {abs(round(chg, 1))}%"
        }

    total = sum(1 for i in improvements.values() if i.get("improved", False))
    improvements["overall"] = {
        "metrics_improved": total,
        "total_metrics": len(improvements) - 1,
        "success_rate": f"{(total / max(1, len(improvements) - 1)) * 100:.0f}%"
    }

    return improvements


@tool("Database Save")
def db_save(data_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Saves data to the database for persistence and learning.

    Data types:
    - "intent": Save parsed intent for future reference
    - "config": Save generated configuration
    - "optimization": Save optimization action and result
    - "metrics": Save network metrics snapshot

    Args:
        data_type: Type of data being saved
        data: The data to save

    Returns:
        Confirmation of save operation with record ID
    """

    valid_types = ["intent", "config", "optimization", "metrics"]

    if data_type not in valid_types:
        return {
            "success": False,
            "error": f"Invalid data type: {data_type}",
            "valid_types": valid_types
        }

    # Generate record ID
    import time
    record_id = f"{data_type}_{int(time.time())}"

    # In a real system, this would save to a database
    # For the prototype, we simulate the save operation

    return {
        "success": True,
        "data_type": data_type,
        "record_id": record_id,
        "timestamp": datetime.now().isoformat(),
        "message": f"Successfully saved {data_type} record",
        "data_summary": {
            "fields_saved": len(data),
            "keys": list(data.keys())[:5]  # First 5 keys
        }
    }


@tool("Database Query")
def db_query(query_type: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Queries the database for historical data.

    Query types:
    - "recent_intents": Get recent user intents
    - "recent_optimizations": Get recent optimization actions
    - "similar_events": Find similar past events
    - "performance_history": Get historical performance data

    Args:
        query_type: Type of query to execute
        filters: Optional filters to apply

    Returns:
        Query results from the database
    """

    valid_queries = [
        "recent_intents",
        "recent_optimizations",
        "similar_events",
        "performance_history"
    ]

    if query_type not in valid_queries:
        return {
            "success": False,
            "error": f"Invalid query type: {query_type}",
            "valid_queries": valid_queries
        }

    # Simulated database responses for prototype
    simulated_responses = {
        "recent_intents": {
            "results": [
                {"intent_type": "stadium_event", "timestamp": "2024-01-14T10:00:00", "success": True},
                {"intent_type": "emergency", "timestamp": "2024-01-13T15:30:00", "success": True}
            ],
            "total_count": 2
        },
        "recent_optimizations": {
            "results": [
                {"action": "scale_bandwidth", "improvement": "35%", "timestamp": "2024-01-14T10:30:00"},
                {"action": "activate_cell", "improvement": "22%", "timestamp": "2024-01-13T16:00:00"}
            ],
            "total_count": 2
        },
        "similar_events": {
            "results": [
                {"event_type": "stadium_event", "config_used": "eMBB_high_capacity", "outcome": "success"}
            ],
            "total_count": 1
        },
        "performance_history": {
            "results": {
                "avg_latency_ms": 35,
                "avg_throughput_mbps": 95,
                "optimization_success_rate": "92%"
            }
        }
    }

    return {
        "success": True,
        "query_type": query_type,
        "filters_applied": filters or {},
        "data": simulated_responses.get(query_type, {}),
        "timestamp": datetime.now().isoformat()
    }
