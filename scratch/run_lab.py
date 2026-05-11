import yaml
from pathlib import Path
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics, write_metrics
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.report import write_report
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import initial_state

def main():
    config_path = Path("configs/lab.yaml")
    output_path = Path("outputs/metrics.json")
    
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    
    metrics = []
    for scenario in scenarios:
        print(f"Running scenario: {scenario.id}")
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(metric_from_state(final_state, scenario.expected_route.value, scenario.requires_approval))
        
    report = summarize_metrics(metrics)
    write_metrics(report, output_path)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    print(f"Wrote metrics to {output_path}")

if __name__ == "__main__":
    main()
