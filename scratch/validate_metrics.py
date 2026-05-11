import json
from pathlib import Path
from langgraph_agent_lab.metrics import MetricsReport

def main():
    metrics_path = Path("outputs/metrics.json")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        print("Error: Expected at least 6 scenarios")
        return
    print(f"Metrics valid. success_rate={report.success_rate:.2%}")
    print(f"Total scenarios: {report.total_scenarios}")
    print(f"Average nodes visited: {report.avg_nodes_visited:.2f}")
    print(f"Total retries: {report.total_retries}")
    print(f"Total interrupts: {report.total_interrupts}")

if __name__ == "__main__":
    main()
