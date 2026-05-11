from langgraph_agent_lab.graph import build_graph
from pathlib import Path

def main():
    graph = build_graph()
    mermaid_png = graph.get_graph().draw_mermaid_png()
    output_path = Path("reports/graph.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(mermaid_png)
    print(f"Graph diagram saved to {output_path}")
    
    # Also save as mermaid text just in case
    mermaid_text = graph.get_graph().draw_mermaid()
    Path("reports/graph.mermaid").write_text(mermaid_text)
    print(f"Graph mermaid text saved to reports/graph.mermaid")

if __name__ == "__main__":
    main()
