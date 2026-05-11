import os
import sys

# Add src to sys.path
sys.path.append(os.path.join(os.getcwd(), "src"))

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.state import initial_state, Scenario, Route

def test_s07():
    scenario = Scenario(
        id="S07_dead_letter",
        query="System failure cannot recover after multiple attempts",
        expected_route=Route.ERROR,
        max_attempts=1
    )
    state = initial_state(scenario)
    graph = build_graph()
    final_state = graph.invoke(state)
    
    print(f"Scenario: {scenario.id}")
    print(f"Final Answer: {final_state.get('final_answer')}")
    print(f"Attempts: {final_state.get('attempt')}")
    print(f"Events: {[e['node'] for e in final_state.get('events', [])]}")
    
    assert "maximum retry attempts" in final_state.get('final_answer', '').lower()
    assert final_state.get('attempt') == 1

def test_s05_retry():
    scenario = Scenario(
        id="S05_error",
        query="Timeout failure while processing request",
        expected_route=Route.ERROR,
        max_attempts=3
    )
    state = initial_state(scenario)
    graph = build_graph()
    final_state = graph.invoke(state)
    
    print(f"\nScenario: {scenario.id}")
    print(f"Final Answer: {final_state.get('final_answer')}")
    print(f"Attempts: {final_state.get('attempt')}")
    print(f"Events: {[e['node'] for e in final_state.get('events', [])]}")
    
    # In my new logic:
    # 1. Classify -> error
    # 2. Router -> retry (att=1)
    # 3. Tool (att=1). 1 < 1 is false -> SUCCESS.
    # So it should have attempt=1 and succeed.
    
    assert "SUCCESS" in final_state.get('tool_results', [])[-1]
    assert final_state.get('attempt') == 1

if __name__ == "__main__":
    test_s07()
    test_s05_retry()
    print("\nAll verification tests passed!")
