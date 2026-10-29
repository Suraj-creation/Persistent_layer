import httpx
import json
import time

BASE_URL = "http://127.0.0.1:8001/api/v1"

def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_turn_result(turn_num: int, user_msg: str, res: dict):
    print(f"\n--- Turn {turn_num} ---")
    print(f"User: \"{user_msg}\"")
    print(f"Tutor Reply: \"{res['reply']}\"")
    
    ev = res.get('evidence', {})
    interp = ev.get('candidate_interpretation', {})
    print(f"\n[Extracted Evidence]")
    print(f"  • Verbatim Quote: \"{ev.get('source_quote')}\"")
    print(f"  • Extracted Claim: {ev.get('extracted_claim')}")
    print(f"  • Category: {interp.get('category')} (Confidence: {interp.get('hypothesis_confidence', 0):.0%})")
    print(f"  • Hypothesis ID: {interp.get('hypothesis_id')}")
    print(f"  • Grounded KC: {[cl['concept_id'] for cl in ev.get('concept_links', [])]}")
    
    state = res.get('new_state', {}).get('concept_states', {}).get('fraction_addition_unlike_denominators', {})
    print(f"\n[Persistent Beta-Bayesian State in SQLite]")
    print(f"  • Alpha: {state.get('alpha')} | Beta: {state.get('beta')}")
    print(f"  • Mastery E[L]: {state.get('mastery_estimate', 0):.1%}")
    print(f"  • Epistemic Uncertainty (sd): {state.get('uncertainty', 0):.1%}")
    print(f"  • Sufficiency: {state.get('evidence_sufficiency')}")
    
    misconceptions = state.get('candidate_misconceptions', {})
    if misconceptions:
        print(f"  • Candidate Misconceptions:")
        for mid, m in misconceptions.items():
            print(f"    - {m['misconception_name']}: {m['posterior_probability']:.1%} [{m['status'].upper()}]")
    else:
        print(f"  • Candidate Misconceptions: None")

    events = res.get('events', [])
    if events:
        print(f"\n[State Change Events & Math Traces ({len(events)})]")
        for e in events:
            print(f"  • Rule Applied: {e['applied_rule']}")
            print(f"    Summary: {e['change_summary']}")
            mt = e.get('math_trace', {})
            if mt:
                print(f"    Trace: alpha {mt.get('prior_alpha')} -> {mt.get('posterior_alpha')}, beta {mt.get('prior_beta')} -> {mt.get('posterior_beta')}")


def run_learner_simulation(learner_id: str, turns: list):
    print_banner(f"STARTING SIMULATION FOR LEARNER: {learner_id}")
    history = []
    
    for idx, turn in enumerate(turns, 1):
        payload = {
            "learner_id": learner_id,
            "message": turn["message"],
            "item_id": turn.get("item_id", "ITEM_FRAC_01"),
            "item_text": turn.get("item_text", "Solve: 1/2 + 1/3"),
            "selected_option": turn.get("selected_option"),
            "chat_history": history,
        }
        
        resp = httpx.post(f"{BASE_URL}/chat", json=payload, timeout=40.0)
        assert resp.status_code == 200, f"Error {resp.status_code}: {resp.text}"
        res = resp.json()
        
        print_turn_result(idx, turn["message"], res)
        
        history.append({"role": "user", "content": turn["message"]})
        history.append({"role": "assistant", "content": res["reply"]})
        time.sleep(1)

    # Fetch and print final audit report
    print_banner(f"FINAL DETERMINISTIC AUDIT REPORT FOR: {learner_id}")
    audit_resp = httpx.get(f"{BASE_URL}/learners/{learner_id}/audit_report", timeout=10.0)
    assert audit_resp.status_code == 200
    report_data = audit_resp.json()
    print(report_data["report_markdown"])


if __name__ == "__main__":
    # -------------------------------------------------------------
    # Simulation 1: Dumb / Struggling Learner (Persistent Bug)
    # -------------------------------------------------------------
    struggling_turns = [
        {
            "message": "I think the answer is 2/5 because 1+1 is 2 on top, and 2+3 is 5 on the bottom. It's just simple addition.",
            "selected_option": "2/5",
        },
        {
            "message": "I don't understand what you mean by slice sizes. In math class, whenever you have numbers separated by a plus sign, you just add them together. So 1+1=2 and 2+3=5 makes total sense to me.",
            "selected_option": "2/5",
        },
        {
            "message": "Wait, what even is a common denominator? Why can't I just add denominators? What does denominator mean?",
            "selected_option": None,
        },
    ]
    run_learner_simulation("sim_learner_dumb", struggling_turns)

    # -------------------------------------------------------------
    # Simulation 2: Recovering Learner (Bug -> Socratic -> Aha -> Slip -> Fix)
    # -------------------------------------------------------------
    recovering_turns = [
        {
            "message": "The answer is definitely 2/5. You just add both parts.",
            "selected_option": "2/5",
        },
        {
            "message": "Oh wait, I see your point about pizza slices! If I have half a pizza and a third of a pizza, the slices are different sizes. 1/2 is 3/6 and 1/3 is 2/6. So 3/6 + 2/6 = 5/6!",
            "selected_option": "5/6",
        },
        {
            "message": "Now for 1/4 + 1/2, I convert 1/2 to 2/4. But while multiplying in my head I accidentally multiplied 1*2 and wrote 4/4 so I got 5/4. My bad, arithmetic calculation error.",
            "selected_option": "5/4",
        },
        {
            "message": "Let me correct my arithmetic slip: 1/4 + 2/4 = 3/4. Both have denominator 4 now.",
            "selected_option": "3/4",
        },
    ]
    run_learner_simulation("sim_learner_recovering", recovering_turns)

    # -------------------------------------------------------------
    # Simulation 3: Smart Learner (Demonstrated Mastery)
    # -------------------------------------------------------------
    smart_turns = [
        {
            "message": "To solve 1/2 + 1/3, we cannot add them directly because they have unlike denominators. The LCM of 2 and 3 is 6. We convert 1/2 into 3/6 by multiplying by 3/3, and 1/3 into 2/6 by multiplying by 2/2. Then 3/6 + 2/6 equals 5/6.",
            "selected_option": "5/6",
        },
        {
            "message": "Similarly for 3/8 + 1/4, the common denominator is 8 because 8 is a multiple of 4. 1/4 converts to 2/8. Adding 3/8 + 2/8 gives 5/8.",
            "selected_option": "5/8",
            "item_id": "ITEM_FRAC_02",
            "item_text": "Solve: 3/8 + 1/4",
        },
    ]
    run_learner_simulation("sim_learner_smart", smart_turns)
