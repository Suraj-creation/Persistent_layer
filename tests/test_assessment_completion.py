"""Completion and continued-question regressions for the reported student flow."""
from test_conversation_assessment import start, answer, flow


def test_submit_and_finish_saves_answer_without_generating_an_unused_question(flow):
    client, provider = flow
    session = start(client).json()
    generated = len(provider.question_contexts)
    result = answer(client, session, finish_after_answer=True)
    assert result.status_code == 200, result.text
    completed = result.json()
    assert completed["conversation"]["status"] == "completed"
    assert len(completed["turns"]) == 1
    assert completed["active_question"] is None
    assert len(provider.question_contexts) == generated
    # A retry must neither submit the answer twice nor generate another question.
    duplicate = answer(client, session, finish_after_answer=True).json()
    assert len(duplicate["turns"]) == 1


def test_finish_without_submitting_keeps_pending_question_for_resume(flow):
    client, _ = flow
    session = start(client).json()
    cid = session["conversation"]["conversation_id"]
    finished = client.post(f"/api/v2/conversations/{cid}/finish", json={"learner_id": "real-user"}).json()
    assert finished["state"] == session["state"]
    assert finished["turns"] == []
    assert finished["conversation"]["current_question_id"] == session["active_question"]["question_id"]
    resumed = client.post(f"/api/v2/conversations/{cid}/resume", json={"learner_id": "real-user"}).json()
    assert resumed["active_question"]["question_id"] == session["active_question"]["question_id"]


def test_repeated_repair_varies_question_kind_and_rechecks_one_skill(flow):
    client, _ = flow
    session = start(client).json()
    first = answer(client, session, "I use a wrong rule because I think tokens must always equal complete words.").json()
    second = answer(client, first, "I still use a wrong rule and assume a token always equals one word.").json()
    next_plan = second["active_question"]["generation_trace"]["plan"]
    assert next_plan["intent"] == "repair"
    assert next_plan["policy_version"] == "focused-adaptive-planner-v2"
    assert next_plan["kind"] != first["active_question"]["kind"]
    assert len(next_plan["required_criteria"]) == 1


def test_assessment_can_continue_beyond_three_followups(flow):
    client, provider = flow
    original = provider.generate_question
    distinct_tasks = [
        "Explain how word boundaries preserve a sentence's individual units.",
        "A parser merges adjacent words. Identify the resulting loss of information.",
        "Compare punctuation handling in two sentence segmentation strategies.",
        "Design a vocabulary that treats an unfamiliar term as several pieces.",
        "Diagnose why splitting a contraction by spaces can retain ambiguity.",
        "Apply the source distinction to the representation of an unseen compound.",
        "Justify the order of tokens while explaining the difference between units.",
        "Analyze an example in which one written word has several model tokens.",
    ]

    def varied(context):
        result = original(context)
        result["prompt"] = distinct_tasks[len(context["previous_questions"]) % len(distinct_tasks)]
        return result

    provider.generate_question = varied
    session = start(client).json()
    for index in range(6):
        response = answer(client, session, f"I guess about token boundaries and need another explanation of this distinction, attempt {index}.")
        assert response.status_code == 200
        session = response.json()
        assert session["active_question"] is not None
        assert session["conversation"]["answered_count"] == index + 1
