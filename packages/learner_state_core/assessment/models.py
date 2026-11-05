from __future__ import annotations

import uuid
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


def new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


class ConceptDefinition(BaseModel):
    concept_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,70}$")
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=10, max_length=1800)
    prerequisites: list[str] = Field(default_factory=list, max_length=8)


class TopicDefinition(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    overview: str = Field(min_length=20, max_length=10000)
    concepts: list[ConceptDefinition] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def valid_graph(self):
        graph = {c.concept_id: c.prerequisites for c in self.concepts}
        if len(graph) != len(self.concepts):
            raise ValueError("Duplicate concepts")
        visited, active = set(), set()

        def visit(cid):
            if cid not in graph or cid in active:
                raise ValueError("Invalid or cyclic prerequisites")
            if cid in visited:
                return
            active.add(cid)
            for parent in graph[cid]:
                visit(parent)
            active.remove(cid)
            visited.add(cid)

        for cid in graph:
            visit(cid)
        return self


QuestionKind = Literal["explain", "apply", "contrast", "debug", "transfer", "guided_practice"]


class Criterion(BaseModel):
    criterion_id: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=8, max_length=800)


class GeneratedQuestion(BaseModel):
    concept_id: str
    kind: QuestionKind
    prompt: str = Field(min_length=15, max_length=3500)
    expected_focus: str = Field(min_length=10, max_length=900)
    learning_support: str = Field(default="", max_length=1800)
    criteria: list[Criterion] = Field(min_length=1, max_length=4)
    reference_answer: str = Field(min_length=10, max_length=3500)
    source_citations: list[str] = Field(min_length=1, max_length=8)


class ExtractedObservation(BaseModel):
    concept_id: str
    source_quote: str = Field(min_length=1, max_length=8000)
    claim: str = Field(min_length=5, max_length=1600)
    category: Literal["understanding", "partial", "misconception", "prerequisite_gap", "slip", "ambiguous"]
    confidence: float = Field(ge=0, le=1)
    met_criteria: list[str] = Field(max_length=4)
    missing_criteria: list[str] = Field(max_length=4)
    misconception: str | None = Field(None, max_length=1200)
    contradicts_hypothesis_ids: list[str] = Field(default_factory=list, max_length=8)
    feedback: str = Field(min_length=5, max_length=1500)
    coaching: str = Field(default="", max_length=1800)
    source_citations: list[str] = Field(min_length=1, max_length=8)


class IdentityRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    learner_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")


class StartConversation(IdentityRequest):
    model_config = ConfigDict(str_strip_whitespace=False)
    request_id: str = Field(default_factory=lambda: new_id("request"), min_length=1, max_length=120)
    topic: str = Field(min_length=2, max_length=180)
    notes: str = Field(default="", max_length=60000)

    @field_validator("topic", mode="before")
    @classmethod
    def trim_topic(cls, value):
        return value.strip() if isinstance(value, str) else value


class SourceUpdateRequest(IdentityRequest):
    mode: Literal['study_notes', 'model_reference']
    notes: str = Field(default='', max_length=60000)


class AnswerRequest(IdentityRequest):
    model_config = ConfigDict(str_strip_whitespace=False)
    request_id: str = Field(default_factory=lambda: new_id("request"), min_length=1, max_length=120)
    question_id: str = Field(min_length=1, max_length=80)
    response_text: str = Field(min_length=1, max_length=8000)
    finish_after_answer: bool = False

    @field_validator("response_text")
    @classmethod
    def nonblank_observation(cls, value):
        if not value.strip():
            raise ValueError("An explanation cannot be blank.")
        return value
