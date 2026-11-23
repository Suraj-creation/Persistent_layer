# Defer type annotation evaluation for portable model definitions.
from __future__ import annotations

# Generate unique IDs on the application side.
import uuid
# List the exact valid strings for controlled model and request values.
from typing import Literal
# Use Pydantic to parse and validate the API and inference data contracts.
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


# Create an ID with a prefix that makes the record kind recognizable.
def new_id(prefix):
    # Add a random UUID so separately created records get distinct IDs.
    return f"{prefix}_{uuid.uuid4().hex}"


# Define one skill in the topic's prerequisite graph.
class ConceptDefinition(BaseModel):
    # Require a stable machine-readable concept ID.
    concept_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,70}$")
    # Save the displayed skill name.
    name: str = Field(min_length=1, max_length=160)
    # Store explanatory text used in planning and retrieval.
    description: str = Field(min_length=10, max_length=1800)
    # Keep the IDs of up to eight skills that come before this one.
    prerequisites: list[str] = Field(default_factory=list, max_length=8)


# Define the concept map and reference overview for a topic.
class TopicDefinition(BaseModel):
    # Save the topic title.
    title: str = Field(min_length=1, max_length=180)
    # Require a substantive overview within a storage limit.
    overview: str = Field(min_length=20, max_length=10000)
    # Bound the map to between two and eight concepts.
    concepts: list[ConceptDefinition] = Field(min_length=2, max_length=8)

    # Validate the relationships once each concept record has been parsed.
    @model_validator(mode="after")
    # Ensure all prerequisite references point to real concepts and form no cycle.
    def valid_graph(self):
        # Build a lookup from concept ID to its required earlier concepts.
        graph = {c.concept_id: c.prerequisites for c in self.concepts}
        # Detect repeated concept IDs hidden by dictionary key replacement.
        if len(graph) != len(self.concepts):
            # Reject a topic with ambiguous concept identities.
            raise ValueError("Duplicate concepts")
        # Track concepts fully inspected and concepts currently on the search path.
        visited, active = set(), set()

        # Walk the graph recursively from a single concept.
        def visit(cid):
            # Reject unknown prerequisites or an edge back to the active path.
            if cid not in graph or cid in active:
                # Report an invalid or cyclic prerequisite graph.
                raise ValueError("Invalid or cyclic prerequisites")
            # Skip a concept already checked from another branch.
            if cid in visited:
                # End this recursive step when no further checks are needed.
                return
            # Mark the current concept as active on this search path.
            active.add(cid)
            # Examine the prerequisites listed for this concept.
            for parent in graph[cid]:
                # Recursively validate each prerequisite chain.
                visit(parent)
            # Leave this concept's active path after all prerequisite edges passed.
            active.remove(cid)
            # Mark this concept fully validated.
            visited.add(cid)

        # Start a traversal for each concept to cover disconnected subgraphs too.
        for cid in graph:
            # Validate the concept and its prerequisite chain.
            visit(cid)
        # Return the accepted topic model.
        return self


# Define the question formats the planner and provider use.
QuestionKind = Literal["explain", "apply", "contrast", "debug", "transfer", "guided_practice"]


# Describe one independently scored item of a question rubric.
class Criterion(BaseModel):
    # Give the rubric item an identifier that the model returns during assessment.
    criterion_id: str = Field(min_length=1, max_length=80)
    # Describe the skill or reasoning the answer must demonstrate.
    description: str = Field(min_length=8, max_length=800)


# Define the structured question proposed by the model.
class GeneratedQuestion(BaseModel):
    # Link the task to the assigned topic concept.
    concept_id: str
    # Record the required question format.
    kind: QuestionKind
    # Store the actual learner-facing question text.
    prompt: str = Field(min_length=15, max_length=3500)
    # State what reasoning the learner should focus on.
    expected_focus: str = Field(min_length=10, max_length=900)
    # Allow a brief source-grounded explanation for guided repair.
    learning_support: str = Field(default="", max_length=1800)
    # Bound the rubric to one to four items.
    criteria: list[Criterion] = Field(min_length=1, max_length=4)
    # Store the private reference explanation for later grading.
    reference_answer: str = Field(min_length=10, max_length=3500)
    # Require at least one reference chunk ID.
    source_citations: list[str] = Field(min_length=1, max_length=8)


# Define the structured interpretation of the submitted explanation.
class ExtractedObservation(BaseModel):
    # Identify which concept this response is being assessed against.
    concept_id: str
    # Keep the verbatim excerpt from the learner.
    source_quote: str = Field(min_length=1, max_length=8000)
    # Summarize the evidence claim grounded in that excerpt.
    claim: str = Field(min_length=5, max_length=1600)
    # Select one of six evidence categories supported by the state policy.
    category: Literal["understanding", "partial", "misconception", "prerequisite_gap", "slip", "ambiguous"]
    # Keep model confidence between zero and one.
    confidence: float = Field(ge=0, le=1)
    # List IDs for the current skills the answer demonstrated.
    met_criteria: list[str] = Field(max_length=4)
    # List IDs for the current skills it did not demonstrate.
    missing_criteria: list[str] = Field(max_length=4)
    # Include a possible incorrect belief when there is evidence for one.
    misconception: str | None = Field(None, max_length=1200)
    # Reference prior hypotheses only when the current answer contradicts them.
    contradicts_hypothesis_ids: list[str] = Field(default_factory=list, max_length=8)
    # Store the brief feedback intended for the learner.
    feedback: str = Field(min_length=5, max_length=1500)
    # Allow optional coaching based on retrieved material.
    coaching: str = Field(default="", max_length=1800)
    # Require factual reference IDs for the interpretation and guidance.
    source_citations: list[str] = Field(min_length=1, max_length=8)


# Share the learner identity field between assessment request types.
class IdentityRequest(BaseModel):
    # Trim strings by default for request types that use this configuration.
    model_config = ConfigDict(str_strip_whitespace=True)
    # Accept a short local identity made of letters, digits, underscores or hyphens.
    learner_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")


# Define what the client sends when starting an assessment.
class StartConversation(IdentityRequest):
    # Preserve the original supplied notes rather than stripping their whitespace.
    model_config = ConfigDict(str_strip_whitespace=False)
    # Allow a caller-provided retry ID or create one for the request.
    request_id: str = Field(default_factory=lambda: new_id("request"), min_length=1, max_length=120)
    # Require a nontrivial topic name within the API size limit.
    topic: str = Field(min_length=2, max_length=180)
    # Allow optional notes to ground this assessment.
    notes: str = Field(default="", max_length=60000)

    # Clean the topic before the length and type checks.
    @field_validator("topic", mode="before")
    # Run the cleanup as a class-level validator.
    @classmethod
    # Trim the topic independently of the raw notes.
    def trim_topic(cls, value):
        # Clean a string and let normal validation handle other input types.
        return value.strip() if isinstance(value, str) else value


# Define the available ways to replace source material.
class SourceUpdateRequest(IdentityRequest):
    # Let the learner select supplied notes or an explicitly model-authored reference.
    mode: Literal['study_notes', 'model_reference']
    # Keep the optional replacement notes for the service.
    notes: str = Field(default='', max_length=60000)


# Define the request to submit one explanation.
class AnswerRequest(IdentityRequest):
    # Preserve exact whitespace so raw text and quote offsets remain meaningful.
    model_config = ConfigDict(str_strip_whitespace=False)
    # Support safely retrying an answer with the same request ID.
    request_id: str = Field(default_factory=lambda: new_id("request"), min_length=1, max_length=120)
    # Identify the current saved question the answer belongs to.
    question_id: str = Field(min_length=1, max_length=80)
    # Store the raw response within the API size limit.
    response_text: str = Field(min_length=1, max_length=8000)
    # Allow the learner to submit and finish in the same operation.
    finish_after_answer: bool = False

    # Apply the extra check specifically to the response text.
    @field_validator("response_text")
    # Run it as a class-level validator.
    @classmethod
    # Reject whitespace-only explanations.
    def nonblank_observation(cls, value):
        # Inspect a trimmed view without changing the original submitted value.
        if not value.strip():
            # Return a useful validation message when the answer contains only whitespace.
            raise ValueError("An explanation cannot be blank.")
        # Return the original text for storage, extraction and offsets.
        return value
