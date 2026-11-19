"""Tool arguments and the structured final result.

The tool classes are what the LLM sees. Their fields are plain values, there is no field for a query text.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

REASON = "One short sentence on why this is the right next step."
DISCLAIMER = (
    "A register hit does not show creditworthiness, operational activity or the capacity to act. "
    "A missing hit does not prove that a company does not exist."
)


class LookupUid(BaseModel):
    """Find the company with one Swiss UID. Fast and exact, the input is checked first."""
    model_config = ConfigDict(title="lookup_uid")
    uid: str = Field(description="The UID, for example CHE-242.294.601")
    reason: str = Field(description=REASON)


class ExactName(BaseModel):
    """Find companies with exactly this legal name. Fast, but case sensitive and needs the exact spelling."""
    model_config = ConfigDict(title="exact_name")
    name: str = Field(description="The complete legal name in the assumed register spelling")
    reason: str = Field(description=REASON)


class TextSearch(BaseModel):
    """Search the fast text index by words, in any order. Also finds translated names. Not complete, and accents must fit."""
    model_config = ConfigDict(title="text_search")
    words: str = Field(description="Plain words of the name, no search syntax")
    locality: Optional[str] = Field(default=None, description="Optional part of the place name, keeps only companies there")
    fuzzy: bool = Field(default=False, description="Accept small spelling differences, this also finds wrong companies")
    prefix: bool = Field(default=False, description="Let the last word match as the start of a word")
    reason: str = Field(description=REASON)


class ScanLegalNames(BaseModel):
    """Read all legal names for a part of a name. Takes about 6 seconds and has a small budget. Required before NO_MATCH."""
    model_config = ConfigDict(title="scan_legal_names")
    fragment: str = Field(description="A specific part of the name, without the legal form, at least 3 characters")
    locality: Optional[str] = Field(default=None, description="Optional part of the place name")
    reason: str = Field(description=REASON)


class GetCompanyRecords(BaseModel):
    """Read the full records of up to 5 companies, with a deterministic comparison to the request. Required before MATCH."""
    model_config = ConfigDict(title="get_company_records")
    company_ids: List[str] = Field(description="The company_id values from search results")
    reason: str = Field(description=REASON)


class UpdateRequest(BaseModel):
    """Add a detail from the latest user message to the request. The value must appear in that message."""
    model_config = ConfigDict(title="update_request")
    field: Literal["name", "locality", "address", "uid"] = Field(description="The field to change")
    value: str = Field(description="The new value, in the words of the user")
    reason: str = Field(description=REASON)


class SubmitDecision(BaseModel):
    """Finish this turn with your decision. Call this tool alone."""
    model_config = ConfigDict(title="submit_decision")
    outcome: Literal["MATCH", "AMBIGUOUS", "NO_MATCH", "TECHNICAL_ERROR"]
    company_id: Optional[str] = Field(default=None, description="The company for MATCH")
    candidate_ids: List[str] = Field(default_factory=list, description="The candidates for AMBIGUOUS")
    rationale: str = Field(description="Two or three plain sentences that explain the decision from the tool results")
    uncertainties: List[str] = Field(default_factory=list, description="What remains unclear")
    clarification_question: Optional[str] = Field(default=None, description="One short question to the user, only for AMBIGUOUS")
    answer_to_user: Optional[str] = Field(default=None, description="A direct answer if the user asked a follow-up question")


AGENT_TOOLS = [LookupUid, ExactName, TextSearch, ScanLegalNames, GetCompanyRecords, UpdateRequest, SubmitDecision]


class Company(BaseModel):
    """A company as stated in the register. All values come from the register tool, none from the LLM."""
    company_id: str
    company_uri: str
    legal_name: Optional[str] = None
    other_names: List[str] = []
    uid: Optional[str] = None
    uid_display: Optional[str] = None
    chid: Optional[str] = None
    ehraid: Optional[str] = None
    legal_form: Optional[str] = None
    legal_form_code: Optional[str] = None
    street: Optional[str] = None
    postal_code: Optional[str] = None
    locality: Optional[str] = None
    canton: Optional[str] = None
    municipality: Optional[str] = None


class Candidate(BaseModel):
    """A company seen in a search, which is not an identification."""
    company_id: str
    legal_name: str
    locality: Optional[str] = None


class Evidence(BaseModel):
    """One piece of evidence behind the result."""
    field: str
    value: Optional[str] = None
    company_id: str
    retrieved_at: Optional[str] = None
    source: Optional[str] = None
    simulated: bool = False


class IdentificationResult(BaseModel):
    """The structured answer of the agent for one turn."""
    outcome: Literal["MATCH", "AMBIGUOUS", "NO_MATCH", "TECHNICAL_ERROR", "INVALID_INPUT"]
    request: dict
    company: Optional[Company] = None
    candidates: List[Candidate] = []
    evidence: List[Evidence] = []
    retrieved_at: Optional[str] = None
    register_mode: str
    rationale: str = ""
    answer_to_user: Optional[str] = None
    uncertainties: List[str] = []
    verifier: dict = {}
    removed_statements: List[str] = []
    removed_identifiers: List[str] = []
    tool_calls_used: int = 0
    disclaimer: str = DISCLAIMER
