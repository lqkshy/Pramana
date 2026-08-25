from pydantic import BaseModel


class VerifyRequest(BaseModel):
    text: str


class VerificationResult(BaseModel):
    claim: str
    verdict: str
    confidence: float
    reason: str
    sources: list[str]


class VerifyResponse(BaseModel):
    results: list[VerificationResult]


class ClaimResult(BaseModel):
    claims: list[str]