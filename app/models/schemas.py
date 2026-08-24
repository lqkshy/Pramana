from pydantic import BaseModel


class VerifyRequest(BaseModel):
    text: str


class ClaimResult(BaseModel):
    claims: list[str]