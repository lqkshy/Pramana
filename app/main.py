from fastapi import FastAPI
from app.models.schemas import VerifyRequest, VerifyResponse, VerificationResult

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-claims")
def extract_claims_endpoint(request: VerifyRequest):
    try:
        from app.pipeline.claims import extract_claims
        claims = extract_claims(request.text)
        return ClaimResult(claims=claims)
    except Exception as e:
        return {"detail": str(e)}, 400


@app.post("/verify")
def verify_endpoint(request: VerifyRequest):
    try:
        from app.pipeline.orchestrator import run_pipeline
        results = run_pipeline(request.text)
        return VerifyResponse(results=results)
    except Exception as e:
        return {"detail": str(e)}, 400


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)