from app.models.schemas import ClaimInput, ExtractedClaim, VerdictEnum, VerificationResult, VerifyResponse

# Test ClaimInput
ci = ClaimInput(text='  Tesla was founded in 2003.  ')
assert ci.text == 'Tesla was founded in 2003.', f'strip failed: {ci.text!r}'
print('ClaimInput OK')

# Test ExtractedClaim
ec = ExtractedClaim(
    original='Tesla was founded in 2003',
    disambiguated='Tesla, the EV company, was founded in 2003',
    sub_claims=['Tesla is an EV company', 'Tesla was founded in 2003'],
)
assert len(ec.sub_claims) == 2
print('ExtractedClaim OK')

# Test VerdictEnum
assert VerdictEnum.SUPPORTED == 'SUPPORTED'
assert VerdictEnum.CONTRADICTED == 'CONTRADICTED'
assert VerdictEnum.INSUFFICIENT_EVIDENCE == 'INSUFFICIENT_EVIDENCE'
assert VerdictEnum.CONFLICTING_EVIDENCE == 'CONFLICTING_EVIDENCE'
print('VerdictEnum OK')

# Test VerificationResult: valid SUPPORTED
vr = VerificationResult(
    claim='Tesla was founded in 2003',
    verdict='SUPPORTED',
    confidence=0.95,
    evidence_strength=0.88,
    explanation='Multiple sources confirm July 2003.',
    matched_claim_id='claim_001',
)
assert vr.verdict == 'SUPPORTED'
print('VerificationResult SUPPORTED OK')

# Test VerificationResult: valid INSUFFICIENT_EVIDENCE low confidence
vr2 = VerificationResult(
    claim='Tesla is the most valuable car company',
    verdict='INSUFFICIENT_EVIDENCE',
    confidence=0.3,
    evidence_strength=0.2,
    explanation='Insufficient public data found.',
)
assert vr2.matched_claim_id is None
print('VerificationResult INSUFFICIENT_EVIDENCE OK')

# Test VerifyResponse
resp = VerifyResponse(claims=[vr, vr2])
assert len(resp.claims) == 2
print('VerifyResponse OK')

# Test validation errors
import sys

try:
    ClaimInput(text='')
    print('FAIL: empty text should be rejected')
    sys.exit(1)
except Exception:
    pass

try:
    ExtractedClaim(
        original='x', disambiguated='y', sub_claims=['valid', '   ']
    )
    print('FAIL: whitespace-only sub_claim should be rejected')
    sys.exit(1)
except Exception:
    pass

try:
    VerificationResult(
        claim='x', verdict='SUPPORTED', confidence=0.2,
        evidence_strength=0.5, explanation='low conf'
    )
    print('FAIL: SUPPORTED + confidence<0.5 should be rejected')
    sys.exit(1)
except Exception:
    pass

print('All assertions passed.')