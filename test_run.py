import sys
sys.path.insert(0, r'C:\Users\laksh\Desktop\Pramana')
from app.pipeline.claims.extractor import extract_claims
import asyncio

result = asyncio.run(extract_claims('Elon Musk founded Tesla in 2003 and it is now the world\'s most valuable car company with over 100 billion in revenue.'))
print('Result:', result)