"""Database setup for the FastAPI app using Supabase."""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

# Load environment variables from a .env file into the process.
load_dotenv()


def get_db() -> Client:
    """Create and return a Supabase client.

    Reads the Supabase URL and key from environment variables. It prefers
    SUPABASE_URL + SUPABASE_KEY, but will fall back to DATABASE_URL (used as
    both the URL and key placeholder) if the Supabase-specific variables are
    not set. Raises an error if neither set of credentials is available.
    """
    url = os.getenv("SUPABASE_URL") or os.getenv("DATABASE_URL")
    key = os.getenv("SUPABASE_KEY") or os.getenv("DATABASE_URL")

    if not url or not key:
        raise ValueError(
            "Missing database credentials. Set SUPABASE_URL and SUPABASE_KEY, "
            "or DATABASE_URL, in your environment or .env file."
        )

    return create_client(url, key)


async def create_claims_table() -> None:
    """Check connectivity to the 'claims' table without running raw SQL.

    Performs a simple select against the 'claims' table using the Supabase
    client to verify the table exists and the connection works. It does not
    create or alter any tables (no DDL); it only tests connectivity.
    """
    db = get_db()
    db.table("claims").select("*").limit(1).execute()


if __name__ == "__main__":
    # Create a Supabase client and test a simple query against the "claims"
    # table, printing a success message or a clear error message.
    try:
        db = get_db()
        result = db.table("claims").select("*").limit(1).execute()
        print("Supabase connected OK")
    except Exception as e:
        print("Supabase connection error:")
        print(e)
