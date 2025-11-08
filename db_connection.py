from sqlalchemy import create_engine

# Replace credentials with yours
DB_USER = "postgres"
DB_PASS = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "fluz_ds"

def get_engine():
    """Returns a SQLAlchemy engine for PostgreSQL connection"""
    conn_str = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(conn_str)
    return engine