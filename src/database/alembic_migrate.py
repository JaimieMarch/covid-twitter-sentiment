import subprocess

def run_alembic_migration(message):
    # Run alembic revision --autogenerate to create the migration script
    try:
        print(f"Running Alembic revision to generate migration: {message}")
        subprocess.run(['alembic', 'revision', '--autogenerate', '-m', message], check=True)
        print("Migration file created successfully.")

        # Run alembic upgrade to apply the migration
        print("Running Alembic upgrade to apply the migration.")
        subprocess.run(['alembic', 'upgrade', 'head'], check=True)
        print("Migration applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running Alembic commands: {e}")
