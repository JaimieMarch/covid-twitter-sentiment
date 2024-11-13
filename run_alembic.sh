#!/bin/bash

# Navigate to the root directory of your project

# Run this command if changes to tables occur
# alembic stamp head

# Generate new migration schema
alembic -c alembic.ini revision --autogenerate -m "Added table"
