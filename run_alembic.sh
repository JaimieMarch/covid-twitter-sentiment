#!/bin/bash

# Navigate to the root directory of your project

# Run the Alembic revision command
alembic stamp head
alembic -c alembic.ini revision --autogenerate -m "Added table"
