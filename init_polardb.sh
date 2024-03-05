#!/bin/bash

# Wait for PostgreSQL to become available
until psql -h localhost -p 5432 -U postgres -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

# Run setup commands
psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE pg_extension;"
psql -h localhost -p 5432 -U postgres -d pg_extension -c "CREATE EXTENSION pg_extension;"
psql -h localhost -p 5432 -U postgres -d pg_extension -f /home/postgres/Trails/internal/pg_extension/sql/model_selection_cpu.sql
# Load example dataset into database
bash /home/postgres/Trails/internal/ml/model_selection/scripts/database/load_data_to_db.sh //home/postgres/Trails/dataset/frappe frappe 5432

echo "Done!"
