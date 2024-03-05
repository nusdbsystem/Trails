#!/bin/bash

# Those cmds will triggered after docker run .

# Compile code, and run postgresql
cd /project/Trails/internal/pg_extension || exit
/bin/bash -c "source $HOME/.cargo/env && echo '\q' | cargo pgrx run --release"

# Wait for PostgreSQL to become available
until psql -h localhost -p 28814 -U postgres -d pg_extension -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

# Run setup commands
psql -h localhost -p 28814 -U postgres -d pg_extension -c "CREATE EXTENSION pg_extension;"
psql -h localhost -p 28814 -U postgres -d pg_extension -f /project/Trails/internal/pg_extension/sql/model_selection_cpu.sql
# Load example dataset into database
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/Trails/dataset/frappe frappe 28814

# Continue with the rest of your container's CMD
tail -f /dev/null

echo "Done!"
