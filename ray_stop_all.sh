#!/bin/bash
# Script to stop all running Ray jobs on the cluster
# Usage: ./ray_stop_all.sh [ray_address]
# Default: http://127.0.0.1:8266

# Set default Ray address
DEFAULT_ADDRESS="http://127.0.0.1:8265"

# Use first argument if provided, otherwise use default
RAY_ADDRESS=${1:-$DEFAULT_ADDRESS}

echo "Using Ray address: $RAY_ADDRESS"

echo "Fetching list of running Ray jobs..."

eval $(poetry env activate)

# Get list of running job_ids using Python
RUNNING_JOBS=$(python3 - <<EOF
import ray
from ray.dashboard.modules.job.sdk import JobSubmissionClient

client = JobSubmissionClient("$RAY_ADDRESS")
jobs = client.list_jobs()
running_ids = [job.job_id for job in jobs if job.status.name == "RUNNING" and job.job_id]
print(" ".join(running_ids))
EOF
)

# Check if any running jobs were found
if [ -z "$RUNNING_JOBS" ]; then
  echo "No running jobs found."
  exit 0
fi

# Stop each running job
for JOB_ID in $RUNNING_JOBS; do
  echo "Stopping job: $JOB_ID"
  ray job stop --address="$RAY_ADDRESS" "$JOB_ID"
done

echo "All running jobs have been stopped."
