#!/usr/bin/env fish

set -l repo_root (realpath (dirname (dirname (status --current-filename))))
set -l dataset /home/guz/Code/fish-mask/data/fishnet-filtered-yolo-dataset/data.yaml
set -l tracking_uri http://127.0.0.1:5000
set -l experiment_name fish_segmentation
set -l run_name fishnet_filtered_no_oth
set -l server_log "$repo_root/mlflow-server.log"

cd $repo_root; or exit 1
set -lx MLFLOW_TRACKING_URI $tracking_uri
set -lx MLFLOW_EXPERIMENT_NAME $experiment_name
set -lx MLFLOW_RUN $run_name

# Activate environment
source $repo_root/.venv/bin/activate.fish; or exit 1

uv run mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --workers 1 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    >$server_log 2>&1 &
set server_pid $last_pid

echo "Waiting for MLflow at $tracking_uri..."

function stop_mlflow --on-signal INT --on-signal TERM
    kill $server_pid 2>/dev/null
    exit 130
end

curl --fail --silent --retry 30 --retry-delay 1 --retry-connrefused \
    $tracking_uri/health >/dev/null; or begin
    echo "MLflow failed to start. See $server_log"
    kill $server_pid 2>/dev/null
    exit 1
end

uv run python -m fish_segmentation_model.train \
    --data $dataset \
    --mlflow
set training_status $status

kill $server_pid 2>/dev/null
exit $training_status
