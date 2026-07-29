if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/waymo/prepare-flow.sh <path-to-pinned-cotracker-repo>" >&2
    exit 2
fi

python scripts/flow.py ./data/waymo/scene006 --cotracker_repo "$1"
python scripts/flow.py ./data/waymo/scene026 --cotracker_repo "$1"
python scripts/flow.py ./data/waymo/scene090 --cotracker_repo "$1"
python scripts/flow.py ./data/waymo/scene105 --cotracker_repo "$1"
python scripts/flow.py ./data/waymo/scene108 --cotracker_repo "$1"
python scripts/flow.py ./data/waymo/scene134 --cotracker_repo "$1"
python scripts/flow.py ./data/waymo/scene150 --cotracker_repo "$1"
python scripts/flow.py ./data/waymo/scene181 --cotracker_repo "$1"
