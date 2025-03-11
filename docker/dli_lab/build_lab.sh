cp ../../docs_input/notebooks/scripts/run_matx.py ./
docker build -f lab.Dockerfile -t ghcr.io/nvidia/matx/lab:latest .
rm run_matx.py
