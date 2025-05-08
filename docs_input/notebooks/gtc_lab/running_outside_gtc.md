# Notebook Startup (Not needed for GTC Lab)

## Container Startup 
Start container with all normal options, adding `-p 8888:8888`

a sample `run.sh` script is provided in `MatX/docs_input/notebooks`

## Start Jupyter server locally in container
`jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root` 

copy the token from the server start (specifically the local token should be something similar to):
`http://127.0.0.1:8888/tree?token=a3ad60a152dcafe98d4eaecc22bd773b38f1e6e93312adae`

Since Jupyter is binding to localhost by default, you may need to change the IP address to access it from another machine.