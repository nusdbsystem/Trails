# TRAILS: A Database Native Model Selection System

![image-20230702035806963](internal/ml/model_selection/documents/imgs/image-20230702035806963.png)

# Build & Run examples

## PyTorch + PostgreSQL

```bash
# Create project folder.
mkdir project && cd project
# Download the Dockerile.
wget -O Dockerfile https://raw.githubusercontent.com/NLGithubWP/Trails/main/torch.psql.Dockerfile

# Build Dockerile and run the docker.
docker build -t trails .
docker run -d --name trails --network="host" trails
# Monitor the logs until the setup step is done.
docker logs -f trails

docker exec -it trails bash
# Connect to the pg server and use pg_extension database.
psql -h localhost -p 28814 -U postgres
\c pg_extension

# Run an example, wait one min, it will run filtering + refinemnt + training the selected model.
CALL model_selection_end2end('frappe_train', ARRAY['col1', 'col2', 'col3', 'col4','col5','col6','col7','col8','col9','col10', 'label'], '10', '/project/Trails/internal/ml/model_selection/config.ini');

```
## PyTorch + PolarDB

## Singa + PostgreSQL

## Singa + PolarDB

# Reproduce the result
Document is at [here](https://github.com/NLGithubWP/Trails/blob/main/internal/ml/model_selection/documents/README.md)

