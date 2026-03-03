FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y

RUN apt-get install -y build-essential
RUN apt-get install -y jq

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir

# # Install any other dependencies
# COPY ./scripts/ /opt/ml/code/

# WORKDIR /opt/ml/code

# # # Setup the entry point
# ENTRYPOINT ["./run.sh"]