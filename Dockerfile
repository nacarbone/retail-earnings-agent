FROM continuumio/miniconda3

# set a directory for the app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . .

# install system dependencies
RUN apt-get update
RUN apt-get install -y build-essential
RUN apt-get -y install cmake
RUN apt-get install zlib1g-dev
RUN apt-get install libz-dev
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install curl
RUN apt-get install zip unzip
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip -o awscliv2.zip
RUN ./aws/install
RUN rm -r awscliv2.zip
RUN rm -r aws
    
# download model object from S3
RUN aws --no-sign-request --region=us-west-2 s3 cp s3://stats404-ncarbone-final-project-bucket/model.py model.py
RUN aws --no-sign-request --region=us-west-2 s3 cp s3://stats404-ncarbone-final-project-bucket/config.json config.json
RUN mkdir checkpoint
RUN aws --no-sign-request --region=us-west-2 s3 cp s3://stats404-ncarbone-final-project-bucket/checkpoint/.is_checkpoint checkpoint/.is_checkpoint
RUN aws --no-sign-request --region=us-west-2 s3 cp s3://stats404-ncarbone-final-project-bucket/checkpoint/checkpoint checkpoint/checkpoint
RUN aws --no-sign-request --region=us-west-2 s3 cp s3://stats404-ncarbone-final-project-bucket/checkpoint/checkpoint.tune_metadata checkpoint/checkpoint.tune_metadata

# setup conda env
ADD environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml
# Note: env name = prod-env
RUN echo "source activate prod-env" > ~/.bashrc
ENV PATH /opt/conda/envs/prod-env/bin:$PATH
SHELL ["conda", "run", "-n", "prod-env", "/bin/bash", "-c"]

# install other python dependencies
RUN yes | pip install 'ray[rllib]' \
 && yes | pip install -U https://ray-wheels.s3-us-west-2.amazonaws.com/master/9dc671ae026db94b820ef177dc7c3b8bc3022ab3/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl \
 && yes | pip install -e market-env \
 && yes | pip install -e ppo-earnings-trader

# define the port number the container should expose
EXPOSE 5000
