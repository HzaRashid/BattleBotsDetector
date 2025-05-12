FROM python:3.12

# Install GCC build dependency
RUN apt-get update && \
    apt-get install -y gcc g++

# Install necessary Python packages
RUN pip install requests
RUN pip install pydantic

RUN pip install pandas
RUN pip install polars
RUN pip install scikit-learn
RUN pip install -U sentence-transformers
RUN pip install protobuf
RUN pip install tiktoken
RUN pip install sentencepiece
RUN pip install --upgrade huggingface_hub
RUN pip install ijson
RUN pip install emoji
# https://github.com/huggingface/transformers/issues/37311
# https://github.com/opendatalab/MinerU/issues/2112
RUN pip install transformers==4.51.0
RUN pip install --upgrade pip && \
    pip install \
        torch>=2.5.1 \
        torchvision \
        optuna \
        numpy \
        pillow \
        accelerate==1.6.0



# Copy source files (ensuring run.sh is available)
COPY . . 

CMD ["sh", "run.sh"]