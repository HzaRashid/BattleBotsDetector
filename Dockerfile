FROM python:3.12

# Install GCC build dependency
RUN apt-get update && \
    apt-get install -y gcc g++

# Install necessary Python packages
RUN pip install requests
RUN pip install pydantic
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install -U sentence-transformers
RUN pip install hdbscan
RUN pip install bertopic


# Copy source files (ensuring run.sh is available)
COPY . . 

CMD ["sh", "run.sh"]
