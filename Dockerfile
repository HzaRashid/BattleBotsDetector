FROM python:3.12

# Install GCC build dependency
RUN apt-get update && \
    apt-get install -y gcc g++

RUN pip install requests
RUN pip install pydantic

RUN pip install pandas
RUN pip install scikit-learn
# RUN pip install spacy==3.7.2
# RUN python3 -m spacy download en_core_web_sm

RUN pip install bertopic


#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
