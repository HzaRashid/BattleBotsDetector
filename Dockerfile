FROM python:3

RUN pip install requests
RUN pip install pydantic

RUN pip install pandas
RUN pip install scikit-learn
RUN pip install -U spacy
RUN python3 -m spacy download en_core_web_sm

#Important so we will have access to the run.sh file 
COPY . . 

CMD ["sh", "run.sh"]
