FROM python:3.10.1-slim

ARG USER_NAME=appuser
ARG USER_ID=1001

RUN groupadd -g $USER_ID $USER_NAME && \
    useradd -g $USER_ID -m -u $USER_ID -s /bin/bash $USER_NAME && \
    chgrp -R 0 /home/$USER_NAME && \
    chmod -R g=u /home/$USER_NAME

WORKDIR /app
COPY src/Streamlit.py /app/src/Streamlit.py

COPY src/.streamlit /app/src/.streamlit

COPY --chown=$USER_ID:$USER_ID docker/sl_requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade -r sl_requirements.txt

COPY --chown=$USER_ID:$USER_ID . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "src/Streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]