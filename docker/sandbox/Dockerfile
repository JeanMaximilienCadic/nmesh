FROM cadic/nmesh:vanilla

WORKDIR /tmp

# Add wheels
COPY dist/*.whl /tmp/
COPY Makefile /tmp/
COPY .env /tmp/

RUN pip install /tmp/*.whl && \
    rm /tmp/*.whl && \
    make tests