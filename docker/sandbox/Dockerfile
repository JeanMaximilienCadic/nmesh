FROM cadic/nmesh:vanilla

# Add wheels
COPY dist/*.whl /tmp/
RUN pip install /tmp/*.whl && \
    rm /tmp/*.whl && \
    python -m nmesh.tests