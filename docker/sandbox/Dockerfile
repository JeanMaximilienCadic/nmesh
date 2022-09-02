FROM cadic/nmesh:vanilla


# Add wheels
COPY dist/*.whl /tmp/
RUN pip install /tmp/*.whl

# #Foo
# WORKDIR /opt/nmesh
# RUN ln -s /opt/conda/envs/ray/lib/python3.8/site-packages/ai3ddst/config.yml /opt/ai3ddst/config.yml

# # Overwrite default when mounting workspace
# RUN echo 'export PYTHONPATH=$PYTHONPATH:"/workspace"' >> /home/$SETUSER/.bashrc

# WORKDIR /workspace

# CMD ["/bin/sh", "/entrypoint.sh"]
