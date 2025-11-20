FROM encoderfile:latest AS build

# install uv (required for huggingface_hub only, not for encoderfile)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /encoderfile

# install huggingface cli
RUN uv venv && uv pip install huggingface_hub

# download model
COPY download_model.sh .
RUN sh download_model.sh

# encoderfile
COPY encoderfile.yml transform.lua ./

RUN encoderfile build -f encoderfile.yml

# final image — binary only
FROM debian:bookworm-slim AS final

# copy binary
COPY --from=build /encoderfile/model.encoderfile /usr/local/bin/model.encoderfile

ENTRYPOINT ["/usr/local/bin/model.encoderfile"]

CMD ["serve"]
