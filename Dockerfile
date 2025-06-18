FROM nvcr.io/nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get upgrade -y

# Install apt packages
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y unzip psmisc wget bc jq htop curl git git-lfs nano ssh gcc screen tmux -y --no-install-recommends && rm -rf /var/cache/*

# Install uv
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Install Visual Studio Code (for interactive tunnelling)
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz
RUN tar -xf vscode_cli.tar.gz

# copy above docker folder to /CueTip and set working directory
COPY .. /CueTip
WORKDIR /CueTip

# Install python libraries
RUN uv sync --all-extras --frozen

# Fix PoolTool install for missing font
RUN wget -O HackNerdFontMono-Regular.ttf https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/Hack/Regular/complete/Hack%20Nerd%20Font%20Mono%20Regular.ttf
RUN mv HackNerdFontMono-Regular.ttf /CueTip/.venv/lib/python3.10/site-packages/pooltool/ani/fonts/


CMD [ "sleep", "infinity" ]