ARG UBUNTU_VERSION=jammy

FROM ubuntu:$UBUNTU_VERSION AS build

# Install build tools
RUN apt update && apt install -y git build-essential cmake wget

# Install Vulkan SDK and cURL
RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - && \
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list && \
    apt update -y && \
    apt-get install -y vulkan-sdk libcurl4-openssl-dev curl

RUN apt-get install libxrandr-dev libxinerama-dev libxcursor-dev mesa-common-dev libx11-xcb-dev pkg-config ninja-build libxi-dev python-is-python3 -y
RUN git clone https://github.com/google/dawn && cd dawn
WORKDIR dawn
RUN cmake -S . -B out -DDAWN_FETCH_DEPENDENCIES=ON -DDAWN_ENABLE_INSTALL=ON -DTINT_BUILD_TESTS=OFF
RUN cmake --build out --parallel 4
RUN cmake --install out --prefix install

FROM ubuntu:$UBUNTU_VERSION
COPY --from=build /dawn/install /
