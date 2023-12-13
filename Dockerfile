FROM amazonlinux:latest

RUN yum -y groupinstall "Development tools"
RUN yum -y install gcc-c++ libcurl-devel cmake3 git wget vim openssl-devel blas-devel lapack-devel
RUN cd /opt && curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install && cd ..
RUN cd /opt && git clone https://github.com/awslabs/aws-lambda-cpp.git && cd aws-lambda-cpp && mkdir build && cd build && \
    cmake3 .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/opt/software -DCMAKE_CXX_COMPILER=g++ && make -j12 && make -j12 install
RUN curl -LO https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz && tar -xvf boost_1_77_0.tar.gz && \
    cd boost_1_77_0 && ./bootstrap.sh && ./b2 cxxflags="-fPIC" link=static install && cd .. && rm -rf boost*
RUN cd /opt && git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp.git
RUN cd /opt/aws-sdk-cpp && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/software -DBUILD_ONLY="s3" && \
    cmake --build . --config Release -- -j12 && cmake --install . --config Release
#RUN cd /opt/aws-sdk-cpp && mkdir build && cd build && \
    #cmake3 .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF && make -j12 install
