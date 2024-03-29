#include "../../include/comm/Channel.h"
#include "../../include/comm/Direct.h"
#include <cstring>

std::shared_ptr<FMI::Comm::Channel> FMI::Comm::Channel::get_channel(std::string name, std::map<std::string, std::string> params,
                                                                    std::map<std::string, std::string> model_params) {
    if (name == "Direct") {
        return std::make_shared<Direct>(params, model_params);
    } else {
        throw std::runtime_error("Unknown channel name passed");
    }
}

void FMI::Comm::Channel::gather(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root) {
    if (peer_id != root) {
        send(sendbuf, root);
    } else {
        auto buffer_length = sendbuf.len;
        for (int i = 0; i < num_peers; i++) {
            if (i == root) {
                std::memcpy(recvbuf.buf + root * buffer_length, sendbuf.buf, buffer_length);
            } else {
                channel_data peer_data {recvbuf.buf + i * buffer_length, buffer_length};
                recv(peer_data, i);
            }
        }
    }
}

void FMI::Comm::Channel::scatter(channel_data sendbuf, channel_data recvbuf, FMI::Utils::peer_num root) {
    if (peer_id == root) {
        auto buffer_length = recvbuf.len;
        for (int i = 0; i < num_peers; i++) {
            if (i == root) {
                std::memcpy(recvbuf.buf, sendbuf.buf + root * buffer_length, buffer_length);
            } else {
                channel_data peer_data {sendbuf.buf + i * buffer_length, buffer_length};
                send(peer_data, i);
            }
        }
    } else {
        recv(recvbuf, root);
    }
}

void FMI::Comm::Channel::allreduce(channel_data sendbuf, channel_data recvbuf, raw_function f) {
    reduce(sendbuf, recvbuf, 0, f);
    bcast(recvbuf, 0);
}
