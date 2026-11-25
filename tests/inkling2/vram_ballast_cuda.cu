// Device test driver: runs a command with the free device memory reduced to a stated amount.
//   strix-cuda-vram-ballast --leave-mib N -- COMMAND [ARGS ...]
// allocates (free - N MiB) of device memory, runs COMMAND as a child process and returns its status, then frees the
// ballast. The command is the unchanged production executable: a decoder load whose row workspaces no longer fit is
// refused by the real device allocation (the typed allocation failure) and the shared admission retries. This tool only
// takes memory away; production code has no switch for it.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv) {
    if (argc < 5 || std::strcmp(argv[1], "--leave-mib") != 0 || std::strcmp(argv[3], "--") != 0) {
        std::fprintf(stderr, "usage: %s --leave-mib N -- COMMAND [ARGS ...]\n", argv[0]);
        return 2;
    }
    const std::size_t leave = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) << 20U;
    std::size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemGetInfo failed\n");
        return 2;
    }
    void* ballast = nullptr;
    if (free_bytes > leave && cudaMalloc(&ballast, free_bytes - leave) != cudaSuccess) {
        std::fprintf(stderr, "ballast allocation failed\n");
        return 2;
    }
    std::fprintf(stderr, "vram-ballast free %zu MiB -> left %zu MiB\n", free_bytes >> 20U, leave >> 20U);
    std::string command;
    for (int i = 4; i < argc; ++i) command += (i > 4 ? " \"" : "\"") + std::string(argv[i]) + "\"";
#if defined(_WIN32)
    command = "\"" + command + "\"";  // cmd.exe strips one outer quote pair
#endif
    const int status = std::system(command.c_str());
    if (ballast) cudaFree(ballast);
    return status;
}
