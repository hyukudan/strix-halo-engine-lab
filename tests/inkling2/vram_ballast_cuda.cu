// Device test driver: runs a command with the free device memory reduced to a stated amount.
//   strix-cuda-vram-ballast --leave-mib N -- COMMAND [ARGS ...]
// allocates (free - N MiB) of device memory, runs COMMAND as a child process and returns its status, then frees the
// ballast. The command is the unchanged production executable: a decoder load whose row workspaces no longer fit is
// refused by the real device allocation (the typed allocation failure) and the shared admission retries. This tool only
// takes memory away; production code has no switch for it.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

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
    std::fflush(stderr);
    // The child starts without a shell: a command interpreter caps the command line (8191 characters on Windows), which
    // a long token prompt exceeds. Arguments must not contain spaces (the Windows runtime joins them with spaces).
    int status = 2;
#if defined(_WIN32)
    const intptr_t code = _spawnv(_P_WAIT, argv[4], argv + 4);
    if (code == -1) std::fprintf(stderr, "vram-ballast: cannot start %s\n", argv[4]);
    else status = static_cast<int>(code);
#else
    pid_t pid = 0;
    if (posix_spawn(&pid, argv[4], nullptr, nullptr, argv + 4, environ) != 0) {
        std::fprintf(stderr, "vram-ballast: cannot start %s\n", argv[4]);
    } else {
        int wait_status = 0;
        if (waitpid(pid, &wait_status, 0) == pid) status = WIFEXITED(wait_status) ? WEXITSTATUS(wait_status) : 128;
    }
#endif
    if (ballast) cudaFree(ballast);
    return status;
}
