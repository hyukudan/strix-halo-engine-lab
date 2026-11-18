#pragma once
// Two buffers for a recurrent state that a speculative verify window updates (GDN: recurrent_rows_kernel loads the state
// once into registers and stores it once after the last row). The window READS committed() and WRITES window_target();
// nothing writes committed() until the window is resolved, so the committed buffer IS the checkpoint:
//   accept_all(): the window's final state becomes the committed one (role swap, 0 bytes moved);
//   a partial accept / discard keeps committed() = pre-window state; the caller replays the accepted rows IN PLACE on it.
// This replaces snapshot-before-verify + restore-on-reject (two full state copies). The arithmetic of the kernel is
// unchanged: only the store address of the window differs, so every committed state is bitwise the incumbent's.
// Per-layer slices map by offset: the same layout lives in both buffers. Home: shared (backend core, header only).
#include <cstddef>
#include <utility>

namespace strix::core {

template <class T> class WindowStateBuffers {
public:
    WindowStateBuffers() = default;
    WindowStateBuffers(T* committed, T* spare) noexcept : committed_(committed), spare_(spare) {}
    T* committed() const noexcept { return committed_; }
    T* window_target() const noexcept { return spare_; }
    // The element `offset` of the committed buffer, and the same element of the window's target.
    T* committed_at(std::size_t offset) const noexcept { return committed_ + offset; }
    T* target_at(std::size_t offset) const noexcept { return spare_ + offset; }
    void accept_all() noexcept { std::swap(committed_, spare_); }

private:
    T* committed_ = nullptr;
    T* spare_ = nullptr;
};

}  // namespace strix::core
