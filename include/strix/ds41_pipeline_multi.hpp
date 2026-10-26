#pragma once

#include "strix/ds41_pipeline_wire.hpp"

#include <cstdint>
#include <span>
#include <vector>

// strix/ds41-pipeline-multi-v1: a pipeline FRONT that holds several sequences (slots) and a TAIL that drives it, so the
// FRONT computes one request's layers [0, K) while the TAIL computes another's [K, n) (several users of one split
// decode). The framing, the HELLO checks, the cut and the STEP payload are those of strix/ds41-pipeline-v1
// (ds41_pipeline_wire.hpp); only the direction of the requests changes after the handshake.
//
// Session: the FRONT connects and sends MULTI_HELLO (a HELLO plus its slot count and rows per step); the TAIL answers
// HELLO_ACK (status as for HELLO). From then on the TAIL sends requests and the FRONT answers every request with exactly
// one message, in the order the requests were sent (the TAIL may send several before reading the answers: a pipeline):
//   SEQ_STEP {slot, position, rows, commit, tokens}  -> SEQ_BOUNDARY (header flags = slot; payload = a STEP payload:
//                                                      the boundary of those rows after layer K - 1, flags 0)
//   SEQ_RESET {slot}                                 -> ACK {slot}
//   SEQ_IMPORT {slot, STATE payload}                 -> ACK {slot}  (the FRONT keeps the state's layers [0, K))
//   CLOSE                                            -> ACK
// commit (a verify window, DSpark): the rows of the slot's previous step that the TAIL kept (1..its rows), applied by the
// FRONT before this step (CudaPrefill::commit_window(commit, tokens[0])); 0 = the previous step is kept whole (or none).
// A FRONT that cannot run a request answers ERROR (text) and the session ends. Each slot's sequence is independent: the
// FRONT runs a slot's rows exactly as it would run that request alone, so every row's bits do not depend on the other
// slots (batch invariance), and the TAIL's batch step keeps that property.
namespace strix::ds41::pipe {

inline constexpr const char* kMultiProtocol = "strix/ds41-pipeline-multi-v1";

struct MultiHello {
    Hello hello;
    std::uint32_t slots = 0, max_rows = 0;
};
struct SeqStep {
    std::uint32_t slot = 0, position = 0, commit = 0;
    std::vector<std::uint32_t> tokens;  // the rows (1..max_rows)
};

[[nodiscard]] std::vector<std::uint8_t> encode(const MultiHello& m);
[[nodiscard]] MultiHello decode_multi_hello(std::span<const std::uint8_t> p);
[[nodiscard]] std::vector<std::uint8_t> encode(const SeqStep& m);
[[nodiscard]] SeqStep decode_seq_step(std::span<const std::uint8_t> p, std::uint32_t max_rows);
// SEQ_IMPORT: the slot, then a STATE payload (manifest + attention state bytes).
[[nodiscard]] std::vector<std::uint8_t> encode_seq_import(std::uint32_t slot, const StateMsg& state);
[[nodiscard]] std::pair<std::uint32_t, StateMsg> decode_seq_import(std::span<const std::uint8_t> p);

}  // namespace strix::ds41::pipe
