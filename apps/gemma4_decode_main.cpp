#include "strix/device/hip_gate.hpp"
#include "gemma4/engine/runtime.hpp"
#include "gemma4/ref/npz.hpp"
#include "gemma4_prefill_ab.hpp"
#include "bitcalc_measurement.hpp"
#include "strix/env_flags.hpp"
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
#include "gemma4/engine/pair.hpp"
#include "strix/pair/link.hpp"
#endif

#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>

namespace e = strix::gemma4::engine;
namespace g4 = strix::gemma4;
using Json = strix::serve::Json;
namespace {
Json ids_json(std::span<const std::uint32_t> ids) { auto j = Json::array(); for (auto id : ids) j.push(std::uint64_t(id)); return j; }
// Token IDs separated by whitespace and/or commas (the option_qual case files use commas).
std::vector<std::uint32_t> read_ids(const std::string& path) {
    auto text = e::read_text(path);
    for (auto& ch : text) if (ch == ',') ch = ' ';
    std::istringstream input(text); std::string word; std::vector<std::uint32_t> ids;
    while (input >> word) {
        if (word.empty() || word[0] == '-') throw std::invalid_argument("invalid token ID");
        std::size_t end = 0; const auto value = std::stoull(word,&end);
        if (end != word.size() || value > UINT32_MAX) throw std::invalid_argument("invalid token ID");
        ids.push_back(std::uint32_t(value));
    }
    return ids;
}
void emit(const e::Options& options, const e::Runtime& runtime, std::span<const std::uint32_t> ids, Json result) {
    if (options.output == "ids") { for (std::size_t i = 0; i < ids.size(); ++i) std::cout << (i ? " " : "") << ids[i]; std::cout << '\n'; }
    else if (options.output == "text") {
        if (!runtime.tokenizer) throw std::invalid_argument("text output requires --tokenizer");
        std::cout << runtime.tokenizer->decode(ids) << '\n';
    } else {
        result.set("ids",ids_json(ids));
        if (runtime.tokenizer) result.set("text",runtime.tokenizer->decode(ids));
        std::cout << result.dump() << std::endl;
    }
}
}
int main(int argc, char** argv) {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    for (int i = 1; i < argc; ++i) if (std::string_view(argv[i]) == "--ab-switch") return prefill_ab::run(argc, argv);
    try {
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
        // STRIX_G4_PAIR=1: --rank/--peer/--transport belong to the two-node route, not the option parser.
        std::vector<char*> pair_arguments;
        if (e::pair_enabled()) pair_arguments = strix::pair::strip_flags(argc,argv,"STRIX_G4_PAIR");
#endif
        const auto options = e::parse_options(argc,argv);
        if (options.help) {
            std::cout << "strix-gemma4-decode MODEL --rope-tables TABLE_DIR [--ids IDS.txt | --prompt TEXT | --chat REQUEST.json | --fixture FIXTURE.npz] [--tokenizer tokenizer.json]\n"
                "[--max-tokens 256] [--streams 1..8] [--dflash on|off|auto] [--draft-mtp DIR | --draft DRAFT.gguf --draft-rope TABLE.g4dr | --draft-eagle FILE] [--draft-k 0..7]\n"
                "[--temperature 0 --top-p 1 --top-k 0 --seed 0] [--output json|ids|text] [--threads 1..16] [--backend host|device|intel] [--ignore-eos] [--print-effective-config]\n"
                "drafters: --draft-mtp DIR (Gemma-4-E4B-it MTP assistant: a drafter-mtp/ package or a model dir containing one; recommended),\n"
                "  --draft DRAFT.gguf --draft-rope TABLE.g4dr (DFlash), --draft-eagle EAGLE.gguf (e4b-eagle fallback)\n"
                "env: STRIX_G4_DRAFT_KIND=dflash|mtp|eagle (how --draft is read), STRIX_G4_MTP_TOPK=1..64 (clusters, default 64),\n"
                "  STRIX_G4_MTP_Q8=1 (Q8_0 assistant matrices), STRIX_G4_MTP_GRAPHS=0 (no HIP graph replay), STRIX_G4_DFLASH=0 (no drafting)\n"
                "profiles: --profile exact|fast|turbo|quality (default: exact; explicit options and env win); the MTP drafter is found at\n"
                "  <model dir>/drafter-mtp unless --draft-mtp DIR is given; --draft off or STRIX_G4_DRAFT=0 disables it\n";
            return 0;
        }
        e::Runtime runtime(options);
        if (options.inspect) {
            // Inspection allocates no stream/cache and runs no target work.
            auto config = e::effective_config(*runtime.backend,options.request,bool(runtime.drafts),options.streams);
            std::cout << config.dump() << '\n'; return 0;
        }
        if (!options.chat.empty()) {
            if (!runtime.tokenizer || options.streams != 1) throw std::invalid_argument("chat requires --tokenizer and --streams 1");
            auto request = e::parse_chat_request(strix::serve::parse_json_unique(e::read_text(options.chat)),options.request.dflash,options.request);
            const g4::serve::Adapter adapter(runtime.tokenizer);
            const auto chat = e::run_chat(*runtime.backend,adapter,std::move(request),runtime.drafts);
            Json result(Json::Object{{"content",chat.output.content},{"reasoning_content",chat.output.reasoning},{"tool_calls",chat.output.tool_calls},{"finish_reason",chat.finish}});
            result.set("budget_continued",chat.budget_continued);
            auto receipts = Json::array(); for (const auto& r : chat.receipts) receipts.push(r); result.set("receipts",receipts);
            emit(options,runtime,chat.ids,result); return 0;
        }
        std::vector<std::vector<std::uint32_t>> prompts, expected;
        for (const auto& path : options.fixtures) {
            const auto npz = g4::read_npz(path); std::vector<std::uint32_t> prompt, gold;
            const auto prompt_ids = npz.at("prompt_ids").values<std::int32_t>("<i4"), gen_ids = npz.at("gen_ids").values<std::int32_t>("<i4");
            for (auto id : prompt_ids) { if (id < 0) throw std::invalid_argument("negative fixture id"); prompt.push_back(std::uint32_t(id)); }
            for (auto id : gen_ids) { if (id < 0) throw std::invalid_argument("negative fixture id"); gold.push_back(std::uint32_t(id)); }
            if (gold.size() < options.request.max_tokens) throw std::invalid_argument("fixture generation prefix unavailable");
            gold.resize(options.request.max_tokens); prompts.push_back(std::move(prompt)); expected.push_back(std::move(gold));
        }
        // Quality-kit capture (shared apps/bitcalc_measurement.hpp): STRIX_BITCALC_FORCE_TOKENS / STRIX_BITCALC_LOGITS_OUT
        // for one prompt, or STRIX_BITCALC_CASES=FILE ("FORCE OUT PROMPT_IDS" per line) for a list served by one model load.
        std::vector<strix::decode_cli::BitcalcCase> bitcalc_cases;
        if (const char* list = strix::envcfg::text("STRIX_BITCALC_CASES"); list && *list) {
            if (!prompts.empty() || !options.ids.empty() || !options.prompt.empty())
                throw std::invalid_argument("STRIX_BITCALC_CASES names the prompts itself (no --ids/--prompt/--fixture)");
            bitcalc_cases = strix::decode_cli::read_bitcalc_cases(list);
            for (const auto& c : bitcalc_cases) {
                if (c.prompt.empty()) throw std::invalid_argument("STRIX_BITCALC_CASES lines need FORCE OUT PROMPT_IDS");
                prompts.push_back(read_ids(c.prompt));
            }
        }
        if (prompts.empty()) {
            std::vector<std::uint32_t> prompt;
            if (!options.ids.empty()) prompt = read_ids(options.ids);
            else if (!options.prompt.empty() && runtime.tokenizer) prompt = runtime.tokenizer->encode(options.prompt);
            else throw std::invalid_argument("provide --ids, --prompt with tokenizer, --chat or --fixture");
            prompts.push_back(std::move(prompt));
        }
        const bool bitcalc_env = !bitcalc_cases.empty() || strix::decode_cli::BitcalcMeasurement().enabled();
        if (bitcalc_env && (options.streams != 1 || (bitcalc_cases.empty() && prompts.size() != 1)))
            throw std::invalid_argument("STRIX_BITCALC capture runs one stream (one prompt, or STRIX_BITCALC_CASES for a list)");
#ifdef STRIX_GEMMA4_DEVICE_BACKEND
        if (e::pair_enabled()) {
            if (bitcalc_env) throw std::invalid_argument("STRIX_BITCALC capture runs on the single-device route");
            if (prompts.size() != 1 || !expected.empty()) throw std::invalid_argument("pair route runs one --ids/--prompt request");
            return e::run_pair(options,runtime,prompts.front(),[&](std::span<const std::uint32_t> ids, Json receipt) {
                emit(options,runtime,ids,std::move(receipt));
            });
        }
#endif
        bool matched = true, failed = false;
        for (std::size_t p = 0; p < prompts.size(); ++p) {
            e::Coordinator coordinator(*runtime.backend,runtime.drafts);
            std::optional<strix::decode_cli::BitcalcMeasurement> bitcalc;
            if (!bitcalc_cases.empty()) bitcalc.emplace(bitcalc_cases[p].out.c_str(),bitcalc_cases[p].force.c_str());
            else if (bitcalc_env) bitcalc.emplace();
            std::vector<std::size_t> slots;
            for (std::uint32_t s = 0; s < options.streams; ++s) {
                auto r = options.request; r.prompt = prompts[p];
                if (bitcalc) {
                    bitcalc->prepare(runtime.backend->config().vocab,r.prompt.size(),r.max_tokens);
                    r.emission_hook = bitcalc->hook();
                }
                slots.push_back(coordinator.submit(std::move(r)));
            }
            coordinator.run();
            if (bitcalc) bitcalc->finish();
            for (const auto slot : slots) {
                const auto& result = coordinator.result(slot);
                failed |= result.status == e::Status::failed;
                auto receipt = coordinator.receipt(slot); receipt.set("stream",std::uint64_t(slot));
                if (bitcalc) {
                    receipt.set("prompt_index",std::uint64_t(p));
                    receipt.set("bitcalc_forced_steps",std::uint64_t(bitcalc->forced_steps()));
                    receipt.set("bitcalc_force_sha256",bitcalc->force_tokens_sha256());
                }
                if (!expected.empty()) { const bool equal = result.ids == expected[p]; receipt.set("reference_equal",equal); matched &= equal; }
                emit(options,runtime,result.ids,receipt);
            }
        }
        return failed ? 1 : matched ? 0 : 2;
    } catch (const std::exception& error) { std::cerr << "gemma4 decode: " << error.what() << '\n'; return 1; }
}
