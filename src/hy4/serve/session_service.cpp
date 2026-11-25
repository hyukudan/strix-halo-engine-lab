#include "hy4/serve/session_service.hpp"
#include "strix/profiles/hy4.hpp"
#include "strix/gguf_reader.hpp"
#include "strix/model_family_registry.hpp"
#include "strix/weight/tensor_map.hpp"
#include <algorithm>
#include <charconv>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace strix::hy4::serve {
namespace {

template <class T>
T number(const std::string& flag, const std::string& text) {
    T value{};
    const auto* end = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), end, value);
    if (text.empty() || ec != std::errc{} || ptr != end) throw std::invalid_argument(flag + " needs a non-negative integer, got '" + text + "'");
    return value;
}

std::string origin(bool given, const char* otherwise) { return given ? "cli" : otherwise; }

sv::Json entry(sv::Json value, std::string from) {
    auto e = sv::Json::object();
    e.set("value", std::move(value));
    e.set("origin", sv::Json(std::move(from)));
    return e;
}

sv::Json strings(const std::vector<std::string>& values) {
    auto a = sv::Json::array();
    for (const auto& v : values) a.push(sv::Json(v));
    return a;
}

std::vector<std::string> thinking_off(const sv::Hy4ChatDialect& dialect) {
    // The template closes the reasoning block at once for no_think (render_hy4_chat); every other effort opens it.
    std::vector<std::string> off;
    if (std::find(dialect.efforts.begin(), dialect.efforts.end(), "no_think") != dialect.efforts.end()) off.emplace_back("no_think");
    return off;
}

struct Prepared final : sv::ServicePreparation {
    Options options;
    PackFacts pack;
};

const Prepared& prepared(const sv::ServiceSelection& selection) {
    const auto p = std::dynamic_pointer_cast<const Prepared>(selection.preparation);
    if (!p) throw std::invalid_argument("hy4_flash service has not passed validation");
    return *p;
}

}  // namespace

Options parse_options(std::span<const std::string> arguments) {
    Options o;
    std::vector<std::string> args(arguments.begin(), arguments.end());
    ::strix::profile::take_options(profile::engine(), args, o.profile);
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto& a = args[i];
        const auto value = [&]() -> const std::string& {
            if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
            return args[++i];
        };
        if (a == "--gguf") o.gguf = value();
        else if (a == "--ctx") { o.ctx = number<std::uint64_t>(a, value()); o.ctx_given = true; }
        else if (a == "--max-tokens") {
            o.max_tokens = number<std::uint32_t>(a, value());
            if (o.max_tokens == 0) throw std::invalid_argument("--max-tokens must be positive");
            o.max_tokens_given = true;
        } else if (a == "--model-id") {
            o.model_id = value();
            if (o.model_id.empty()) throw std::invalid_argument("--model-id must not be empty");
            o.model_id_given = true;
        } else if (a == "--print-effective-config") o.print_effective_config = true;
        else if (a == "--prompt-rows") (void)number<std::uint32_t>(a, value());
        else if (a == "--head-q8") (void)value();
        else if (a == "--help" || a == "-h") o.help = true;
        else throw std::invalid_argument("unknown hy4_flash option: " + a + " (see --help)");
    }
    return o;
}

PackFacts read_pack_facts(const Options& options) {
    PackFacts facts;
    if (options.gguf.empty()) return facts;
    const auto info = gguf::read_file(options.gguf);
    const auto* arch = info.find_metadata("general.architecture");
    if (!arch || arch->string_value.value_or("") != kArchitecture)
        throw std::invalid_argument(options.gguf + ": not a " + std::string(kArchitecture) + " pack");
    const auto* tmpl = info.find_metadata("tokenizer.chat_template");
    if (!tmpl || !tmpl->string_value) throw std::invalid_argument(options.gguf + ": the pack carries no tokenizer.chat_template");
    facts.dialect = sv::hy4_chat_dialect(*tmpl->string_value);
    facts.dialect_from_pack = true;
    if (const auto* eos = info.find_metadata("tokenizer.ggml.eos_token_id"); eos && eos->unsigned_value &&
        *eos->unsigned_value != sv::hy4_stop_tokens[0])
        throw std::invalid_argument(options.gguf + ": eos_token_id " + std::to_string(*eos->unsigned_value) + " differs from the template's 120025");
    if (const auto* ctx = info.find_metadata(std::string(kArchitecture) + ".context_length"); ctx && ctx->unsigned_value)
        facts.context_length = *ctx->unsigned_value;
    if (const auto* t = info.find_metadata("general.sampling.temp"); t && t->floating_value) {
        facts.card_temperature = *t->floating_value;
        facts.card_from_pack = true;
    }
    if (const auto* p = info.find_metadata("general.sampling.top_p"); p && p->floating_value) {
        facts.card_top_p = *p->floating_value;
        facts.card_from_pack = true;
    }
    if (const auto* k = info.find_metadata("general.sampling.top_k"); k && (k->signed_value || k->unsigned_value)) {
        facts.card_top_k = k->signed_value ? *k->signed_value : static_cast<std::int64_t>(*k->unsigned_value);
        facts.card_from_pack = true;
    }
    if (options.ctx_given && facts.context_length && options.ctx > facts.context_length)
        throw std::invalid_argument("--ctx " + std::to_string(options.ctx) + " exceeds the pack's context_length " + std::to_string(facts.context_length));
    return facts;
}

sv::Json effective_config(const Options& options, const PackFacts& pack) {
    auto values = sv::Json::object();
    values.set("gguf", entry(options.gguf.empty() ? sv::Json() : sv::Json(options.gguf), options.gguf.empty() ? "unset" : "cli"));
    const std::uint64_t ctx = options.ctx_given && options.ctx ? options.ctx : pack.context_length;
    values.set("ctx", entry(sv::Json(ctx), options.ctx_given && options.ctx ? "cli" : pack.context_length ? "pack" : "unresolved"));
    values.set("max_tokens", entry(sv::Json(std::uint64_t{options.max_tokens}), origin(options.max_tokens_given, "builtin")));
    values.set("model_id", entry(sv::Json(options.model_id), origin(options.model_id_given, "builtin")));
    auto dialect = sv::Json::object();
    dialect.set("suffix", sv::Json(pack.dialect.suffix));
    dialect.set("efforts", strings(pack.dialect.efforts));
    values.set("chat_dialect", entry(std::move(dialect), pack.dialect_from_pack ? "pack" : "builtin (official release)"));

    auto request = sv::Json::object();
    request.set("reasoning_effort", sv::Json("request value, else the template default high; none / enable_thinking false = no_think"));
    request.set("thinking_off_efforts", strings(thinking_off(pack.dialect)));
    auto stop = sv::Json::array();
    for (const auto id : sv::hy4_stop_tokens) stop.push(sv::Json(std::uint64_t{id}));
    request.set("stop_tokens", std::move(stop));
    request.set("tool_grammar", sv::Json("shared chat_kit key/value parser with the dialect's spellings (parse_hy4_output)"));
    auto sampling = sv::Json::object();
    sampling.set("default", sv::Json("greedy (exact)"));
    sampling.set("policy", sv::Json("greedy"));
    sampling.set("engine_samples", sv::Json(false));
    auto card = sv::Json::object();
    card.set("temperature", sv::Json(pack.card_temperature));
    card.set("top_p", sv::Json(pack.card_top_p));
    card.set("top_k", sv::Json(pack.card_top_k));
    sampling.set("card", entry(std::move(card), pack.card_from_pack ? "pack general.sampling" : "builtin generation_config.json"));
    sampling.set("rule", sv::Json("the card values are the model's generation_config; they apply only when the client sends them and the engine "
                                  "samples; until then a request with temperature > 0 runs greedy and the response says so (shared api.cpp)"));
    request.set("sampling", std::move(sampling));

    auto serve_only = sv::Json::array();
    const auto lever = [&](const char* name, const char* reason) {
        auto l = sv::Json::object();
        l.set("lever", sv::Json(name));
        l.set("reason", sv::Json(reason));
        serve_only.push(std::move(l));
    };
    lever("bind/port/api_key", "strix-serve shell options (before --): listener and authentication only, no effect on tokens");
    lever("http_slots", "concurrent requests are a serve concept; the CLI runs one request (no device binding yet: unresolved)");

    auto out = sv::Json::object();
    out.set("family", sv::Json(kFamily));
    out.set("architecture", sv::Json(kArchitecture));
    out.set("values", std::move(values));
    out.set("request", std::move(request));
    out.set("serve_only", std::move(serve_only));
    const auto resolved = profile::resolve({profile::Device::unknown, profile::Topology::one_node, ctx}, options.profile);
    out.set("profile", sv::parse_json(resolved->resolution.json()));
    out.set("device", sv::Json("not built: hy4_hipbind / hy4_cudabind"));
    return out;
}

void bind_api(sv::ApiConfig& config, const Options& options, const PackFacts& pack) {
    config.model_id = options.model_id;
    config.default_max_tokens = options.max_tokens;
    config.default_thinking = true;  // the generation prompt opens the reasoning block (effort high)
    config.thinking_off_efforts = thinking_off(pack.dialect);
    config.sampling_policy = "greedy";
    const auto dialect = pack.dialect;
    config.chat_renderer = [dialect](const sv::ChatTemplateInput& in) { return sv::render_hy4_chat(in, dialect); };
    config.output_parser = [dialect](const std::string& text, bool thinking, const sv::Json& tools) {
        return sv::parse_hy4_output(text, thinking, tools, dialect);
    };
    config.tool_call_marker = sv::hy4_tool_marker;
    config.stop_tokens.assign(sv::hy4_stop_tokens.begin(), sv::hy4_stop_tokens.end());
}

const char* usage() {
    return "hy4_flash family options (strix-serve --model hy4_flash -- OPTIONS):\n"
           "  --gguf PATH               first part of the Hy4-flash pack (tokenizer, chat template, weights)\n"
           "  --ctx N                   context cap (default: the pack's context_length)\n"
           "  --max-tokens N            default reply cap (default 4096)\n"
           "  --model-id NAME           model id reported by the API (default hy4-flash)\n"
           "  --profile exact|fast|turbo  tier to resolve for host inspection\n"
           "  --prompt-rows N --head-q8 0|1  explicit profile overrides for inspection\n"
           "  --print-effective-config  print the effective values and the serve-only levers, then exit\n";
}

void install_session_service(sv::ServiceFactories& factories) {
    sv::ServiceFactory f;
    f.asset_option = "--gguf";
    f.validate = [](const sv::ServiceSelection& s) {
        auto p = std::make_shared<Prepared>();
        p->options = parse_options(s.arguments);
        p->pack = read_pack_facts(p->options);
        s.preparation = std::move(p);
    };
    f.early_exit = [](const sv::ServiceSelection& s, std::ostream& out) -> std::optional<int> {
        const auto& p = prepared(s);
        if (p.options.help) { out << usage(); return 0; }
        if (p.options.print_effective_config) { out << effective_config(p.options, p.pack).dump() << '\n'; return 0; }
        return std::nullopt;
    };
    f.inspect = [](const sv::ServiceSelection& s, std::ostream& out) {
        const auto& p = prepared(s);
        out << s.model.id << '\t' << s.model.name << '\t' << s.model.description << '\n' << effective_config(p.options, p.pack).dump() << '\n';
    };
    f.create = [](const sv::ServiceSelection& s) -> std::unique_ptr<sv::SessionService> {
        if (prepared(s).options.gguf.empty()) throw std::invalid_argument("hy4_flash serve requires --gguf PATH");
        throw std::runtime_error("hy4_flash: no device binding is built yet (hy4_hipbind / hy4_cudabind); --inspect, --family-inspect and "
                                 "--print-effective-config run without one");
    };
    factories.install(kFamily, std::move(f));
    register_family();  // a program serving hy4_flash admits its packs
}

void register_family() {
    const auto& map = weight::tensor_map::hy4_flash_map();
    const auto resolve = [&map](const gguf::FileInfo& info) {
        weight::tensor_map::Model model;
        std::string error;
        if (!map.model(info, model, error)) throw std::invalid_argument("hy4_flash: " + error);
        return model;
    };
    ModelFamily family;
    family.name = kFamily;
    family.architectures = {std::string(map.architecture)};
    family.validate = [resolve](const gguf::FileInfo& info) { (void)resolve(info); };
    family.describe = [resolve](const gguf::FileInfo& info) {
        using weight::tensor_map::Sym;
        const auto m = resolve(info);
        const auto v = [&m](Sym s) { return std::to_string(m.value[static_cast<std::size_t>(s)]); };
        return std::string(kFamily) + ": " + std::to_string(m.blocks) + " blocks, hidden " + v(Sym::hidden) + ", " + v(Sym::experts) +
               " experts, vocab " + v(Sym::vocab) + ", " + std::to_string(info.tensors.size()) + " tensors in this part";
    };
    family.reader_limits = [&map] {
        gguf::Limits limits;
        limits.max_retained_array_elements = map.retained_array_elements;
        return limits;
    };
    if (!model_family_registered(kFamily)) register_model_family(std::move(family));
}

}  // namespace strix::hy4::serve
