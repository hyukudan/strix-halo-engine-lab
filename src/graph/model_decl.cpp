#include "strix/graph/model_decl.hpp"

#include "strix/weight/admission.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

namespace strix::graph {
namespace {
using model_description::Value;
using Int = std::uint64_t;
using V = std::string_view;
using T = DeclFields;
using A = Value::Array;
constexpr Int class_set = Int{1} << 63;

[[noreturn]] void fail(V where, V message) {
    throw std::invalid_argument("model declaration " + std::string(where) + ": " + std::string(message));
}

const Value* get(const T& t, V key) {
    const auto it = t.find(key);
    return it == t.end() ? nullptr : &it->second;
}

const Value* dotted(const T& t, V key) {
    if (const auto* v = get(t, key)) return v;
    const auto dot = key.find('.');
    if (dot == V::npos) return nullptr;
    const auto* root = get(t, key.substr(0, dot));
    const auto* child = root ? std::get_if<T>(&root->data) : nullptr;
    return child ? dotted(*child, key.substr(dot + 1)) : nullptr;
}

template<class Type>
const Type& as(const Value& v, V where, V expected) {
    if (const auto* p = std::get_if<Type>(&v.data)) return *p;
    fail(where, "expected " + std::string(expected));
}

V word(const T& t, V key, V fallback = {}) {
    const auto* v = get(t, key);
    return v ? as<std::string>(*v, key, "a string") : fallback;
}

V required(const T& t, V key, V where) {
    const auto result = word(t, key);
    if (result.empty()) fail(where, "missing " + std::string(key));
    return result;
}

const T& table(const T& t, V key) {
    static const T empty;
    const auto* v = get(t, key);
    return v ? as<T>(*v, key, "a table") : empty;
}

const A& array(const T& t, V key) {
    static const A empty;
    const auto* v = get(t, key);
    return v ? as<A>(*v, key, "an array") : empty;
}

std::vector<T> rows(const T& t, V key) {
    const auto* v = get(t, key);
    if (!v) return {};
    if (const auto* single = std::get_if<T>(&v->data)) return {*single};
    std::vector<T> result;
    for (const auto& row : as<A>(*v, key, "a table or array of tables"))
        result.push_back(as<T>(row, key, "a table"));
    return result;
}

std::vector<V> words(const T& t, V key) {
    const auto* v = get(t, key);
    if (!v) return {};
    if (const auto* single = std::get_if<std::string>(&v->data)) return {*single};
    std::vector<V> result;
    for (const auto& row : as<A>(*v, key, "a string or array of strings or ports")) {
        if (const auto* port = std::get_if<T>(&row.data)) result.push_back(required(*port, "id", key));
        else result.push_back(as<std::string>(row, key, "a string"));
    }
    return result;
}

Int integer(const Value& v, V where) {
    const auto n = as<std::int64_t>(v, where, "an unsigned integer");
    if (n < 0) fail(where, "expected an unsigned integer");
    return static_cast<Int>(n);
}

template<class U>
U narrow(Int n, V where) {
    if (n > std::numeric_limits<U>::max()) fail(where, "integer out of range");
    return static_cast<U>(n);
}

bool flag(const T& t, V key) {
    const auto* v = get(t, key);
    return v && as<bool>(*v, key, "a boolean");
}

void keys_only(const T& t, V keys, V where) {
    for (const auto& [key, ignored] : t)
        if (keys.find("|" + key + "|") == V::npos) fail(where, "unknown field " + key);
}

// The same grammar resolves dimensions, slice bounds and declarative guards.
// Both sides of a Boolean expression are parsed, including inactive arms.
class Expression {
public:
    using Resolve = std::function<Int(V)>;
    Expression(V s, V where, const Resolve& resolve, bool class_comparison = false)
        : text_(s), where_(where), resolve_(resolve), class_comparison_(class_comparison) {}
    Int read() {
        const auto n = either(0);
        if (!space().empty()) fail(where_, "unexpected expression token " + std::string(text_));
        return n;
    }

private:
    V text_, where_;
    const Resolve& resolve_;
    bool class_comparison_;
    static bool digit(char c) { return c >= '0' && c <= '9'; }
    static bool letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_'; }
    V space() {
        while (!text_.empty() && (text_.front() == ' ' || text_.front() == '\t')) text_.remove_prefix(1);
        return text_;
    }
    bool take(V token) {
        if (!space().starts_with(token)) return false;
        text_.remove_prefix(token.size());
        return true;
    }
    Int either(unsigned depth) {
        auto n = both(depth);
        while (take("||") || take("|")) { const auto rhs = both(depth); n = n != 0 || rhs != 0; }
        return n;
    }
    Int both(unsigned depth) {
        auto n = comparison(depth);
        while (take("&&") || take("&")) { const auto rhs = comparison(depth); n = n != 0 && rhs != 0; }
        return n;
    }
    Int comparison(unsigned depth) {
        const auto n = sum(depth);
        const auto equals = [&](Int rhs) {
            return class_comparison_ && (n & class_set) ? (n & rhs & ~class_set) != 0 : n == rhs;
        };
        if (take("==")) return equals(sum(depth));
        if (take("!=")) return !equals(sum(depth));
        if (take("<=")) return n <= sum(depth);
        if (take(">=")) return n >= sum(depth);
        if (take("<")) return n < sum(depth);
        if (take(">")) return n > sum(depth);
        return n;
    }
    Int sum(unsigned depth) {
        auto n = product(depth);
        for (;;) {
            if (take("+")) {
                const auto rhs = product(depth);
                if (rhs > std::numeric_limits<Int>::max() - n) fail(where_, "overflow in expression");
                n += rhs;
            } else if (take("-")) {
                const auto rhs = product(depth);
                if (rhs > n) fail(where_, "negative expression");
                n -= rhs;
            } else return n;
        }
    }
    Int product(unsigned depth) {
        auto n = atom(depth);
        for (;;) {
            if (take("*")) {
                const auto rhs = atom(depth);
                if (rhs && n > std::numeric_limits<Int>::max() / rhs) fail(where_, "overflow in expression");
                n *= rhs;
            } else {
                const bool divide = take("/");
                if (!divide && !take("%")) return n;
                const auto rhs = atom(depth);
                if (!rhs) fail(where_, "zero divisor");
                n = divide ? n / rhs : n % rhs;
            }
        }
    }
    Int atom(unsigned depth) {
        if (depth > 32) fail(where_, "expression nesting limit");
        if (take("!")) return !atom(depth + 1);
        if (take("(")) {
            const auto n = either(depth + 1);
            if (!take(")")) fail(where_, "unclosed expression");
            return n;
        }
        if (space().empty()) fail(where_, "expected expression operand");
        if (digit(text_.front())) {
            Int n{};
            const auto [end, error] = std::from_chars(text_.data(), text_.data() + text_.size(), n);
            if (error != std::errc{}) fail(where_, "integer out of range");
            text_.remove_prefix(static_cast<std::size_t>(end - text_.data()));
            return n;
        }
        if (!letter(text_.front())) fail(where_, "expected expression operand");
        const auto end = std::find_if_not(text_.begin(), text_.end(), [](char c) { return letter(c) || digit(c); });
        const auto size = static_cast<std::size_t>(end - text_.begin());
        const auto name = text_.substr(0, size);
        text_.remove_prefix(size);
        return resolve_(name);
    }
};

std::uint32_t decimal(V text, V where) {
    if (text.empty()) fail(where, "expected a decimal layer ordinal");
    std::uint32_t number{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), number);
    if (error != std::errc{} || end != text.data() + text.size()) fail(where, "expected a decimal layer ordinal");
    return number;
}

V trim(V text) {
    const auto a = text.find_first_not_of(" \t");
    return a == V::npos ? V{} : text.substr(a, text.find_last_not_of(" \t") - a + 1);
}

std::pair<std::uint32_t, std::uint32_t> range(V text, std::uint32_t count, V where) {
    const auto dots = text.find("..");
    const auto first = decimal(trim(text.substr(0, dots)), where);
    const auto last = dots == V::npos ? first : decimal(trim(text.substr(dots + 2)), where);
    if (first > last || last >= count) fail(where, "layer range outside trunk");
    return {first, last};
}

std::vector<std::uint32_t> layer_list(const Value& value, std::uint32_t count, V where) {
    std::vector<std::uint32_t> result;
    if (const auto* text = std::get_if<std::string>(&value.data)) {
        const auto [first, last] = range(*text, count, where);
        for (auto i = first; i <= last; ++i) result.push_back(i);
    } else {
        for (const auto& row : as<A>(value, where, "a layer list")) {
            const auto i = integer(row, where);
            if (i >= count) fail(where, "layer outside trunk");
            result.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return result;
}

std::vector<LayerVisit> schedule(V order, std::uint32_t count) {
    std::vector<LayerVisit> result;
    std::vector<std::uint32_t> passes(count);
    const auto add = [&](std::uint32_t first, std::uint32_t last, std::uint32_t repeat) {
        if (!repeat || Int{repeat} * (last - first + 1U) + result.size() > 1'000'000)
            fail("schedule", "invalid repeat or too many visits");
        for (std::uint32_t p = 0; p < repeat; ++p)
            for (auto layer = first; layer <= last; ++layer)
                result.push_back({layer, passes[layer]++, static_cast<std::uint32_t>(result.size())});
    };
    if (order.empty()) { add(0, count - 1, 1); return result; }
    for (;;) {
        const auto comma = order.find(',');
        auto token = trim(order.substr(0, comma));
        std::uint32_t repeat = 1;
        if (token.starts_with('(')) {
            const auto close = token.find(")*");
            if (close == V::npos) fail("schedule", "expected (first..last)*passes");
            repeat = decimal(trim(token.substr(close + 2)), "schedule");
            token = token.substr(1, close - 1);
        }
        const auto [first, last] = range(token, count, "schedule");
        add(first, last, repeat);
        if (comma == V::npos) break;
        order.remove_prefix(comma + 1);
    }
    if (std::ranges::find(passes, 0U) != passes.end()) fail("schedule", "omits a physical layer");
    return result;
}

constexpr std::pair<V, Traits> class_bits[] = {
    {"full", trait::attn_dense}, {"gdn", trait::attn_gdn}, {"window", trait::attn_window},
    {"indexed", trait::attn_indexed}, {"dense", trait::ffn_dense}, {"moe", trait::ffn_moe},
    {"inject", trait::inject}
};

template<class E>
E named(V name, std::span<const std::pair<V, E>> names, V where) {
    for (const auto& [key, value] : names) if (key == name) return value;
    fail(where, "unknown key " + std::string(name));
}

constexpr std::pair<V, Op> ops[] = {
    {"linear", Op::linear}, {"attend", Op::attend}, {"select", Op::select}, {"route", Op::route},
    {"combine", Op::combine}, {"norm", Op::norm}, {"mix", Op::mix}, {"inject", Op::inject},
    {"head", Op::head}, {"elementwise", Op::elementwise}, {"store", Op::store}, {"copy", Op::copy},
    {"clear", Op::clear}, {"upload", Op::upload}, {"download", Op::download}, {"gather", Op::gather},
    {"argmax", Op::argmax}, {"custom", Op::elementwise}
};
constexpr std::pair<V, Split> splits[] = {
    {"none", Split::none}, {"replicate", Split::replicate}, {"by_heads", Split::by_heads},
    {"by_experts", Split::by_experts}, {"by_k", Split::by_k}, {"by_vocab", Split::by_vocab}
};
constexpr std::pair<V, Format> formats[] = {
    {"f32", Format::f32}, {"f16", Format::f16}, {"bf16", Format::bf16}, {"q8_0", Format::q8_0},
    {"q4_k", Format::q4_k}, {"q6_k", Format::q6_k}, {"iu4", Format::iu4}
};

Format legacy_format(V name) {
    for (const auto& [key, value] : formats) if (key == name) return value;
    return Format::none;
}

// Match a single-star rule against either the declared symbol or its expanded name.
bool matches(V pattern, V name) {
    if (pattern.empty()) return true;
    const auto star = pattern.find('*');
    if (star == V::npos) return pattern == name;
    const auto prefix = pattern.substr(0, star), suffix = pattern.substr(star + 1);
    return name.size() >= prefix.size() + suffix.size() && name.starts_with(prefix) && name.ends_with(suffix);
}

void merge(T& base, const T& overlay) {
    for (const auto& [key, value] : overlay) {
        auto it = base.find(key);
        const auto* children = std::get_if<T>(&value.data);
        if (it != base.end() && children && std::holds_alternative<T>(it->second.data))
            merge(std::get<T>(it->second.data), *children);
        else base.insert_or_assign(key, value);
    }
}

std::vector<T> overlay_rows(std::vector<T> base, const std::vector<T>& overlay) {
    std::set<V> seen;
    for (const auto& row : overlay) {
        const auto id = required(row, "id", "stage");
        if (!seen.insert(id).second) fail(id, "duplicate stage id");
        const auto find_id = [&](V target) {
            return std::ranges::find_if(base, [&](const T& t) { return word(t, "id") == target; });
        };
        auto found = find_id(id);
        const auto after = word(row, "after");
        const auto before = word(row, "before");
        if (!after.empty() && !before.empty()) fail(id, "multiple insertion anchors");
        if (found != base.end()) {
            if (!after.empty() || !before.empty()) fail(id, "cannot insert an existing stage");
            merge(*found, row);
        } else if (!after.empty() || !before.empty()) {
            const auto anchor = after.empty() ? before : after;
            found = find_id(anchor);
            if (found == base.end()) fail(id, "unknown insertion anchor " + std::string(anchor));
            base.insert(found + (after.empty() ? 0 : 1), row);
        } else base.push_back(row);
    }
    return base;
}

T template_fields(const ModelDeclOptions& options, V key, std::set<std::string, std::less<>>& seen) {
    if (key.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789/_-") != V::npos ||
        key.starts_with('/') || key.ends_with('/') || key.find("//") != V::npos)
        fail(key, "invalid template path");
    if (seen.size() >= 32 || !seen.emplace(key).second) fail(key, "template inheritance cycle or limit");
    auto t = model_description::load((options.templates / (std::string(key) + ".toml")).string(),
                                     model_description::TableSyntax::root_array_tables).fields;
    keys_only(t, "|extends|omit|stage|value|", key);
    const auto parent = word(t, "extends");
    if (parent.empty()) return t;
    auto inherited = template_fields(options, parent, seen);
    auto stages = rows(inherited, "stage");
    for (const auto omit : words(t, "omit")) {
        if (!std::erase_if(stages, [&](const T& row) { return word(row, "id") == omit; }))
            fail(key, "unknown omitted stage " + std::string(omit));
    }
    A merged;
    for (auto& row : overlay_rows(std::move(stages), rows(t, "stage"))) merged.push_back(Value{std::move(row)});
    inherited.insert_or_assign("stage", Value{std::move(merged)});
    if (const auto* values = get(t, "value")) inherited.insert_or_assign("value", *values);
    return inherited;
}

struct Fnv {
    Int hash = 14695981039346656037ULL;
    void byte(unsigned b) { hash = (hash ^ b) * 1099511628211ULL; }
    void number(Int n) {
        for (unsigned i = 0; i < 8; ++i) { byte(static_cast<unsigned>(n & 255)); n >>= 8; }
    }
    void string(V s) { number(s.size()); for (const char c : s) byte(static_cast<unsigned char>(c)); }
    void value(const Value& v) {
        byte(static_cast<unsigned>(v.data.index()));
        std::visit([&](const auto& item) { add(item); }, v.data);
    }
    void add(bool b) { byte(b ? 1U : 0U); }
    void add(std::int64_t n) { number(std::bit_cast<Int>(n)); }
    void add(double n) { number(std::bit_cast<Int>(n)); }
    void add(const std::string& s) { string(s); }
    void add(const A& a) { number(a.size()); for (const auto& v : a) value(v); }
    void add(const T& t) { number(t.size()); for (const auto& [k, v] : t) { string(k); value(v); } }
};
} // namespace

struct ModelDecl::Storage {
    std::set<std::string, std::less<>> strings;
    std::vector<std::vector<std::uint32_t>> format_layers;
    std::vector<std::array<std::vector<StageDecl>, 3>> tables;
    std::vector<std::vector<FormatRule>> formats;
    std::vector<DeclGraph> graphs;
};

std::span<const DeclGraph> ModelDecl::graphs() const noexcept {
    return storage_ ? std::span<const DeclGraph>(storage_->graphs) : std::span<const DeclGraph>{};
}

class ModelDeclLoader {
public:
    ModelDeclLoader(const ModelDeclOptions& options, const weight::TensorIndex* index)
        : options_(options), index_(index) {}
    ModelDecl load(const model_description::ModelDescription& description);

private:
    using Sym = weight::tensor_map::Sym;
    struct Class { std::vector<std::uint32_t> layers; V predicate; Traits bit; bool except; const T* sets; };
    struct Rule { V tensor, format, when; std::vector<std::uint32_t> layers; };
    const ModelDeclOptions& options_;
    const weight::TensorIndex* index_;
    std::shared_ptr<ModelDecl::Storage> storage_ = std::make_shared<ModelDecl::Storage>();
    ModelDecl out_;
    T dims_, names_;
    std::map<std::string, Class, std::less<>> classes_;
    std::vector<Rule> rules_;
    std::map<std::string, std::size_t, std::less<>> claimed_;
    std::vector<Traits> traits_;
    Fnv identity_;

    V keep(V s) { return s.empty() ? V{} : V(*storage_->strings.emplace(s).first); }
    Int dimension(V name, V where) const;
    Int extent(const T& t, V key, Int fallback, V where) const;
    bool guard(V predicate, const LayerVisit& visit, V phase, V where, unsigned depth = 0) const;
    bool in_class(const Class&, const LayerVisit&, V phase, V where, unsigned depth) const;
    void read_classes(const T& fields, std::uint32_t count);
    void read_formats(const T& fields, std::uint32_t count);
    void graph(const T& config, std::vector<T> stages, std::size_t index);
    StageDecl stage(StageFields& detail);
    void bind_weights(const StageFields&, const StageDecl&);
    void admit_tensors();
    void validate_values(const DeclGraph&, const std::vector<T>& values) const;
    void validate_schema(const T& fields);
    std::string expand(V pattern, const LayerVisit* visit, V where) const;
};

Int ModelDeclLoader::dimension(V name, V where) const {
    const auto* v = get(dims_, name);
    if (!v) fail(where, "unknown name " + std::string(name));
    return integer(*v, where);
}

Int ModelDeclLoader::extent(const T& t, V key, Int fallback, V where) const {
    const auto* v = get(t, key);
    if (!v) return fallback;
    if (const auto* s = std::get_if<std::string>(&v->data))
        return Expression(*s, where, [&](V name) { return dimension(name, where); }).read();
    return integer(*v, where);
}

bool ModelDeclLoader::in_class(const Class& c, const LayerVisit& visit, V phase, V where, unsigned depth) const {
    if (!c.predicate.empty()) return guard(c.predicate, visit, phase, where, depth + 1);
    return (std::ranges::find(c.layers, visit.physical_layer) != c.layers.end()) != c.except;
}

bool ModelDeclLoader::guard(V predicate, const LayerVisit& visit, V phase, V where, unsigned depth) const {
    if (predicate.empty()) return true;
    if (depth > classes_.size()) fail(where, "cycle in class predicates");
    return Expression(predicate, where, [&](V name) -> Int {
        if (name == "L") return visit.physical_layer;
        if (name == "pass") return visit.pass;
        if (name == "visit") return visit.logical_layer;
        if (name == "true") return 1;
        if (name == "false") return 0;
        if (name == "phase") return phase == "decode" ? 1U : phase == "prefill" ? 2U : 3U;
        if (name == "decode") return 1;
        if (name == "prefill") return 2;
        if (name == "verify") return 3;
        if (name == "class") {
            Int mask = class_set;
            for (const auto& [id, c] : classes_)
                if (in_class(c, visit, phase, where, depth)) mask |= c.bit;
            return mask;
        }
        const auto it = classes_.find(name);
        if (it != classes_.end())
            return in_class(it->second, visit, phase, where, depth) ? it->second.bit : 0U;
        return dimension(name, where);
    }, true).read() != 0;
}

void ModelDeclLoader::read_classes(const T& fields, std::uint32_t count) {
    Traits bit = 1U << 7;
    for (const auto& [name, value] : table(fields, "classes")) {
        Class c{{}, {}, 0, false, nullptr};
        for (const auto& [known, flag_bit] : class_bits) if (known == name) c.bit = flag_bit;
        if (!c.bit) {
            if (!bit) fail("classes", "too many layer classes");
            c.bit = bit;
            bit <<= 1;
        }
        if (const auto* settings = std::get_if<T>(&value.data)) {
            keys_only(*settings, "|layers|except|when|set|", name);
            c.sets = &table(*settings, "set");
            c.predicate = word(*settings, "when");
            const auto* layers = get(*settings, "layers");
            if (const auto* excluded = get(*settings, "except")) {
                if (layers) fail(name, "layers and except are mutually exclusive");
                layers = excluded;
                c.except = true;
            }
            if (layers) c.layers = layer_list(*layers, count, name);
            else if (c.predicate.empty()) c.except = true;
            if (layers && !c.predicate.empty()) fail(name, "layer list and predicate are mutually exclusive");
        } else if (const auto* expression = std::get_if<std::string>(&value.data)) c.predicate = *expression;
        else c.layers = layer_list(value, count, name);
        classes_.emplace(name, std::move(c));
    }
    traits_.resize(count);
    for (const auto& visit : out_.schedule) {
        for (const auto& [name, c] : classes_) {
            const bool selected = in_class(c, visit, "decode", name, 0);
            if (visit.pass == 0 && selected) traits_[visit.physical_layer] |= c.bit;
        }
    }
}

void ModelDeclLoader::read_formats(const T& fields, std::uint32_t count) {
    for (const auto& f : rows(fields, "format")) {
        keys_only(f, "|tensor|format|layers|when|transform|", "format");
        const auto pattern = word(f, "tensor");
        const auto star = pattern.find('*');
        if (star != V::npos && pattern.find('*', star + 1) != V::npos) fail("format", "multiple wildcards");
        Rule rule{keep(pattern), keep(word(f, "format")), keep(word(f, "when")), {}};
        if (rule.format.empty() && word(f, "transform").empty()) fail("format", "missing format or transform");
        if (const auto* layers = get(f, "layers")) {
            rule.layers = layer_list(*layers, count, "format");
            if (rule.layers.empty()) fail("format", "empty layer selection");
        }
        for (const auto& visit : out_.schedule) (void)guard(rule.when, visit, "decode", "format");
        rules_.push_back(std::move(rule));
    }
}

std::string ModelDeclLoader::expand(V pattern, const LayerVisit* visit, V where) const {
    std::string expanded;
    while (!pattern.empty()) {
        const auto open = pattern.find_first_of("{}");
        expanded += pattern.substr(0, open);
        if (open == V::npos) break;
        const auto close = pattern.find('}', open + 1);
        if (pattern[open] != '{' || close == V::npos || !visit) fail(where, "invalid layer symbol " + std::string(pattern));
        const auto token = pattern.substr(open + 1, close - open - 1);
        if (token == "L") expanded += std::to_string(visit->physical_layer);
        else if (token == "L-1" && visit->physical_layer != 0) expanded += std::to_string(visit->physical_layer - 1);
        else if (token == "visit") expanded += std::to_string(visit->logical_layer);
        else if (token == "class") {
            const auto selected = std::ranges::find_if(classes_, [&](const auto& item) {
                return in_class(item.second, *visit, "decode", where, 0);
            });
            if (selected == classes_.end()) fail(where, "no class for weight symbol");
            expanded += selected->first;
        } else fail(where, "invalid layer symbol {" + std::string(token) + "}");
        pattern.remove_prefix(close + 1);
    }
    return expanded;
}

StageDecl ModelDeclLoader::stage(StageFields& detail) {
    const auto& t = detail.fields;
    const V id = detail.id;
    keys_only(t, "|id|table|graph|after|before|op|form|entry|site|in|out|weight|operand|operands|"
        "rows|split|numeric|leaves|reduce|kernels|call|when|row_when|pos|state|attn|"
        "in_domain|out_domain|join_order|guard|activation|norm|gate|residual|bias|attrs|", id);
    StageDecl s;
    s.id = keep(id);
    const auto op = required(t, "op", id);
    s.op = named<Op>(op, ops, id);
    s.form = keep(word(t, "form"));
    s.entry = keep(op == "custom" ? required(t, "entry", id) : word(t, "entry"));
    s.site = keep(word(t, "site"));
    const auto inputs = words(t, "in");
    const auto outputs = words(t, "out");
    for (std::size_t i = 0; i < std::min(std::size(s.in), inputs.size()); ++i) s.in[i] = keep(inputs[i]);
    if (!outputs.empty()) s.out = keep(outputs.front());
    else if (s.op == Op::store || s.op == Op::clear || get(t, "state")) s.out = s.id;
    else fail(id, "missing out");
    const auto weights_list = words(t, "weight");
    const auto& operand = table(t, "operand");
    const auto items = extent(operand, "items", 1, id);
    s.operand = weights(keep(word(operand, "tensor", weights_list.empty() ? V{} : weights_list.front())),
                        extent(operand, "rows", 0, id), extent(operand, "cols", 0, id), items,
                        extent(operand, "read", 1, id));
    if ((s.op == Op::linear || s.op == Op::head) && (s.operand.tensor.empty() || !s.operand.rows || !s.operand.cols))
        fail(id, "projection without operand");
    if (!s.operand.read || s.operand.read > s.operand.items) fail(id, "invalid operand item count");
    if (const auto* n = get(t, "numeric")) {
        if (const auto* contract = std::get_if<T>(&n->data)) {
            keys_only(*contract, "|order|math|", id);
            s.numeric = keep(required(*contract, "order", id));
            const auto math = word(*contract, "math", "native");
            if (math != "native" && math != "ieee_cr" && math != "gfx1151-v1") fail(id, "unknown math " + std::string(math));
        } else s.numeric = keep(as<std::string>(*n, id, "a numeric contract"));
    }
    constexpr std::pair<V, Rows> row_kinds[] = {{"per_row", Rows::per_row}, {"cross_row", Rows::cross_row}};
    s.rows = named<Rows>(word(t, "rows", "per_row"), row_kinds, id);
    s.split = named<Split>(word(t, "split", "none"), splits, id);
    s.leaves = narrow<std::uint8_t>(extent(t, "leaves", 0, id), id);
    if (s.leaves != 0 && s.leaves != 4) fail(id, "leaves must be 0 or 4");
    s.reduce = flag(t, "reduce");
    s.kernels = narrow<std::uint8_t>(extent(t, "kernels", 1, id), id);
    s.call = keep(word(t, "call"));
    constexpr std::pair<V, Pos> positions[] = {{"any", Pos::any}, {"prefix_dense", Pos::prefix_dense},
        {"prefix_select", Pos::prefix_select}, {"group_close", Pos::group_close}};
    s.pos = named<Pos>(word(t, "pos", "any"), positions, id);
    const auto when = word(t, "when");
    if (!when.empty() && detail.table != "layer") fail(id, "layer guard on edge stage");
    if (const auto c = classes_.find(when); c != classes_.end()) s.when_all = c->second.bit;
    if (const auto c = classes_.find(when.starts_with('!') ? when.substr(1) : V{}); c != classes_.end())
        s.when_none = c->second.bit;
    if (detail.table == "layer") {
        for (const auto& visit : out_.schedule) {
            bool live = false;
            for (const V phase : {"decode", "prefill", "verify"}) live = guard(when, visit, phase, id) || live;
            if (live) detail.active_visits.push_back(visit);
        }
        if (detail.active_visits.empty()) fail(id, "stage active on no visit");
    }
    if (const auto row = word(t, "row_when"); !row.empty()) {
        (void)Expression(row, id, [&](V name) -> Int {
            if (name == "keys" || name == "topk" || name == "pos" || name == "row_class" ||
                name == "flags" || name == "true") return 1;
            if (name == "false") return 0;
            return dimension(name, id);
        }).read();
    }
    if (const auto& g = table(t, "guard"); !g.empty()) {
        keys_only(g, "|predicate|fast|exact|", id);
        for (const V key : {"predicate", "fast", "exact"}) (void)required(g, key, id);
    }
    for (const V key : {"in", "out"}) {
        const auto* value = get(t, key);
        const auto* ports = value ? std::get_if<A>(&value->data) : nullptr;
        if (!ports) continue;
        std::vector<std::pair<Int, Int>> slices;
        for (const auto& port : *ports) {
            const auto* fields = std::get_if<T>(&port.data);
            if (!fields) continue;
            keys_only(*fields, "|id|cols|epilogue|prologue|split|tile|", id);
            if (const auto split_name = word(*fields, "split"); !split_name.empty())
                (void)named<Split>(split_name, splits, id);
            for (const V attr : {"prologue", "epilogue"}) (void)words(*fields, attr);
            const auto cols = word(*fields, "cols");
            if (cols.empty()) continue;
            const auto colon = cols.find(':');
            if (colon == V::npos) fail(id, "slice requires start:end");
            const auto resolve = [&](V name) { return dimension(name, id); };
            const auto a = Expression(cols.substr(0, colon), id, resolve).read();
            const auto b = Expression(cols.substr(colon + 1), id, resolve).read();
            if (a >= b) fail(id, "invalid slice bounds");
            for (const auto& [first, last] : slices)
                if (a < last && first < b) fail(id, "overlapping slices");
            slices.emplace_back(a, b);
        }
    }
    return s;
}

namespace {
constexpr std::pair<V, std::uint32_t> wire_types[] = {
    {"f32", 0}, {"f16", 1}, {"q4_0", 2}, {"q5_0", 6}, {"q5_1", 7}, {"q8_0", 8},
    {"q2_k", 10}, {"q3_k", 11}, {"q4_k", 12}, {"q5_k", 13}, {"q6_k", 14},
    {"iq2_xxs", 16}, {"iq2_xs", 17}, {"iq3_xxs", 18}, {"iq4_nl", 20}, {"iq3_s", 21},
    {"iq2_s", 22}, {"iq4_xs", 23}, {"bf16", 30}, {"mxfp4", 39},
    {"f8_e4m3", gguf::kGgmlF8E4M3}, {"iu4", gguf::type_strix_iu4_g64}
};

std::uint64_t wire_mask(V format) {
    for (const auto& [name, type] : wire_types) if (name == format) return weight::tensor_map::type_bit(type);
    return 0;
}

std::string wire_format(const weight::RawTensor& t, weight::Container container) {
    if (t.wire_dtype.empty()) {
        if (container == weight::Container::gguf && t.wire_type == gguf::kNvfp4TilePack) return "nvfp4";
        for (const auto& [name, type] : wire_types) if (type == t.wire_type) return std::string(name);
        return {};
    }
    auto word = t.wire_dtype;
    for (auto& c : word) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return word;
}
} // namespace

void ModelDeclLoader::bind_weights(const StageFields& details, const StageDecl& s) {
    const auto& fields = details.fields;
    auto symbols = words(fields, "weight");
    if (symbols.empty() && !s.operand.tensor.empty()) symbols.push_back(s.operand.tensor);
    const auto& overrides = table(fields, "operands");
    for (const V symbol : symbols) {
        const auto& shape = table(overrides, symbol);
        const auto cols = extent(shape, "cols", s.operand.cols, s.id);
        const auto nrows = extent(shape, "rows", s.operand.rows, s.id);
        const auto items = extent(shape, "items", s.operand.items, s.id);
        if (!cols || !nrows || !items) fail(s.id, "missing shape for weight " + std::string(symbol));
        const auto add = [&](const LayerVisit* visit) {
            const auto* mapped = dotted(names_, symbol);
            const auto name = expand(mapped ? V(as<std::string>(*mapped, symbol, "a tensor name")) : symbol, visit, s.id);
            V format;
            for (const auto& r : rules_) {
                if (r.format.empty()) continue;
                if (!r.layers.empty() && (!visit || std::ranges::find(r.layers, visit->physical_layer) == r.layers.end()))
                    continue;
                if (!r.when.empty() && (!visit || !guard(r.when, *visit, "decode", s.id))) continue;
                if (matches(r.tensor, symbol) || matches(r.tensor, name) ||
                    (visit && r.tensor.find('{') != V::npos && matches(expand(r.tensor, visit, s.id), name)))
                    format = r.format;
            }
            std::array<Int, 4> dim{cols, nrows, items, 0};
            auto rank = static_cast<std::uint8_t>(items > 1 ? 3 : 2);
            if (const auto& axes = array(shape, "dims"); !axes.empty()) {
                if (axes.size() > dim.size()) fail(s.id, "weight rank exceeds four");
                dim = {};
                for (std::size_t i = 0; i < axes.size(); ++i) {
                    if (const auto* expr = std::get_if<std::string>(&axes[i].data))
                        dim[i] = Expression(*expr, s.id, [&](V key) { return dimension(key, s.id); }).read();
                    else dim[i] = integer(axes[i], s.id);
                    if (!dim[i]) fail(s.id, "empty weight extent");
                }
                rank = static_cast<std::uint8_t>(axes.size());
            }
            const auto [seen, inserted] = claimed_.try_emplace(name, out_.requirements.size());
            if (!inserted) {
                const auto& prior = out_.requirements.at(seen->second);
                if (prior.dims != dim || prior.rank != rank || prior.format != format)
                    fail(s.id, "conflicting claim for weight " + name);
                return;
            }
            std::array<weight::tensor_map::Extent, 4> legacy{};
            for (std::size_t axis = 0; axis < rank; ++axis)
                for (std::size_t i = 1; i < out_.symbols.value.size(); ++i)
                    if (out_.symbols.value[i] == dim[axis]) {
                        legacy[axis] = {static_cast<Sym>(i), Sym::one};
                        break;
                    }
            out_.requirements.push_back({name, details.id, std::string(format), dim, rank});
            out_.tensors.push_back({keep(name), weight::tensor_map::Layers::global, s.id, legacy, wire_mask(format)});
        };
        if (details.table == "layer") {
            for (const auto& visit : details.active_visits) add(&visit);
        } else add(nullptr);
    }
}

void ModelDeclLoader::admit_tensors() {
    if (!index_) return;
    std::map<V, const weight::RawTensor*> directory;
    for (const auto& tensor : index_->tensors)
        if (!directory.emplace(tensor.name, &tensor).second) fail(tensor.name, "duplicate tensor in index");
    for (std::size_t i = 0; i < out_.requirements.size(); ++i) {
        auto& requirement = out_.requirements[i];
        const auto found = directory.find(requirement.name);
        if (found == directory.end()) fail(requirement.stage, "missing tensor " + requirement.name);
        const auto& tensor = *found->second;
        if (tensor.shape.rank != requirement.rank ||
            !std::equal(requirement.dims.begin(), requirement.dims.begin() + requirement.rank,
                        std::make_reverse_iterator(tensor.shape.dims.begin() + tensor.shape.rank)))
            fail(requirement.stage, "wrong shape for tensor " + requirement.name);
        const auto actual = wire_format(tensor, index_->container);
        if (actual.empty() || (!requirement.format.empty() && actual != requirement.format))
            fail(requirement.stage, "wrong format for tensor " + requirement.name + " (" + actual + ")");
        requirement.format = actual;
        out_.tensors[i].types = wire_mask(actual);
        directory.erase(found);
    }
    const auto& ignored = table(table(out_.fields, "source"), "ignore");
    for (const auto& [name, tensor] : directory) {
        bool claimed = false;
        for (const auto& [pattern, reason] : ignored)
            if (matches(pattern, name) && !as<std::string>(reason, name, "an ignore reason").empty()) claimed = true;
        if (!claimed) fail(name, "unknown tensor in index");
    }
}

void ModelDeclLoader::graph(const T& config, std::vector<T> stage_rows, std::size_t index) {
    auto& result = storage_->graphs.at(index);
    result.id = word(config, "id", "text");
    result.kind = word(config, "kind", "decoder");
    if (result.kind != "decoder" && result.kind != "encoder" && result.kind != "denoiser" && result.kind != "codec")
        fail(result.id, "unknown graph kind " + result.kind);
    T templ;
    if (const auto key = word(config, "template"); !key.empty()) {
        std::set<std::string, std::less<>> seen;
        templ = template_fields(options_, key, seen);
    }
    stage_rows = overlay_rows(rows(templ, "stage"), stage_rows);
    auto values = rows(templ, "value");
    for (auto& v : rows(out_.fields, "value")) {
        if (word(v, "graph", result.id) == result.id) values.push_back(std::move(v));
    }
    for (auto& v : rows(config, "value")) values.push_back(std::move(v));
    if (stage_rows.empty()) fail(result.id, "empty stage table");
    auto& tables = storage_->tables.at(index);
    auto& legacy = storage_->formats.at(index);
    auto& g = result.stages;
    g.layers = traits_;
    g.select_after = narrow<std::uint32_t>(extent(config, "select_after", 0, result.id), result.id);
    g.group = narrow<std::uint32_t>(extent(config, "group", 0, result.id), result.id);
    Traits used = (1U << 7) - 1U;
    for (const auto& [name, c] : classes_) used |= c.bit;
    std::map<std::vector<bool>, Traits> legacy_guards;
    for (auto& row : stage_rows) {
        const auto id = required(row, "id", result.id);
        const auto scope = word(row, "table", "layer");
        std::size_t group{};
        if (scope == "before") group = 0;
        else if (scope == "layer") group = 1;
        else if (scope == "after") group = 2;
        else fail(id, "unknown stage table " + std::string(scope));
        result.rows.push_back({std::string(id), std::string(scope), std::move(row), {}});
        auto& detail = result.rows.back();
        auto s = stage(detail);
        if (const auto when = word(detail.fields, "when"); !when.empty() && !s.when_all && !s.when_none) {
            std::vector<bool> mask(g.layers.size());
            for (const auto& visit : out_.schedule)
                if (visit.pass == 0) mask[visit.physical_layer] = guard(when, visit, "decode", s.id);
            if (std::ranges::find(mask, false) != mask.end()) {
                auto [found, added] = legacy_guards.try_emplace(mask, 0);
                if (added) {
                    if (used == std::numeric_limits<Traits>::max()) fail(s.id, "legacy StageGraph trait capacity exceeded");
                    found->second = Traits{1} << std::countr_one(used);
                    used |= found->second;
                    for (std::size_t l = 0; l < mask.size(); ++l)
                        if (mask[l]) g.layers[l] |= found->second;
                }
                s.when_all = found->second;
            }
        }
        if (std::ranges::any_of(tables[group], [&](const StageDecl& earlier) { return earlier.id == s.id; }))
            fail(s.id, "duplicate stage id");
        tables[group].push_back(s);
        bind_weights(detail, s);
        identity_.add(detail.fields);
    }
    for (const auto& rule : rules_) {
        const auto format = legacy_format(rule.format);
        if (rule.format.empty() || !rule.when.empty()) continue;
        storage_->format_layers.push_back(rule.layers);
        const auto layers = std::span<const std::uint32_t>(storage_->format_layers.back());
        if (rule.tensor.find_first_of("*{") == V::npos) legacy.push_back({rule.tensor, format, layers});
        else {
            std::set<V> matched;
            for (const auto& t : tables)
                for (const auto& stage_row : t)
                    if (!stage_row.operand.tensor.empty() && matches(rule.tensor, stage_row.operand.tensor))
                        matched.insert(stage_row.operand.tensor);
            for (const auto symbol : matched) legacy.push_back({symbol, format, layers});
        }
    }
    g.before = tables[0];
    g.layer = tables[1];
    g.after = tables[2];
    g.formats = legacy;
    for (const auto& v : values) identity_.add(v);
    validate_values(result, values);
}

void ModelDeclLoader::validate_values(const DeclGraph& graph, const std::vector<T>& values) const {
    std::set<V> declared, known;
    for (const auto& value : values) {
        const auto id = required(value, "id", graph.id);
        if (!declared.insert(id).second) fail(id, "duplicate value id");
        const auto scope = word(value, "scope", "visit");
        if (scope != "visit" && scope != "pass" && scope != "step" && scope != "model")
            fail(id, "unknown value scope");
        if (!extent(value, "width", 1, id) || !extent(value, "streams", 1, id)) fail(id, "empty value shape");
    }
    known = declared;
    for (const auto& row : graph.rows)
        for (const auto output : words(row.fields, "out")) known.insert(output);
    for (const auto& row : graph.rows)
        for (const auto input : words(row.fields, "in"))
            if (!known.contains(input.substr(0, input.find_first_of("@["))))
                fail(row.id, "unknown value " + std::string(input));

    struct Node { const StageFields* row; const LayerVisit* visit; std::vector<std::size_t> deps; };
    std::vector<Node> nodes;
    std::map<std::pair<std::uint32_t, V>, std::vector<std::size_t>> by_visit;
    std::vector<std::vector<const LayerVisit*>> by_layer(traits_.size());
    const auto add = [&](const StageFields& row, const LayerVisit* visit, std::uint32_t slot) {
        for (const auto output : words(row.fields, "out")) by_visit[{slot, output}].push_back(nodes.size());
        nodes.push_back({&row, visit, {}});
    };
    const auto pre = static_cast<std::uint32_t>(out_.schedule.size());
    for (const auto& row : graph.rows) if (row.table == "before") add(row, nullptr, pre);
    for (const auto& visit : out_.schedule) {
        by_layer[visit.physical_layer].push_back(&visit);
        for (const auto& row : graph.rows)
            if (row.table == "layer" && std::ranges::binary_search(row.active_visits, visit.logical_layer, {},
                                                                  &LayerVisit::logical_layer))
                add(row, &visit, visit.logical_layer);
    }
    for (const auto& row : graph.rows) if (row.table == "after") add(row, nullptr, pre + 1U);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];
        const auto id = V(node.row->id);
        const auto slot = node.visit ? node.visit->logical_layer : node.row->table == "before" ? pre : pre + 1U;
        for (const auto input : words(node.row->fields, "in")) {
            const auto at = input.find('@');
            if (at == V::npos) {
                const auto value = input.substr(0, input.find('['));
                const auto found = by_visit.find({slot, value});
                if (found == by_visit.end()) continue;
                const auto earlier = std::ranges::lower_bound(found->second, i);
                if (earlier != found->second.begin()) node.deps.push_back(*(earlier - 1));
                else if (!declared.contains(value) && found->second.back() > i) node.deps.push_back(found->second.back());
                continue;
            }
            const auto target = Expression(input.substr(at + 1), id, [&](V name) -> Int {
                if (name != "L" || !node.visit) fail(id, "unknown value reference name " + std::string(name));
                return node.visit->physical_layer;
            }).read();
            if (target >= by_layer.size()) fail(id, "value reference outside physical layers");
            const auto& visits = by_layer[static_cast<std::size_t>(target)];
            const auto pass = node.visit ? node.visit->pass : 0U;
            const auto target_visit = visits[std::min(static_cast<std::size_t>(pass), visits.size() - 1)]->logical_layer;
            const auto found = by_visit.find({target_visit, input.substr(0, at)});
            if (found == by_visit.end()) fail(id, "no producer for " + std::string(input));
            node.deps.push_back(found->second.back());
        }
        const auto state_id = word(node.row->fields, "state");
        if (state_id.empty()) continue;
        const auto state = std::ranges::find_if(out_.states, [&](const T& t) { return word(t, "id") == state_id; });
        if (state == out_.states.end()) fail(id, "unknown state " + std::string(state_id));
        if (word(*state, "rollback") != "replay") continue;
        for (const auto input : words(*state, "replay_inputs")) {
            const auto found = by_visit.find({slot, input});
            if (found == by_visit.end() || found->second.front() >= i)
                fail(id, "replay input not produced in the window: " + std::string(input));
        }
    }
    std::vector<std::vector<std::size_t>> users(nodes.size());
    std::vector<std::size_t> remaining;
    std::vector<std::size_t> ready;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        remaining.push_back(nodes[i].deps.size());
        if (nodes[i].deps.empty()) ready.push_back(i);
        for (const auto dep : nodes[i].deps) users[dep].push_back(i);
    }
    for (std::size_t i = 0; i < ready.size(); ++i)
        for (const auto user : users[ready[i]]) if (--remaining[user] == 0) ready.push_back(user);
    if (ready.size() != nodes.size()) {
        const auto blocked = std::ranges::find_if(remaining, [](auto left) { return left != 0; });
        fail(nodes[static_cast<std::size_t>(blocked - remaining.begin())].row->id, "cycle in value references");
    }
}

void ModelDeclLoader::validate_schema(const T& fields) {
    keys_only(fields, "|schema|name|license|model|source|dims|schedule|classes|graph|value|stage|state|"
        "format|image|head|draft|tiers|split|splice|pipeline|loop|qualification|tokenizer|", "root");
    const auto& source = table(fields, "source");
    keys_only(source, "|container|architecture|revision|tokenizer|chat|files|names|ignore|", "source");
    out_.architecture = required(source, "architecture", "source");
    out_.chat = word(source, "chat");
    out_.tokenizer = word(source, "tokenizer");
    const auto container = required(source, "container", "source");
    constexpr std::pair<V, weight::Container> containers[] = {{"gguf", weight::Container::gguf},
        {"safetensors", weight::Container::safetensors}, {"pack", weight::Container::pack}};
    const auto expected_container = named<weight::Container>(container, containers, "source");
    if (index_ && index_->container != expected_container) fail("source", "checkpoint container mismatch");
    if (index_) {
        const V metadata_key = index_->container == weight::Container::gguf ? "general.architecture" : "model_type";
        const auto arch = std::ranges::find(index_->metadata, metadata_key, &std::pair<std::string, std::string>::first);
        if (arch != index_->metadata.end() && arch->second != out_.architecture)
            fail("source", "checkpoint architecture mismatch");
    }
    names_ = table(source, "names");

    dims_ = table(fields, "dims");
    for (auto& [key, value] : dims_) {
        if (const auto* reference = std::get_if<std::string>(&value.data)) {
            if (!reference->starts_with('@') || reference->size() == 1) fail(key, "expected @metadata.key");
            const auto metadata = V(*reference).substr(1);
            if (!index_) fail(key, "metadata dimension needs a TensorIndex: " + std::string(metadata));
            const auto found = std::ranges::find(index_->metadata, metadata, &std::pair<std::string, std::string>::first);
            if (found == index_->metadata.end()) fail(key, "missing metadata " + std::string(metadata));
            const auto& text = found->second;
            std::int64_t n{};
            const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n);
            if (error == std::errc{} && end == text.data() + text.size()) value.data = n;
            else {
                double d{};
                const auto [last, ec] = std::from_chars(text.data(), text.data() + text.size(), d);
                if (ec != std::errc{} || last != text.data() + text.size() || !std::isfinite(d) || d < 0)
                    fail(key, "invalid numeric metadata " + std::string(metadata));
                value.data = d;
            }
        }
        if (const auto* d = std::get_if<double>(&value.data)) {
            if (*d < 0 || !std::isfinite(*d)) fail(key, "invalid dimension");
        } else (void)integer(value, key);
    }
    out_.fields.insert_or_assign("dims", Value{dims_});
    const auto count = narrow<std::uint32_t>(dimension("layers", "dims"), "layers");
    if (!count || count > 1'000'000) fail("layers", "invalid physical layer count");
    out_.symbols.blocks = count;
    out_.symbols.value[static_cast<std::size_t>(Sym::one)] = 1;
    out_.symbols.value[static_cast<std::size_t>(Sym::two)] = 2;
    constexpr std::pair<V, Sym> legacy_dims[] = {{"hidden", Sym::hidden}, {"vocab", Sym::vocab},
        {"heads", Sym::heads}, {"head_dim", Sym::qk_head}, {"ffn", Sym::dense_ff}, {"experts", Sym::experts},
        {"expert_ff", Sym::expert_ff}, {"latent", Sym::latent}};
    for (const auto& [key, sym] : legacy_dims)
        if (get(dims_, key)) out_.symbols.value[static_cast<std::size_t>(sym)] = dimension(key, "dims");
    out_.schedule = schedule(word(table(fields, "schedule"), "order"), count);
    read_classes(fields, count);
    read_formats(fields, count);
    out_.states = rows(fields, "state");
    std::set<V> ids;
    for (const auto& state : out_.states) {
        const auto id = required(state, "id", "state");
        if (!ids.insert(id).second) fail(id, "duplicate state id");
        (void)required(state, "kind", id);
        (void)required(state, "dtype", id);
        const auto rollback = word(state, "rollback", "rewind");
        if (rollback != "rewind" && rollback != "replay" && rollback != "snapshot" && rollback != "none" &&
            rollback != "recompute" && rollback != "ring") fail(id, "unknown rollback policy");
        if (rollback == "replay" && words(state, "replay_inputs").empty()) fail(id, "missing replay inputs");
        for (const auto& visit : out_.schedule)
            (void)guard(word(state, "when"), visit, "decode", id);
    }
    out_.drafts = rows(fields, "draft");
    for (const auto& draft : out_.drafts) {
        const auto kind = required(draft, "kind", "draft");
        if (kind != "chain" && kind != "block" && kind != "lookup") fail("draft", "unknown kind " + std::string(kind));
        for (const auto state : words(draft, "shares"))
            if (!ids.contains(state)) fail("draft", "unknown shared state " + std::string(state));
    }
    for (const auto& image : rows(fields, "image")) {
        (void)required(image, "of", "image");
        (void)required(image, "format", "image");
        const auto lifetime = word(image, "lifetime", "load");
        if (lifetime != "load" && lifetime != "session" && lifetime != "request") fail("image", "unknown lifetime");
    }
    for (const V key : {"head", "tiers", "split", "qualification", "pipeline", "loop"})
        (void)table(fields, key);
}

ModelDecl ModelDeclLoader::load(const model_description::ModelDescription& description) {
    const auto& fields = description.fields;
    const auto& model = table(fields, "model");
    if (!model.empty()) keys_only(model, "|schema|id|licence|license|reference|", "model");
    if (!model.empty() && (get(fields, "schema") || get(fields, "name") || get(fields, "license")))
        fail("model", "mixed model headers");
    if (word(model.empty() ? fields : model, "schema") != "strix-model/v2") fail("model", "expected strix-model/v2");
    out_.name = required(model.empty() ? fields : model, model.empty() ? "name" : "id", "model");
    out_.license = word(model.empty() ? fields : model, "license", word(model, "licence"));
    out_.reference = word(model, "reference");
    out_.fields = fields;
    validate_schema(fields);
    identity_.add(out_.fields);
    auto graphs = rows(fields, "graph");
    if (graphs.empty()) graphs.emplace_back();
    storage_->graphs.resize(graphs.size());
    storage_->tables.resize(graphs.size());
    storage_->formats.resize(graphs.size());
    std::set<V> ids;
    for (std::size_t i = 0; i < graphs.size(); ++i) {
        const auto& config = graphs[i];
        keys_only(config, "|id|kind|template|stage|value|splice|select_after|group|", "graph");
        const auto id = word(config, "id", "text");
        if (!ids.insert(id).second) fail(id, "duplicate graph id");
        auto stage_rows = rows(config, "stage");
        for (auto& stage_row : rows(fields, "stage"))
            if (word(stage_row, "graph", word(graphs.front(), "id", "text")) == id)
                stage_rows.push_back(std::move(stage_row));
        graph(config, std::move(stage_rows), i);
    }
    for (const V key : {"stage", "value"})
        for (const auto& row : rows(fields, key))
            if (const auto id = word(row, "graph"); !id.empty() && !ids.contains(id)) fail(key, "unknown graph " + std::string(id));
    const auto check_splice = [&](const T& splice, const DeclGraph& target) {
        const auto from = required(splice, "from", target.id);
        const auto into = required(splice, "into", target.id);
        if (!ids.contains(from) || from == target.id) fail(target.id, "invalid splice source " + std::string(from));
        if (std::ranges::find(target.rows, into, &StageFields::id) == target.rows.end())
            fail(target.id, "unknown splice target " + std::string(into));
        const auto* marker = get(splice, "marker");
        if (!marker) fail(target.id, "missing splice marker");
        (void)integer(*marker, target.id);
    };
    for (std::size_t i = 0; i < graphs.size(); ++i)
        for (const auto& splice : rows(graphs[i], "splice")) check_splice(splice, storage_->graphs[i]);
    for (const auto& splice : rows(fields, "splice")) check_splice(splice, storage_->graphs.front());
    admit_tensors();
    for (const auto& req : out_.requirements) { identity_.string(req.name); identity_.string(req.format); }
    out_.identity = identity_.hash;
    out_.stages = storage_->graphs.front().stages;
    out_.storage_ = storage_;
    return std::move(out_);
}

ModelDecl inspect_model_decl(const model_description::ModelDescription& desc, const ModelDeclOptions& options) {
    return ModelDeclLoader(options, nullptr).load(desc);
}

ModelDecl load_model_decl(const model_description::ModelDescription& desc, const weight::TensorIndex& index,
                          const ModelDeclOptions& options) {
    return ModelDeclLoader(options, &index).load(desc);
}

void validate(const ModelDecl& model, const DeclLookup& lookup) {
    const auto need = [&](DeclKeyKind kind, V graph, V stage, V key) {
        if (!lookup({kind, graph, stage, key})) fail(stage, "missing catalogue key " + std::string(key));
    };
    for (const auto& graph : model.graphs()) {
        for (const auto& row : graph.rows) {
            const auto& fields = row.fields;
            const auto op = word(fields, "op");
            const auto form = word(fields, "form");
            const auto operation = std::string(op) + ":" + std::string(op == "custom" ? word(fields, "entry") : form);
            need(DeclKeyKind::operation, graph.id, row.id, operation);
            for (const auto& [key, form_value] : table(fields, "guard"))
                if (key != "predicate")
                    need(DeclKeyKind::operation, graph.id, row.id,
                         std::string(op) + ":" + as<std::string>(form_value, row.id, "a form"));
            for (const V key : {"in", "out"}) {
                const auto* ports = get(fields, key);
                const auto* many = ports ? std::get_if<A>(&ports->data) : nullptr;
                if (!many) continue;
                for (const auto& port : *many)
                    if (const auto* spec = std::get_if<T>(&port.data))
                        for (const V chain : {"prologue", "epilogue"})
                            for (const auto step : words(*spec, chain))
                                need(DeclKeyKind::operation, graph.id, row.id, step);
            }
            if (op == "attend") {
                const auto check_dim = [&](const Value* dim) {
                    if (dim) need(DeclKeyKind::head_dim, graph.id, row.id, std::to_string(integer(*dim, row.id)));
                };
                const auto* local = get(table(fields, "attn"), "head_dim");
                check_dim(local ? local : get(table(model.fields, "dims"), "head_dim"));
                for (const auto& [name, c] : table(model.fields, "classes")) {
                    const auto* traits = std::get_if<T>(&c.data);
                    if (traits) check_dim(dotted(table(*traits, "set"), "attend.head_dim"));
                }
            }
            if (const auto state = word(fields, "state"); !state.empty()) {
                const auto found = std::ranges::find_if(model.states, [&](const T& st) { return word(st, "id") == state; });
                if (found == model.states.end()) fail(row.id, "unknown state " + std::string(state));
                need(DeclKeyKind::state_layout, graph.id, row.id, word(*found, "kind"));
            }
        }
    }
    for (const auto& state : model.states)
        need(DeclKeyKind::state_layout, {}, word(state, "id"), word(state, "kind"));
    for (const auto& tensor : model.requirements)
        if (!tensor.format.empty()) need(DeclKeyKind::decoder, {}, tensor.stage, tensor.format);
    for (const auto& image : rows(model.fields, "image"))
        need(DeclKeyKind::decoder, {}, word(image, "of"), word(image, "format"));
    for (const auto& format : rows(model.fields, "format"))
        if (const auto transform = word(format, "transform"); !transform.empty())
            need(DeclKeyKind::transform, {}, word(format, "tensor", "format"), transform.substr(0, transform.find('{')));
}

} // namespace strix::graph
