#include "strix/serve/serial_generation_cli.hpp"

#include "core/dense_linear.hpp"
#include "strix/exec/route_capabilities.hpp"
#include "strix/env_flags.hpp"
#include "strix/serial_request.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/decode_form.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/media.hpp"
#include "strix/serve/prefill_chunk_fallback.hpp"
#include "strix/session/embed_rows.hpp"
#include "strix/session/cli_output.hpp"
#include "strix/spec/block_setup.hpp"
#include "strix/spec/greedy_continuation.hpp"
#include "strix/spec/probe_policy.hpp"
#include "strix/spec/verify_cost_table.hpp"
#include "strix/token_sampling.hpp"
#include "strix/token_sampling_device.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <deque>
#include <optional>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <process.h>
#include <share.h>
#else
#include <unistd.h>
#endif

namespace strix::serve {
namespace {

constexpr auto speculation_modes = "off|auto|verify|lookup";

std::string_view speculation_name(CliSpeculation mode) {
    switch (mode) {
        case CliSpeculation::off: return "off";
        case CliSpeculation::automatic: return "auto";
        case CliSpeculation::verify: return "verify";
        case CliSpeculation::lookup: return "lookup";
    }
    throw std::logic_error("unknown speculation mode");
}

using Clock = session::CliClock;
namespace fs = std::filesystem;

// The persisted verify walls of this run's speculator (strix/spec/verify_cost_table.hpp), or nothing: no device
// identity from the decoder, the `verify` diagnostic (equal prices), no table directory (STRIX_COST_TABLE=0, baseline,
// no cache directory) or no executable identity. The model identity is the checkpoint's path, size and write time.
std::optional<CostTableCache> open_verify_cost_table(const SerialGenerationFamily& family, const std::string& checkpoint,
                                                     const SerialGenerationDecoder& decoder, CliSpeculation mode,
                                                     bool sampled, std::uint32_t max_rows, std::uint32_t prompt_size) {
    auto device = decoder.verify_cost_device();
    if (device.empty() || mode == CliSpeculation::verify) return std::nullopt;
    const auto directory = cost_table_directory_from_env();
    auto binary = cost_table_executable_identity();
    if (!directory || binary.empty()) return std::nullopt;
    std::error_code ec;
    auto model = fs::weakly_canonical(fs::path(checkpoint), ec).string();
    if (ec) model = checkpoint;
    if (const auto size = fs::file_size(fs::path(checkpoint), ec); !ec) model += ';' + std::to_string(size);
    if (const auto written = fs::last_write_time(fs::path(checkpoint), ec); !ec)
        model += ';' + std::to_string(written.time_since_epoch().count());
    auto policy = cost_table_policy_identity("serial-verify-cost-v1;mode=" + std::string(speculation_name(mode)) +
                                             ";sampled=" + (sampled ? "1" : "0") + ";rows=" + std::to_string(max_rows));
    std::optional<CostTableCache> table;
    table.emplace(*directory,
                  CostTableKey{std::string(family.executable), std::move(binary), std::move(device),
                               cost_table_hash_hex(model), std::move(policy), cost_table_context_bucket(prompt_size)},
                  spec::verify_cost_table_options());
    (void)table->load();
    return table;
}

template <class Number>
Number parse_number(std::string_view value, const char* error) {
    Number result{};
    const auto [end, code] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (code != std::errc{} || end != value.data() + value.size()) throw std::invalid_argument(error);
    return result;
}

// The --logits file. The dump has no name until the run succeeds: on Linux it is an unnamed file (O_TMPFILE) of the
// destination directory, which is held open from the start; on Windows a file under a bounded random name of that
// directory, created exclusively, delete-on-close and not shareable for deletion or writing, whose handle stays open.
// It is written (and flushed) through its descriptor only, and published on success by an operation that never
// replaces: linkat of the descriptor under the name relative to the held directory (EEXIST when the name exists), or
// a by-handle rename with ReplaceIfExists = FALSE relative to the held directory handle followed by clearing the
// delete-on-close disposition. No name is resolved again after creation. A name that exists (a file, a link, even a
// dangling one) is refused at the start and, if it appeared during the run, at publication, and is left untouched. A
// run that fails publishes nothing and leaves nothing behind (closing the descriptor discards the dump). Linux needs a
// file system with O_TMPFILE support in the destination directory; otherwise --logits is refused (no named fallback).
class LogitsOutput {
public:
    explicit LogitsOutput(const fs::path& path)
        : path_(path), directory_(path.has_parent_path() ? path.parent_path() : fs::path(".")) {
        const auto name = path.filename();
        if (name.empty() || name == "." || name == "..") throw std::invalid_argument("logits output must name a file");
#ifdef _WIN32
        dir_ = CreateFileW(directory_.c_str(), FILE_LIST_DIRECTORY | FILE_ADD_FILE | FILE_TRAVERSE | SYNCHRONIZE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS,
                           nullptr);
        if (dir_ == INVALID_HANDLE_VALUE) throw std::invalid_argument("could not open the logits output directory");
        if (name_taken()) {
            discard();
            throw std::invalid_argument("logits output exists (an existing file is never replaced)");
        }
        std::random_device random;
        for (unsigned attempt = 0; fd_ < 0; ++attempt) {
            std::wstring staged = L".strix-logits-";
            for (int part = 0; part < 4; ++part) {
                wchar_t hex[9]{};
                (void)swprintf(hex, 9, L"%08x", unsigned(random()));
                staged += hex;
            }
            staged += L".tmp";
            const HANDLE file = CreateFileW((directory_ / staged).c_str(), GENERIC_WRITE | DELETE | SYNCHRONIZE,
                                            FILE_SHARE_READ, nullptr, CREATE_NEW,
                                            FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
            if (file == INVALID_HANDLE_VALUE) {
                if (GetLastError() == ERROR_FILE_EXISTS && attempt < 15U) continue;
                discard();
                throw std::invalid_argument("could not create the logits output");
            }
            fd_ = _open_osfhandle(reinterpret_cast<intptr_t>(file), _O_WRONLY | _O_BINARY);
            if (fd_ < 0) {
                CloseHandle(file);  // delete-on-close: nothing is left
                discard();
                throw std::runtime_error("could not create the logits output");
            }
        }
#else
        dir_ = ::open(directory_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dir_ < 0) throw std::invalid_argument("could not open the logits output directory");
        if (name_taken()) {
            discard();
            throw std::invalid_argument("logits output exists (an existing file is never replaced)");
        }
#ifdef O_TMPFILE
        fd_ = ::openat(dir_, ".", O_TMPFILE | O_WRONLY | O_CLOEXEC, 0600);
        const int error = errno;
#else
        const int error = EOPNOTSUPP;
#endif
        if (fd_ < 0) {
            discard();
            if (error == EOPNOTSUPP || error == EISDIR || error == ENOENT || error == EINVAL)
                throw std::invalid_argument("--logits needs a file system that supports unnamed files (O_TMPFILE) in "
                                            "the output directory; this one does not");
            throw std::invalid_argument("could not create the logits output");
        }
#endif
        key_ = descriptor_file_key(fd_);
        if (!key_) {
            discard();
            throw std::runtime_error("could not identify logits output");
        }
    }
    ~LogitsOutput() { discard(); }  // a run that fails publishes nothing
    LogitsOutput(const LogitsOutput&) = delete;
    LogitsOutput& operator=(const LogitsOutput&) = delete;

    void refuse(const std::optional<FileKey>& input, const char* error) const {
        if (input && input == key_) throw std::invalid_argument(error);
    }
    void refuse(const fs::path& input, const char* error) const { refuse(file_key(input), error); }
    void write(std::span<const float> logits) {  // through the descriptor only; visible under the name after publish()
        const auto bytes = session::logits_bytes(logits);
        bool ok = fd_ >= 0;
        for (std::size_t at = 0; ok && at < bytes.size();) {
#ifdef _WIN32
            const auto n = _write(fd_, bytes.data() + at, unsigned(std::min<std::size_t>(bytes.size() - at, 1U << 30U)));
#else
            const auto n = ::write(fd_, bytes.data() + at, bytes.size() - at);
#endif
            ok = n > 0;
            at += ok ? std::size_t(n) : 0U;
        }
#ifdef _WIN32
        ok = ok && _commit(fd_) == 0;
#else
        ok = ok && ::fsync(fd_) == 0;
#endif
        if (!ok) throw std::runtime_error("could not write logits output");
        written_ = true;
    }
    void publish() {  // the success boundary of the run: never replaces a name that exists
        if (!written_) return;
        bool taken = false, published = false;
#ifdef _WIN32
        const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
        const std::wstring name = path_.filename().wstring();
        std::vector<unsigned char> buffer(sizeof(FILE_RENAME_INFO) + name.size() * sizeof(wchar_t));
        auto* rename = reinterpret_cast<FILE_RENAME_INFO*>(buffer.data());
        rename->ReplaceIfExists = FALSE;
        rename->RootDirectory = dir_;
        rename->FileNameLength = DWORD(name.size() * sizeof(wchar_t));
        std::copy(name.begin(), name.end(), rename->FileName);
        // NtSetInformationFile, not SetFileInformationByHandle: the latter refuses RootDirectory and resolves a relative
        // name against the process's current directory.
        struct IoStatus {
            union {
                LONG status;
                PVOID pointer;
            };
            ULONG_PTR information;
        } io{};
        using SetInformation = LONG(NTAPI*)(HANDLE, IoStatus*, PVOID, ULONG, int);
        const auto set_information = reinterpret_cast<SetInformation>(
            reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtSetInformationFile")));
        constexpr int file_rename_information = 10;
        constexpr LONG name_collision = LONG(0xC0000035L);  // STATUS_OBJECT_NAME_COLLISION
        const LONG status = set_information ? set_information(file, &io, rename, ULONG(buffer.size()),
                                                              file_rename_information)
                                            : name_collision + 1;
        if (status == 0) {
            FILE_DISPOSITION_INFO_EX keep{};
            keep.Flags = FILE_DISPOSITION_FLAG_ON_CLOSE;  // without FILE_DISPOSITION_FLAG_DELETE: clears delete-on-close
            published = SetFileInformationByHandle(file, FileDispositionInfoEx, &keep, sizeof(keep)) != 0;
        } else {
            taken = status == name_collision;
        }
#else
        const auto self = "/proc/self/fd/" + std::to_string(fd_);
        published = ::linkat(AT_FDCWD, self.c_str(), dir_, path_.filename().c_str(), AT_SYMLINK_FOLLOW) == 0;
        taken = !published && errno == EEXIST;
#endif
        discard();  // closes the descriptor (an unpublished dump goes with it) and the directory
        if (!published)
            throw std::runtime_error(taken ? "logits output exists (it appeared during the run and is never replaced)"
                                           : "could not publish logits output");
    }

private:
    bool name_taken() const {  // anything under the name, a dangling link included
#ifdef _WIN32
        std::error_code ignored;
        return fs::symlink_status(path_, ignored).type() != fs::file_type::not_found;
#else
        struct stat info {};
        return ::fstatat(dir_, path_.filename().c_str(), &info, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT;
#endif
    }
    void discard() {
#ifdef _WIN32
        if (fd_ >= 0) (void)_close(fd_);
        if (dir_ != INVALID_HANDLE_VALUE) CloseHandle(dir_);
        dir_ = INVALID_HANDLE_VALUE;
#else
        if (fd_ >= 0) (void)::close(fd_);
        if (dir_ >= 0) (void)::close(dir_);
        dir_ = -1;
#endif
        fd_ = -1;
    }
    fs::path path_, directory_;
    bool written_ = false;
    int fd_ = -1;
#ifdef _WIN32
    HANDLE dir_ = INVALID_HANDLE_VALUE;
#else
    int dir_ = -1;
#endif
    std::optional<FileKey> key_;
};

struct TextInput {
    std::string text;
    std::optional<FileKey> key;  // of the handle the text was read from
};

TextInput read_text_input(const char* path) {
    std::FILE* file = nullptr;
#ifdef _WIN32
    if (fopen_s(&file, path, "rb") != 0) file = nullptr;
#else
    file = std::fopen(path, "rb");
#endif
    if (!file) throw std::runtime_error("could not open text input: " + std::string(path));
    TextInput input;
#ifdef _WIN32
    input.key = descriptor_file_key(_fileno(file));
#else
    input.key = descriptor_file_key(fileno(file));
#endif
    char buffer[1U << 16U];
    for (std::size_t n; (n = std::fread(buffer, 1, sizeof buffer, file)) > 0;) input.text.append(buffer, n);
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) throw std::runtime_error("could not read text input: " + std::string(path));
    return input;
}

std::string read_text(const char* path) { return read_text_input(path).text; }

// Per-round diagnostic (STRIX_SPEC_ROUND_LOG=N; off by default): the first N verify rounds of a speculating run, one
// stderr line each, written after the round's commit (spec::RoundSink): no effect on any decision or bit. Two runs that
// should decide alike (e.g. an executor switched on and off) are compared line by line; the first differing line names
// the round, and its fields what differs (source / window, drafts, the target's pick at every verified row, accepted).
class RoundLog final : public spec::RoundSink {
public:
    RoundLog(std::ostream& out, std::uint32_t rounds) : out_(out), left_(rounds) {}
    void round(const spec::RoundRecord& record) override {
        if (left_ == 0U) return;
        --left_;
        out_ << "spec-round " << index_++ << " context " << record.context << " source "
             << spec::source_name(record.source) << " rows " << record.rows << " accepted " << record.accepted
             << " copy " << record.copy << " ext " << record.ext << " drafts";
        for (const auto id : record.drafts) out_ << ' ' << id;
        out_ << " picks";
        for (const auto id : record.greedy) out_ << ' ' << id;
        out_ << '\n';
    }

private:
    std::ostream& out_;
    std::uint32_t left_;
    std::uint32_t index_ = 0;
};

}  // namespace

// --messages with image parts: the shared media layer normalizes them (data URLs / base64 only: the CLI has no
// fetcher), the family renders one placeholder per part, its adapter encodes each image within the context left
// by the text, and the placeholders expand to the virtual ids of the rows (session::expand_embed_rows).
struct MediaPrompt {
    std::vector<std::uint32_t> tokens;
    std::shared_ptr<const session::EmbedTable> table;  // null: the body holds no image
};

MediaPrompt render_media_prompt(const SerialGenerationFamily& family, const std::string& checkpoint,
                                const BpeTokenizer& tokenizer, std::string_view body_text,
                                std::uint32_t reserve_tokens, std::unique_ptr<SerialMediaInput>& media,
                                const LogitsOutput* output) {
    MediaPrompt out;
    auto body = parse_json(body_text);
    if (body.is_array()) {  // a bare messages array is the body {"messages": [...]}
        auto wrapped = Json::object();
        wrapped.set("messages", std::move(body));
        body = std::move(wrapped);
    }
    auto normalized = media::normalize(body, media::Api::chat_completions);
    if (normalized.items.empty()) {
        out.tokens = tokenizer.encode(family.render_messages(body_text), true);
        return out;
    }
    media = family.media(checkpoint);
    if (!media) throw std::runtime_error("the family opened no media input");
    if (output)  // before the encoder runs: the output may be none of the files the media input read
        for (const auto& input : media->input_keys())
            output->refuse(input, "logits output must differ from the image input files");
    auto& adapter = media->adapter();
    const auto placeholder_ids = tokenizer.encode(adapter.placeholder(), true);
    if (placeholder_ids.size() != 1U || session::is_embed_id(placeholder_ids[0]))
        throw std::invalid_argument("media placeholder must tokenize to one id");
    const auto placeholder = placeholder_ids[0];
    const auto tokens = tokenizer.encode(family.render_media_messages(normalized.body.dump()), true);
    const auto count = normalized.items.size();
    if (static_cast<std::size_t>(std::count(tokens.begin(), tokens.end(), placeholder)) != count)
        throw std::invalid_argument("the rendered prompt does not hold one placeholder per image");
    const std::uint64_t text = tokens.size() - count + reserve_tokens;
    std::uint64_t room = family.max_context > text ? family.max_context - text : 0U;
    std::vector<std::vector<std::uint32_t>> layouts;
    std::vector<session::EmbedLease> leases;
    for (const auto& item : normalized.items) {
        const auto budget = static_cast<std::uint32_t>(
            std::min<std::uint64_t>({adapter.max_rows(), room, session::kEmbedMaxRows}));
        if (budget == 0U) throw std::invalid_argument("image rows do not fit the context");
        auto span = adapter.encode(item, budget);
        if (!span.lease || span.layout.empty() || span.layout.size() > room)
            throw std::invalid_argument("image rows do not fit the context");
        room -= span.layout.size();
        layouts.push_back(std::move(span.layout));
        leases.push_back(std::move(span.lease));
    }
    auto expanded = session::expand_embed_rows(tokens, placeholder, layouts);
    out.tokens = std::move(expanded.tokens);
    out.table = std::make_shared<const session::EmbedTable>(std::move(leases));
    return out;
}

SerialGenerationOptions parse_serial_generation_cli(int argc, char** argv, const SerialGenerationFamily& family,
                                                    const SerialCliOption& extra) {
    SerialGenerationOptions options;
    options.sampler_options = validate_sampling_startup_environment();
    bool grouped_decode = false;
    options.prefill_chunk = family.default_prefill_chunk;
    if (family.supports_speculation) options.speculation = family.default_speculation;
    std::optional<float> temperature;
    for (int i = 2; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        const auto value = [&]() {
            if (++i == argc) throw std::invalid_argument("missing option value");
            return argv[i];
        };
        if (argument == "--generate") {
            options.generate = core::parse_u32(value());
            options.generate_requested = true;
        }
        else if (argument == "--logits") options.logits = value();
        else if (family.supports_speculation && argument == "--speculation") {
            options.speculation_requested = true;
            const std::string_view mode(value());
            if (mode == "off") options.speculation = CliSpeculation::off;
            else if (mode == "auto") options.speculation = CliSpeculation::automatic;
            else if (mode == "verify") options.speculation = CliSpeculation::verify;
            else if (mode == "lookup") options.speculation = CliSpeculation::lookup;
            else throw std::invalid_argument("--speculation must be off, auto, verify or lookup");
        }
        else if (argument == "--temperature" || argument == "--top-k" || argument == "--top-p" ||
                 argument == "--min-p" || argument == "--seed") {
            options.sampling_requested = true;
            if (argument == "--temperature") temperature = parse_number<float>(value(), "invalid temperature");
            else if (argument == "--top-k") options.sampling.top_k = core::parse_u32(value());
            else if (argument == "--top-p") options.sampling.top_p = parse_number<float>(value(), "invalid top-p");
            else if (argument == "--min-p") options.sampling.min_p = parse_number<float>(value(), "invalid min-p");
            else options.sampling.seed = parse_number<std::uint64_t>(value(), "invalid seed");
        }
        else if (family.load_tokenizer && argument == "--prompt") options.text = value();
        else if (family.render_messages && argument == "--messages") options.messages = value();
        else if (family.load_tokenizer && argument == "--tokenize") options.tokenize_file = value();
        else if (family.load_tokenizer && argument == "--tokenizer") options.tokenizer = value();
        else if (family.supports_packed_decode && argument == "--packed-decode") options.packed_decode = true;
        else if (family.supports_packed_decode && argument == "--grouped-decode") grouped_decode = true;
        else if (family.default_prefill_chunk && argument == "--prefill-chunk") {
            options.prefill_chunk = core::parse_u32(value());
            options.prefill_chunk_explicit = true;
            if (!options.prefill_chunk || options.prefill_chunk > family.max_context)
                throw std::invalid_argument("prefill chunk size must be within the context");
        } else if (argument.starts_with("--")) {
            if (!extra || !extra(argument, value))
                throw std::invalid_argument("unknown option " + std::string(argument));
        } else {
            options.prompt.push_back(core::parse_u32(argument));
        }
    }
    const unsigned inputs = static_cast<unsigned>(!options.prompt.empty()) +
        static_cast<unsigned>(options.text != nullptr) + static_cast<unsigned>(options.messages != nullptr) +
        static_cast<unsigned>(options.tokenize_file != nullptr);
    if (inputs > 1U) throw std::invalid_argument("choose one of token ids, --prompt, --messages or --tokenize");
    if (options.tokenize_file && (options.generate_requested || options.logits || options.sampling_requested))
        throw std::invalid_argument("--tokenize takes no --generate, --logits or sampling options");
    if (options.tokenizer && !options.text && !options.messages && !options.tokenize_file)
        throw std::invalid_argument("--tokenizer requires --prompt, --messages or --tokenize");
    options.packed_decode = resolve_decode_form(options.packed_decode, grouped_decode, family.packed_decode_default);
    options.sampling.temperature = temperature.value_or(omitted_token_sampling_temperature(
        options.sampling.top_k, options.sampling.top_p, options.sampling.min_p));
    validate_token_sampling_policy(options.sampling);
    if ((options.speculation == CliSpeculation::verify || options.speculation == CliSpeculation::lookup) &&
        token_sampling_enabled(options.sampling) && !family.sampled_speculation)
        throw std::invalid_argument("--speculation " + std::string(speculation_name(options.speculation)) +
                                    " requires greedy decoding");
    return options;
}

void run_serial_generation(const std::string& checkpoint, SerialGenerationOptions options,
                           const SerialGenerationFamily& family, std::ostream& out,
                           const std::function<void()>& on_loaded) {
    std::optional<LogitsOutput> output;  // opened first: every input below is checked against the opened file
    if (options.logits) output.emplace(options.logits);
    const auto protect_asset = [&](const fs::path& asset, const char* error) {
        if (output) output->refuse(asset, error);
    };
    protect_asset(checkpoint, "logits output must differ from the checkpoint");
    std::shared_ptr<const BpeTokenizer> tokenizer;
    std::unique_ptr<SerialMediaInput> media;            // outlives the prefill: it owns the interned rows' adapter
    std::shared_ptr<const session::EmbedTable> embeds;  // rows of the virtual ids of an image prompt
    if (options.text || options.messages || options.tokenize_file) {
        const fs::path asset = options.tokenizer ? fs::path(options.tokenizer) :
            fs::path(checkpoint).parent_path() / "tokenizer.json";
        protect_asset(asset, "logits output must differ from the tokenizer");
        tokenizer = family.load_tokenizer(asset.string());
        if (options.tokenize_file) {
            for (const auto id : tokenizer->encode(read_text(options.tokenize_file), false)) out << id << '\n';
            if (!out) throw std::runtime_error("could not write token IDs");
            return;
        }
        if (options.messages) {
            const auto input = read_text_input(options.messages);
            if (output) output->refuse(input.key, "logits output must differ from the messages file");
            const auto& body = input.text;
            if (family.media && family.render_media_messages) {
                auto prompt = render_media_prompt(family, checkpoint, *tokenizer, body, options.generate, media,
                                                  output ? &*output : nullptr);
                options.prompt = std::move(prompt.tokens);
                embeds = std::move(prompt.table);
            } else {
                options.prompt = tokenizer->encode(family.render_messages(body), true);
            }
        } else {
            options.prompt = tokenizer->encode(options.text, false);
        }
    }
    if (options.prompt.empty() || options.prompt.size() > family.max_context ||
        options.generate > family.max_context - options.prompt.size())
        throw std::invalid_argument("prompt and generation must fit the context");
    if (!embeds) {
        core::validate_token_ids(options.prompt.data(), options.prompt.size(), family.input_vocabulary);
    } else {
        for (const auto id : options.prompt)
            if (!session::is_embed_id(id)) core::validate_token_ids(&id, 1U, family.input_vocabulary);
    }
    // Drafters read prompt ids; a prompt with virtual ids runs serial decode (same ids, no drafting).
    if (embeds) options.speculation = CliSpeculation::off;
    const auto prompt_size = static_cast<std::uint32_t>(options.prompt.size());
    const auto positions = prompt_size + options.generate;
    const bool greedy = !token_sampling_enabled(options.sampling);
    const bool rows_decidable = greedy || family.sampled_speculation;
    const auto verify_rows = options.speculation != CliSpeculation::off && rows_decidable && options.generate > 1U
        ? options.generate - 1U : 0U;
    const auto load_start = Clock::now();
    const auto chunk = std::min(options.prefill_chunk, prompt_size);
    const auto fallback = std::min(family.prefill_chunk_fallback, prompt_size);
    std::unique_ptr<SerialGenerationDecoder> decoder;
    const auto load = [&] {
        decoder = load_with_chunk_fallback(chunk, options.prefill_chunk_explicit, fallback, std::cerr,
                                           [&](std::uint32_t rows) {
            return family.load({checkpoint, positions, rows, options.packed_decode, verify_rows,
                                verify_rows ? options.speculation : CliSpeculation::off,
                                prefill_tail_rows(prompt_size, rows, family.prefill_tail_divisor), prompt_size});
        });
    };
    if (media) media->load_checkpoint(checkpoint, load);  // the model reads the file the image rows came from
    else load();
    if (family.report_load_timing) session::write_load_timing(out, load_start);
    if (on_loaded) on_loaded();

    std::optional<TokenSampler> sampler;
    const auto& sampler_options = options.sampler_options;
    if (!greedy) sampler.emplace(options.sampling, sampler_options);
    std::vector<float> logits((options.logits || sampler) ? family.output_vocabulary : 0U);
    const auto prefill_start = Clock::now();
    auto next = embeds ? decoder->prefill_embedded(options.prompt, embeds, logits) : decoder->prefill(options.prompt, logits);
    if (sampler) next = sampler->decide(logits, prompt_size, nullptr).token;
    const auto prefill_ms = session::cli_milliseconds(Clock::now() - prefill_start);
    if (output) output->write(logits);
    if (!sampler) logits.clear();

    out << "next_token " << next << '\n';
    if (options.generate) out << "generated" << std::flush;
    const SerialTokenContinuationRequest request{
        next, prompt_size, family.output_vocabulary, positions, options.generate, family.eos};
    Clock::duration emit_time{};
    const auto emit = [&](std::uint32_t token) {
        const auto start = Clock::now();
        out << ' ' << token << std::flush;
        emit_time += Clock::now() - start;
    };
    const auto max_rows = verify_rows > 1U
        ? std::min(verify_rows, decoder->max_verify_rows(options.speculation)) : 0U;
    auto* verify = max_rows > 1U ? decoder->verify_target() : nullptr;
    // A sampled request verifies rows with the serial sampler's draw at each position (same policy and request
    // identity as the serial path below), so the ids equal serial sampled decode.
    const auto row_rule = greedy ? spec::RowRule{} : spec::RowRule(options.sampling, sampler_options.request_identity);
    if (verify_rows > 1U) {
        const exec::DecodeCapabilities offered{.verifier = verify != nullptr, .sampling = true,
            .sampled_verifier = !greedy && verify && verify->supports_row_rule(true)};
        if (!offered.provides({.verifier = true, .sampled_verifier = !greedy})) {
            if (options.speculation != CliSpeculation::automatic)
                throw std::invalid_argument(verify ? "this target cannot verify sampled rows" :
                                                     "this target cannot verify a window");
            verify = nullptr;
        }
    }
    std::optional<spec::BlockDrafter> drafter;
    // Opened (file read) before the decode clock; stored after it.
    auto cost_table = verify ? open_verify_cost_table(family, checkpoint, *decoder, options.speculation, !greedy,
                                                      max_rows, prompt_size)
                             : std::nullopt;
    // An untimed bucket seeds the curve shape from the nearest timed bucket of the same key (read only).
    const auto cost_prior = cost_table ? spec::open_nearest_verify_cost_prior(*cost_table, max_rows) : std::nullopt;
    std::uint32_t cost_seeded = 0, cost_recorded = 0, chain_k = 0;
    bool cost_stored = false, prices_frozen = false;
    SerialGreedyRequestResult generated;
    bool device_sampled = false;  // the serial path's sampled rows decided on the device (decode_sampled)
    const auto decode_start = Clock::now();
    if (verify) {
        // The shared block setup (strix/spec/block_setup.hpp); this command line adds its request modes, the frozen
        // prices of its cost table and the diagnostic's equal prices.
        spec::BlockSetup setup;
        setup.max_rows = max_rows;
        setup.probe_loss = family.speculation_probe_loss;
        if (options.speculation != CliSpeculation::verify)
            setup.tune = [&](spec::BlockConfig& config) { decoder->tune_speculation(config); };
        // Several drafters (greedy auto): one BlockDrafter per arm under the shared per-regime selector.
        auto arms = options.speculation == CliSpeculation::automatic && !row_rule.sampled()
            ? decoder->chain_arms(next, prompt_size) : std::vector<SerialGenerationDecoder::ChainArm>{};
        if (arms.size() < 2U) arms.clear();
        spec::ChainDrafter* arm_chain = nullptr;
        setup.chain = [&](spec::BlockConfig& config) -> spec::ChainDrafter* {
            config.force_lookup = options.speculation == CliSpeculation::lookup;
            if (!arms.empty()) return arm_chain;
            return options.speculation == CliSpeculation::automatic ? decoder->chain_drafter(next, prompt_size) : nullptr;
        };
        setup.prices = [&](spec::BlockConfig& config, spec::ChainDrafter* chain) {
            chain_k = chain && config.chain ? std::min(config.chain_max_k, max_rows > 1U ? max_rows - 1U : 0U) : 0U;
            // Frozen prices (strix/spec/verify_cost_table.hpp): a table that prices every window of this run (and a
            // chain depth) is the whole cost model, so the row choices are a function of the table and the acceptance
            // history, not of this run's host clock. Otherwise auto learns verify + commit and draft walls in this run.
            // The diagnostic keeps its original equal prices.
            prices_frozen = cost_table && spec::table_prices_run(*cost_table, config.arbiter.windows, max_rows, chain_k);
            config.fixed_costs = options.speculation == CliSpeculation::verify || prices_frozen;
            config.frozen_prices = prices_frozen;
            if (config.fixed_costs) spec::set_context_cost(config.arbiter, "off");
            // Frozen prices also price the exploration (ArbiterConfig::priced_exploration: probes never buy a window
            // that cannot pay and never move the arm hysteresis) and stop a drafter whose realized cycles lose at those
            // prices (ArbiterConfig::cycle_half_life_tokens).
            if (prices_frozen) spec::set_frozen_price_rules(config.arbiter);
        };
        setup.row_rule = row_rule;
        setup.seed = [&](spec::BlockDrafter& built) {
            if (options.speculation == CliSpeculation::verify)
                for (std::uint32_t row = 1; row <= max_rows; ++row) built.seed_verify_ms(row, 1.0);
            else
                decoder->seed_speculation(built);
        };
        // Walls measured by earlier runs win over the engine's declared seeds (applied after them).
        const auto seed_costs = [&](spec::BlockDrafter& built) {
            if (!cost_table) return;
            const auto& seeds = cost_prior ? *cost_prior : *cost_table;
            cost_seeded = spec::seed_verify_costs(seeds, built, max_rows);
            if (chain_k > 0U) (void)spec::seed_draft_costs(seeds, built, chain_k);
        };
        std::optional<RoundLog> round_log;
        if (const int rounds = envcfg::integer("STRIX_SPEC_ROUND_LOG", 0); rounds > 0)
            round_log.emplace(std::cerr, static_cast<std::uint32_t>(rounds));
        if (arms.empty()) {
            (void)spec::make_block_drafter(drafter, setup);
            seed_costs(*drafter);
            generated = spec::execute_block_token_continuation(request, options.prompt, *drafter, *verify, max_rows, emit,
                                                               row_rule, round_log ? &*round_log : nullptr);
        } else {
            std::deque<std::optional<spec::BlockDrafter>> built(arms.size());
            spec::ArmSelector selector;
            for (std::size_t a = 0; a < arms.size(); ++a) {
                arm_chain = arms[a].chain;
                auto& arm = spec::make_block_drafter(built[a], setup);
                seed_costs(arm);
                (void)selector.add(arm, arms[a].name, arms[a].probe_min_tokens, {}, arms[a].wins_ties);
            }
            generated = spec::execute_block_token_continuation(request, options.prompt, selector, *verify, max_rows, emit,
                                                               row_rule, round_log ? &*round_log : nullptr);
            // The selector's receipt beside the shared speculation stats: passes / tokens / probes / wall per arm.
            out << "strix-speculation-selector/v1 arms " << selector.size();
            for (std::size_t a = 0; a < selector.size(); ++a) {
                const auto& st = selector.stats(a);
                out << " | " << st.name << " passes " << st.passes << " tokens " << st.tokens << " probes " << st.probes
                    << " ms " << static_cast<std::uint64_t>(st.ms);
            }
            if (selector.active() < selector.size()) out << " | active " << selector.stats(selector.active()).name;
            out << '\n';
        }
    } else {
        // A decoder that samples on the device decides every sampled row there: the same ids for the request's seed,
        // without the logits row's host copy and host transform (STRIX_SAMPLE_PATH=host keeps the host path).
        device_sampled = sampler && device_token_sampling_enabled(options.sampling) && decoder->samples_on_device();
        generated = execute_serial_token_continuation(request, [&](std::uint32_t input, std::uint64_t position) {
            if (device_sampled) return decoder->decode_sampled(input, *sampler, position + 1U);
            const auto selected = decoder->decode(input, logits);
            return sampler ? sampler->decide(logits, position + 1U, nullptr).token : selected;
        }, emit);
    }
    const auto decode_ms = session::cli_milliseconds(Clock::now() - decode_start - emit_time);
    const auto decode_tok_s = generated.steps_executed && decode_ms > 0.0
        ? 1000.0 * static_cast<double>(generated.steps_executed) / decode_ms : 0.0;
    if (cost_table && drafter) {
        cost_recorded = spec::record_verify_costs(*cost_table, drafter->stats());
        if (chain_k > 0U) cost_recorded += spec::record_draft_costs(*cost_table, *drafter, chain_k);
        cost_stored = cost_table->store();
    }
    if (options.generate) out << '\n';
    out << "timing prompt_tokens " << prompt_size << " prefill_ms " << prefill_ms
        << " decode_tokens " << generated.steps_executed << " decode_ms "
        << (generated.steps_executed ? decode_ms : 0.0) << " decode_tok_s " << decode_tok_s << '\n';
    if (sampler) out << "sampling path " << (device_sampled ? "device" : "host") << '\n';
    if (options.speculation != CliSpeculation::off) {
        const spec::SpecStats empty;
        const auto& stats = drafter ? drafter->stats() : empty;
        const auto sum = [](const auto& counts) { return std::accumulate(counts.begin(), counts.end(), std::uint64_t{}); };
        const auto accepted = sum(stats.source_accepted);
        const auto drafted = sum(stats.source_proposed);
        const auto draft_passes = stats.passes - stats.source_passes[static_cast<std::size_t>(spec::Source::serial)];
        out << "speculation mode " << speculation_name(options.speculation)
            << " acceptance " << accepted << '/' << drafted << " draft_passes " << draft_passes
            << " mean_accepted_length "
            << (draft_passes ? static_cast<double>(accepted) / static_cast<double>(draft_passes) : 0.0) << '\n';
        if (drafter) out << stats.line() << '\n';
        if (cost_table)
            out << "speculation cost-table " << cost_table_load_name(cost_table->load_status()) << " seeded "
                << cost_seeded << " recorded " << cost_recorded << " stored " << (cost_stored ? 1 : 0) << " file "
                << cost_table->file().filename().string() << " prices " << (prices_frozen ? "frozen" : "learned")
                << (cost_prior ? " prior-bucket " + std::to_string(cost_prior->key().context_bucket) : std::string{})
                << '\n';
    }
    if (tokenizer && !generated.generated_tokens.empty())
        out << "text " << tokenizer->decode(generated.generated_tokens, true) << '\n';
    if (output) output->publish();  // the run succeeded: the logits dump becomes visible under its name
}

std::uint32_t SerialGenerationDecoder::prefill_embedded(std::span<const std::uint32_t> ids,
                                                       const std::shared_ptr<const session::EmbedTable>& table,
                                                       std::span<float> logits) {
    if (table) throw std::invalid_argument("this family takes no provided embedding rows");
    return prefill(ids, logits);
}

std::uint32_t SerialGenerationDecoder::decode_sampled(std::uint32_t, TokenSampler&, std::uint64_t) {
    throw std::logic_error("this decoder does not sample on the device");
}

int run_serial_generation_cli(int argc, char** argv, const SerialGenerationFamily& family,
                              std::ostream& out, std::ostream& error) {
    if (argc < 3) {
        error << "usage: " << family.executable << ' ' << family.checkpoint_argument;
        if (family.load_tokenizer) error << " (TOKEN [TOKEN ...] | --prompt TEXT [--tokenizer FILE])";
        else error << " TOKEN [TOKEN ...]";
        if (family.default_prefill_chunk) error << " [--prefill-chunk N]";
        if (family.supports_packed_decode) error << " [--packed-decode]";
        if (family.supports_speculation) error << " [--speculation " << speculation_modes << "]";
        error << " [--generate N] [--logits FILE] [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]\n";
        return 2;
    }
    try {
        run_serial_generation(argv[1], parse_serial_generation_cli(argc, argv, family), family, out);
    } catch (const std::exception& failure) {
        error << family.diagnostic_name << ": " << failure.what() << '\n';
        return 1;
    }
    return 0;
}

}  // namespace strix::serve
