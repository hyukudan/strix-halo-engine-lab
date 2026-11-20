// Included only in the diagnostic test executable. Observes production buffers;
// it never reconstructs an intermediate or changes the production kernel.
#include "strix/cuda/qwen_quant.cuh"
#include <fstream>
#include <iomanip>
#include <sstream>

static void wfmt_record(const std::filesystem::path& directory, const std::string& name,
                        const char* arithmetic, u32 T,
                        const std::vector<std::pair<std::string,std::vector<u8>>>& sections) {
    std::ostringstream header;
    header << "{\"case\":\"" << name << "\",\"arithmetic\":\"" << arithmetic
           << "\",\"shape\":[" << T << ",2560,640],\"sections\":[";
    bool first=true;
    for (const auto& [key,data] : sections) {
        u64 hash=14695981039346656037ull;
        for (auto byte:data) { hash^=byte; hash*=1099511628211ull; }
        if (!first) header << ',';
        first=false;
        header << "{\"name\":\"" << key << "\",\"bytes\":" << std::dec << data.size()
               << ",\"fnv1a64\":\"" << std::hex << std::setw(16) << std::setfill('0') << hash << "\"}";
    }
    header << "]}";
    const auto text=header.str();
    std::filesystem::create_directories(directory);
    std::ofstream out(directory/(name+".wfmt"),std::ios::binary);
    out.write("WFMT\0\1\r\n",8);
    for(unsigned i=0;i<8;++i) out.put(char((u64(text.size())>>(8*i))&255));
    out.write(text.data(),std::streamsize(text.size()));
    for(const auto& [key,data]:sections) out.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));
    if(!out) throw std::runtime_error("MoE capture write failed");
}

static HostWeights wfmt_native_weights(u32 gt,u32 dt,std::mt19937& rng) {
    HostWeights w{gt,dt,3,{},{},{}};
    const auto fill=[&](std::vector<u8>& bytes,u32 type,u32 n,u32 k) {
        const auto geo=strix::cuda::quant::geometry(type);
        if(!geo.block_elements || k%geo.block_elements) throw std::invalid_argument("capture shape");
        bytes.resize(size_t(3)*n*(k/geo.block_elements)*geo.block_bytes);
        for(auto& x:bytes) x=u8(rng());
        // Small finite scales keep the graph's intermediate arithmetic in range.
        for(size_t off=0;off<bytes.size();off+=geo.block_bytes) {
            const u16 d=half_of(0x1p-10F);
            std::memcpy(bytes.data()+off,&d,2);
        }
    };
    fill(w.gate,gt,mo::ffn,mo::hidden); fill(w.up,gt,mo::ffn,mo::hidden);
    fill(w.down,dt,mo::hidden,mo::ffn);
    return w;
}

static void wfmt_capture_case(const std::filesystem::path& directory,const am::Tables& tables,
                              const HostWeights& w,u32 T,bool fast,std::mt19937& rng) {
    State state; init_state(state,T); synthetic_inputs(state,2,1|2|4,rng);
    Device d; upload_weights(d,w); upload_state(d,state,true);
    mo::enqueue(d.rows_view,T,mo::hidden,d.weights,tables,d.record,nullptr,fast);
    CK(cudaDeviceSynchronize());
    std::vector<std::pair<std::string,std::vector<u8>>> sections;
    auto append=[&](const char* key,const std::vector<u8>& data) {
        auto found=std::find_if(sections.begin(),sections.end(),[&](const auto& s){return s.first==key;});
        if(found==sections.end()) { sections.push_back({key,data}); return; }
        found->second.insert(found->second.end(),data.begin(),data.end());
    };
    append("source",w.gate); append("source",w.up); append("source",w.down);
    for(const int v:{V_input,V_routes,V_rw}) append("state",state.view[v]);
    for(const int v:{V_codes,V_dcodes}) append("activation_codes",snap(d,state,v));
    for(const int v:{V_deltas,V_sums,V_ddeltas,V_dsums}) append("activation_scales",snap(d,state,v));
    append("partials",snap(d,state,V_inter)); append("partials",snap(d,state,V_raw));
    append("output",snap(d,state,V_out)); append("output",snap(d,state,V_acc));
    append("state",snap(d,state,V_status));
    std::vector<u8> summary(128); CK(cudaMemcpy(summary.data(),d.summary,summary.size(),cudaMemcpyDeviceToHost));
    append("state",summary);
    std::sort(sections.begin(),sections.end(),[](const auto& a,const auto& b){return a.first<b.first;});
    const bool native=mo::native_types(d.weights);
    const std::string name="moe-g"+std::to_string(w.gate_type)+"-d"+std::to_string(w.down_type)+"-m"+std::to_string(T)+(fast?"-fast":"-exact");
    wfmt_record(directory,name,native?"cuda-native-iq-fp32@1":fast?"cuda-moe-mma@1":"cuda-four-chain-imma@1",T,sections);
    free_device(d);
}

static void wfmt_capture(const am::Tables& tables,const char* directory) {
    std::mt19937 rng(0x11ca);
    unsigned count=0;
    for(const auto gt:{mo::type_q4_k,mo::type_q5_k}) for(const auto dt:{mo::type_q5_1,mo::type_q8_0}) {
        const auto w=synthetic_weights(gt,dt,3,rng);
        for(const auto T:{1U,8U,9U,33U,65U}) for(const bool fast:{false,true}) {
            wfmt_capture_case(directory,tables,w,T,fast,rng); ++count;
        }
    }
    for(const auto gt:{18U,21U,22U,23U,42U}) for(const auto dt:{20U,42U}) {
        const auto w=wfmt_native_weights(gt,dt,rng);
        for(const auto T:{1U,9U}) { wfmt_capture_case(directory,tables,w,T,false,rng); ++count; }
    }
    std::printf("WFMT MoE capture: %u records\n",count);
}

// TURBO stage probe (diagnostic, mode turbo-probe): one MoE layer per GSQ type pair, the SAME inputs and weights through
// the production routes E (incumbent native FP32), F (gsq-iq-bf16mma), T (gsq-iq-i8mma-turbo-cuda-v1). Per arm the
// rel-RMS against E of u (gate/up projection alone), h (SwiGLU output, the down input), rawdots (down) and acc (routed
// sum). Inputs: Gaussian rows over a 2^-2..2^2 scale range plus four massive channels and 1 % x8 outliers (heavy-tailed
// groups, as the model's MoE input).
static void turbo_probe_inputs(State& s, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_real_distribution<float> uni(0.0F, 1.0F);
    for (u32 r = 0; r < s.T; ++r) {
        const float sc = std::exp2(-2.0F + 4.0F * uni(rng));
        u16* in = s.at<u16>(V_input, r);
        for (u32 i = 0; i < mo::hidden; ++i) in[i] = bf16_of(nd(rng) * sc * (uni(rng) < 0.01F ? 8.0F : 1.0F));
        for (const u32 i : {7U, 300U, 1201U, 2047U}) in[i] = bf16_of(sc * (40.0F + 20.0F * uni(rng)) * (i & 1U ? -1.0F : 1.0F));
        u32* routes = s.at<u32>(V_routes, r); float* rw = s.at<float>(V_rw, r);
        for (u32 k = 0; k < 10; ++k) { routes[k] = rng() % 3U; rw[k] = 0.02F + 0.28F * uni(rng); }
    }
}
struct TurboProbeOut { std::vector<u8> inter, raw, acc; u32 bad = 0; };
static TurboProbeOut turbo_probe_arm(const am::Tables& tables, const HostWeights& w, const State& s0, mo::ExpertArithmetic a) {
    State s = s0;
    Device d; upload_weights(d, w); upload_state(d, s, true);
    d.weights.arithmetic = a;
    mo::enqueue(d.rows_view, s.T, mo::hidden, d.weights, tables, d.record, nullptr, false);
    CK(cudaDeviceSynchronize());
    TurboProbeOut o{snap(d, s, V_inter), snap(d, s, V_raw), snap(d, s, V_acc), 0U};
    const auto st = snap(d, s, V_status);
    for (u32 r = 0; r < s.T; ++r) o.bad |= *reinterpret_cast<const u32*>(st.data() + size_t(r) * stride_of(V_status));
    u32 summary = 0; CK(cudaMemcpy(&summary, d.summary, 4, cudaMemcpyDeviceToHost)); o.bad |= summary;
    free_device(d);
    return o;
}
static int turbo_probe(const am::Tables& tables) {
    std::mt19937 rng(0x7b3);
    int bad = 0;
    std::printf("TURBO stage probe: rel-RMS vs E (incumbent native FP32) per stage, arms F | T\n");
    for (const auto gt : {18U, 21U, 22U, 23U, 42U}) for (const auto dt : {20U, 42U}) {
        const auto w = wfmt_native_weights(gt, dt, rng);
        for (const auto T : {9U, 300U}) {
            State s; init_state(s, T); turbo_probe_inputs(s, rng);
            const auto e = turbo_probe_arm(tables, w, s, mo::ExpertArithmetic::incumbent);
            std::printf("PROBE g%u d%u T%3u", gt, dt, T);
            for (const auto a : {mo::ExpertArithmetic::gsq_bf16_mma, mo::ExpertArithmetic::gsq_i8_turbo}) {
                const auto x = turbo_probe_arm(tables, w, s, a);
                double n[4] = {}, q[4] = {};  // u, h, raw, acc
                for (u32 r = 0; r < T; ++r) {
                    const auto* ei = reinterpret_cast<const float*>(e.inter.data() + size_t(r) * stride_of(V_inter));
                    const auto* xi = reinterpret_cast<const float*>(x.inter.data() + size_t(r) * stride_of(V_inter));
                    for (u32 k = 0; k < mo::top_k; ++k) for (u32 f = 0; f < mo::ffn; ++f) {
                        const size_t h = size_t(k) * 2U * mo::ffn + f, u = h + mo::ffn;
                        n[0] += double(ei[u]) * ei[u]; q[0] += (double(xi[u]) - ei[u]) * (double(xi[u]) - ei[u]);
                        n[1] += double(ei[h]) * ei[h]; q[1] += (double(xi[h]) - ei[h]) * (double(xi[h]) - ei[h]);
                    }
                    const auto* er = reinterpret_cast<const float*>(e.raw.data() + size_t(r) * stride_of(V_raw));
                    const auto* xr = reinterpret_cast<const float*>(x.raw.data() + size_t(r) * stride_of(V_raw));
                    for (u32 i = 0; i < mo::top_k * mo::hidden; ++i) { n[2] += double(er[i]) * er[i]; q[2] += (double(xr[i]) - er[i]) * (double(xr[i]) - er[i]); }
                    const auto* ea = reinterpret_cast<const float*>(e.acc.data() + size_t(r) * stride_of(V_acc));
                    const auto* xa = reinterpret_cast<const float*>(x.acc.data() + size_t(r) * stride_of(V_acc));
                    for (u32 i = 0; i < mo::hidden; ++i) { n[3] += double(ea[i]) * ea[i]; q[3] += (double(xa[i]) - ea[i]) * (double(xa[i]) - ea[i]); }
                }
                std::printf(" | u %.1e h %.1e raw %.1e acc %.1e%s", std::sqrt(q[0] / n[0]), std::sqrt(q[1] / n[1]), std::sqrt(q[2] / n[2]),
                            std::sqrt(q[3] / n[3]), x.bad ? " BAD" : "");
                bad |= x.bad ? 1 : 0;
            }
            std::printf("%s\n", e.bad ? " E-BAD" : "");
            bad |= e.bad ? 1 : 0;
        }
    }
    std::printf("TURBO stage probe %s\n", bad ? "STATUS-FLAGS" : "done");
    return bad;
}
