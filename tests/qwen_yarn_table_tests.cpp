#include "strix/qwen_yarn_table.hpp"
#include "strix/model_content_hash.hpp"
#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F&& fn) {
    try { fn(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid table/request accepted");
}
constexpr std::array<std::uint32_t, 14> positions{0,1,8191,8192,32767,65535,131071,
    196607,262143,262144,524287,524288,999999,1048575};
// Independent fp64 evaluation of u=10**(-7*i/32), r=clamp((i-14)/8),
// w=u*(1-r*(1-1/f)). These retain more digits than the display table.
constexpr std::array<double,32> expected2{1,
    0.60429639023813286,
    0.36517412725483772,
    0.22067340690845899,
    0.1333521432163324,
    0.080584218776148187,
    0.048696752516586311,
    0.029427271762092817,
    0.017782794100389229,
    0.010746078283213174,
    0.006493816315762113,
    0.0039241897584845363,
    0.0023713737056616554,
    0.0014330125702369627,
    0.00086596432336006539,
    0.00049059292001390131,
    0.00027669929526473317,
    0.0001552649292163483,
    8.6608648851709363e-05,
    4.7975852709115815e-05,
    2.635603146428639e-05,
    1.4334169207383824e-05,
    7.6996326302974593e-06,
    4.6528602046484952e-06,
    2.8117066259517456e-06,
    1.6991041644712796e-06,
    1.026762513228573e-06,
    6.2046888037585976e-07,
    3.7494710466622792e-07,
    2.265791818800409e-07,
    1.3692098171321807e-07,
    8.2740854997159068e-08};
constexpr std::array<double,32> expected4{1,
    0.60429639023813286,
    0.36517412725483772,
    0.22067340690845899,
    0.1333521432163324,
    0.080584218776148187,
    0.048696752516586311,
    0.029427271762092817,
    0.017782794100389229,
    0.010746078283213174,
    0.006493816315762113,
    0.0039241897584845363,
    0.0023713737056616554,
    0.0014330125702369627,
    0.00086596432336006539,
    0.00047423982268010459,
    0.00025693505988868084,
    0.0001373497450760004,
    7.2173874043091133e-05,
    3.7072249820680398e-05,
    1.8449222025000472e-05,
    8.7597700711790026e-06,
    3.8498163151487296e-06,
    2.3264301023242476e-06,
    1.4058533129758728e-06,
    8.4955208223563982e-07,
    5.1338125661428652e-07,
    3.1023444018792988e-07,
    1.8747355233311396e-07,
    1.1328959094002045e-07,
    6.8460490856609035e-08,
    4.1370427498579534e-08};
bool close(double a, double b) { return std::abs(a-b) <= 1e-15*std::abs(b); }
void coefficients(const strix::RotaryDescriptor& d, std::uint32_t p) {
    const auto row = strix::qwen_yarn_row(d, p);
    const auto f = strix::qwen_yarn_frequencies(d);
    for (std::size_t i=0; i<32; ++i) {
        const double angle = static_cast<double>(p)*f.omega[i];
        const double a = d.attention_factor();
        for (auto values : {std::pair{row[i].cosine, a*std::cos(angle)},
                            std::pair{row[i].sine, a*std::sin(angle)}}) {
            const auto [value, exact] = values;
            require(std::isfinite(value), "finite coefficient");
            const float neighbor = std::nextafter(value, exact > value ?
                std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity());
            require(std::abs(static_cast<double>(value)-exact) <=
                0.5*std::abs(static_cast<double>(neighbor)-value), "one RNE f32 conversion");
        }
    }
}
void emit(const char* path) {
    std::ofstream out(path);
    out << std::setprecision(17);
    for (const auto factor : {2U,4U}) {
        const auto d = strix::RotaryDescriptor::yarn(factor);
        const auto f = strix::qwen_yarn_frequencies(d);
        for (auto p : positions) if (p<d.effective_max()) {
            const auto r = strix::qwen_yarn_row(d,p);
            for (std::size_t i=0; i<32; ++i)
                out << factor << '\t' << p << '\t' << i << '\t' << f.omega[i] << '\t'
                    << std::bit_cast<std::uint32_t>(r[i].cosine) << '\t'
                    << std::bit_cast<std::uint32_t>(r[i].sine) << '\n';
        }
    }
    require(static_cast<bool>(out), "sample write");
}
void tables() {
    using namespace strix;
    const auto d = RotaryDescriptor::yarn(4);
    std::ostringstream out(std::ios::binary);
    write_qwen_yarn_table(out,d,8191,8200,3);
    const auto bytes = out.str();
    auto validate = [&](const std::string& value) {
        std::istringstream input(value,std::ios::binary);
        validate_qwen_yarn_table(input,d,8191,8200);
    };
    validate(bytes);
    const auto prefix = bytes.size()-9U*256U;
    // Every metadata and digest byte is checked, including schema, endian,
    // theta/A bits, dimensions, sections, factor, bounds and version.
    for (std::size_t i=0; i<prefix; ++i) {
        auto bad=bytes; bad[i]^=1; rejects([&]{validate(bad);});
    }
    auto bad=bytes; bad.back()^=1; rejects([&]{validate(bad);});
    for (std::size_t i=0; i<bytes.size(); ++i)
        rejects([&]{validate(bytes.substr(0,i));});
    rejects([&]{validate(bytes+"x");});
    rejects([&]{std::istringstream in(bytes); validate_qwen_yarn_table(in,RotaryDescriptor::yarn(2),8191,8200);});
    rejects([&]{std::istringstream in(bytes); validate_qwen_yarn_table(in,d,8192,8201);});
    const auto payload = std::as_bytes(std::span(bytes.data()+prefix,bytes.size()-prefix));
    const auto digest = sha256_bytes(payload);
    for (std::size_t i=0; i<32; ++i)
        require(static_cast<unsigned char>(bytes[prefix-32+i])==digest.bytes[i], "payload SHA256");
    auto nonfinite=bytes;
    nonfinite[prefix]=0; nonfinite[prefix+1]=0;
    nonfinite[prefix+2]=static_cast<char>(0xc0); nonfinite[prefix+3]=0x7f;
    const auto forged_digest=sha256_bytes(std::as_bytes(std::span(nonfinite.data()+prefix,nonfinite.size()-prefix)));
    for (std::size_t i=0; i<32; ++i) nonfinite[prefix-32+i]=static_cast<char>(forged_digest.bytes[i]);
    rejects([&]{validate(nonfinite);});
    // First float2 pair is serialized little-endian, independent of host byte order.
    const auto first = qwen_yarn_row(d,8191);
    const auto bits = std::bit_cast<std::uint32_t>(first[0].cosine);
    for (unsigned i=0; i<4; ++i)
        require(static_cast<unsigned char>(bytes[prefix+i])==((bits>>(8*i))&255U), "little endian float2");
    std::ostringstream other;
    write_qwen_yarn_table(other,d,8191,8200,7);
    require(other.str()==bytes,"chunk boundaries preserve serialized bytes");
    std::ostringstream empty;
    write_qwen_yarn_table(empty,d,10,10);
    std::istringstream empty_in(empty.str()); validate_qwen_yarn_table(empty_in,d,10,10);
    std::uint32_t next=262141;
    generate_qwen_yarn_rows(d,next,262150,[&](auto first_row, auto rows){
        require(first_row==next && rows.size()<=3,"bounded contiguous chunks across original maximum");
        next += static_cast<std::uint32_t>(rows.size());
    },3);
    require(next==262150,"all rows delivered");
    rejects([&]{generate_qwen_yarn_rows(d,3,2,[](auto,auto){});});
    rejects([&]{generate_qwen_yarn_rows(d,0,1,[](auto,auto){},0);});
    rejects([&]{generate_qwen_yarn_rows(d,0,1,[](auto,auto){},4097);});
    rejects([&]{generate_qwen_yarn_rows(d,1048575,1048577,[](auto,auto){});});
    bool called=false;
    const auto native=RotaryDescriptor::native();
    generate_qwen_yarn_rows(native,0,1048576,[&](auto,auto){called=true;});
    std::ostringstream native_out;
    write_qwen_yarn_table(native_out,native,0,262144);
    require(!called && native_out.str().empty(),"native generates nothing");
    std::istringstream native_in; validate_qwen_yarn_table(native_in,native,0,262144);
    rejects([&]{std::istringstream in(bytes);validate_qwen_yarn_table(in,native,0,262144);});
    rejects([&]{(void)qwen_yarn_row(native,0);});
}
void pairing() {
    const auto d=strix::RotaryDescriptor::yarn(4);
    const auto cs=strix::qwen_yarn_row(d,1);
    std::array<double,128> x{}, y{};
    for (std::size_t i=0; i<x.size(); ++i) x[i]=static_cast<double>(i+1);
    y=x;
    for (std::size_t i=0; i<32; ++i) {
        y[i]=x[i]*cs[i].cosine-x[i+32]*cs[i].sine;
        y[i+32]=x[i+32]*cs[i].cosine+x[i]*cs[i].sine;
    }
    require(y[0] != x[0]*cs[0].cosine-x[1]*cs[0].sine,"rotate_half differs from adjacent pairing");
    require(y[31] != x[31] && y[63] != x[63],"index head rotates all first 64 dimensions, not 32");
    require(std::equal(y.begin()+64,y.end(),x.begin()+64),"index nonrotary tail unchanged");
    // Mistaken d=32 changes frequency i=1 by orders of magnitude.
    require(!close(strix::qwen_yarn_frequencies(d).omega[1],std::pow(1e7,-2.0/32.0)),"index d64 witness");
}
}
int main(int argc,char** argv) {
    if (argc==3 && std::string_view(argv[1])=="--emit-samples") { emit(argv[2]); return 0; }
    for (const auto factor : {2U,4U}) {
        const auto d=strix::RotaryDescriptor::yarn(factor);
        const auto f=strix::qwen_yarn_frequencies(d);
        require(f.low==14 && f.high==22,"floor/ceil correction bounds");
        const auto& expected=factor==2?expected2:expected4;
        for (std::size_t i=0; i<32; ++i) require(close(f.omega[i],expected[i]),"all 32 Appendix frequencies");
        const auto zero=strix::qwen_yarn_row(d,0);
        for (const auto& c : zero) require(c.cosine==static_cast<float>(d.attention_factor()) &&
            std::bit_cast<std::uint32_t>(c.sine)==0,"p0 scales every pair by A");
        for (auto p : positions) {
            if (p<d.effective_max()) coefficients(d,p);
            else rejects([&]{(void)strix::qwen_yarn_row(d,p);});
        }
        rejects([&]{(void)strix::qwen_yarn_row(d,d.effective_max());});
        std::mt19937 rng(0x51A7);
        for (int i=0; i<4096; ++i) coefficients(d,static_cast<std::uint32_t>(rng()%d.effective_max()));
    }
    const int previous=std::fegetround();
    require(std::fesetround(FE_DOWNWARD)==0,"set rounding mode");
    rejects([]{(void)strix::qwen_yarn_row(strix::RotaryDescriptor::yarn(2),1);});
    require(std::fesetround(previous)==0,"restore rounding mode");
    tables(); pairing();
    std::cout << "YaRN frequencies, boundary/random rows, RNE, pairing, bounded serialization and corruption tests passed\n";
}
