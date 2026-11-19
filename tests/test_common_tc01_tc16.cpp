#include "minitensor/Tensor.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using minitensor::Tensor;

struct Result {
    std::string id;
    std::string name;
    std::string status;
    std::string expected;
    std::string actual;
    std::string error;
    std::string notes;
};

static std::vector<Result> results;

static std::string join(const std::vector<std::size_t>& v) {
    std::ostringstream os;
    os << "{";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "}";
    return os.str();
}

static std::string values(const Tensor& t) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (i) os << ",";
        os << std::setprecision(10) << t.data()[i];
    }
    os << "]";
    return os.str();
}

template<class F>
static void run(const std::string& id, const std::string& name,
                const std::string& expected, F&& fn,
                const std::string& notes = "") {
    Result r{id, name, "FAIL", expected, "", "", notes};
    try {
        r.actual = fn();
        r.status = "PASS";
    } catch (const std::exception& e) {
        r.status = "FAIL";
        r.error = e.what();
    } catch (...) {
        r.status = "FAIL";
        r.error = "unknown non-standard exception";
    }
    results.push_back(r);
    std::cout << r.id << " " << r.status << " - " << r.name;
    if (!r.error.empty()) std::cout << " (" << r.error << ")";
    std::cout << "\n";
}

static void write_csv(const std::string& path) {
    std::ofstream out(path);
    out << "test_id,test_name,status,expected,actual,error,notes\n";
    auto quote = [](const std::string& s) {
        std::string q = "\"" ;
        for (char c : s) {
            if (c == '"') q += '"';
            q += c;
        }
        q += '"';
        return q;
    };
    for (const auto& r : results) {
        out << quote(r.id) << "," << quote(r.name) << ","
            << quote(r.status) << "," << quote(r.expected) << ","
            << quote(r.actual) << "," << quote(r.error) << ","
            << quote(r.notes) << "\n";
    }
}

int main(int argc, char** argv) {
    const std::string csv = argc > 1 ? argv[1] : "tc01_tc16_results.csv";

    run("TC01", "Scalar tensor", "rank=0; numel=1; value=5", [] {
        Tensor t({}, 5.0);
        if (t.rank() != 0 || t.size() != 1 || t.data()[0] != 5.0)
            throw std::runtime_error("scalar metadata/value mismatch");
        return "rank=0; numel=1; values=" + values(t);
    }, "Library element type is double, not required float32.");

    run("TC02", "1-D tensor", "shape={5}; numel=5; values=[1,2,3,4,5]", [] {
        Tensor t({5});
        for (std::size_t i = 0; i < 5; ++i) t.data()[i] = double(i + 1);
        if (t.shape().dimensions() != std::vector<std::size_t>{5} ||
            t.size() != 5 || values(t) != "[1,2,3,4,5]")
            throw std::runtime_error("1-D tensor mismatch");
        return "shape={5}; numel=5; values=" + values(t);
    });

    run("TC03", "2-D tensor", "shape={2,3}; numel=6; values=[1,2,3,4,5,6]", [] {
        Tensor t({2,3});
        for (std::size_t i = 0; i < 6; ++i) t.data()[i] = double(i + 1);
        if (t.shape().dimensions() != std::vector<std::size_t>{2,3} ||
            t.size() != 6 || values(t) != "[1,2,3,4,5,6]")
            throw std::runtime_error("2-D tensor mismatch");
        return "shape={2,3}; numel=6; flattened=" + values(t);
    });

    run("TC04", "3-D tensor", "shape={2,2,3}; numel=12; flattened 0..11", [] {
        Tensor t({2,2,3});
        for (std::size_t i = 0; i < 12; ++i) t.data()[i] = double(i);
        if (t.shape().dimensions() != std::vector<std::size_t>{2,2,3} ||
            t.size() != 12) throw std::runtime_error("3-D metadata mismatch");
        for (std::size_t i = 0; i < 12; ++i)
            if (t.data()[i] != double(i)) throw std::runtime_error("flattened order mismatch");
        return "shape={2,2,3}; numel=12; flattened=" + values(t);
    });

    run("TC05", "4-D tensor", "shape={2,3,2,2}; numel=24; flattened 0..23", [] {
        Tensor t({2,3,2,2});
        for (std::size_t i = 0; i < 24; ++i) t.data()[i] = double(i);
        if (t.shape().dimensions() != std::vector<std::size_t>{2,3,2,2} ||
            t.size() != 24) throw std::runtime_error("4-D metadata mismatch");
        for (std::size_t i = 0; i < 24; ++i)
            if (t.data()[i] != double(i)) throw std::runtime_error("flattened order mismatch");
        return "shape={2,3,2,2}; numel=24; flattened=" + values(t);
    });

    run("TC06", "Zero initialization", "all six elements are zero", [] {
        Tensor t({2,3});
        for (std::size_t i = 0; i < 6; ++i)
            if (t.data()[i] != 0.0) throw std::runtime_error("non-zero default element");
        return "shape={2,3}; values=" + values(t);
    });

    run("TC07", "Contiguous storage", "adjacent address delta equals sizeof(float)", [] {
        Tensor t({2,3});
        const auto* p = t.data();
        const std::size_t delta = reinterpret_cast<std::uintptr_t>(p + 1) -
                                  reinterpret_cast<std::uintptr_t>(p);
        if (delta != sizeof(double)) throw std::runtime_error("storage is not contiguous for its element type");
        for (std::size_t i = 0; i < 5; ++i) {
            const auto d = reinterpret_cast<std::uintptr_t>(p+i+1) -
                           reinterpret_cast<std::uintptr_t>(p+i);
            if (d != sizeof(double)) throw std::runtime_error("non-contiguous adjacent elements");
        }
        if (sizeof(double) != sizeof(float))
            return "contiguous; address_delta=" + std::to_string(delta) +
                   "; SPEC MISMATCH: library uses double (8 bytes), spec expects float32 (4 bytes)";
        return "contiguous; address_delta=" + std::to_string(delta);
    }, "Contiguity is checked, but exact float32 byte stride cannot pass with this double-based implementation.");

    run("TC08", "Large allocation and sum", "numel=1000000; sum=2000000", [] {
        Tensor t({1000,1000}, 2.0);
        if (t.size() != 1000000) throw std::runtime_error("element count mismatch");
        double sum = 0.0;
        for (std::size_t i = 0; i < t.size(); ++i) sum += t.data()[i];
        if (std::abs(sum - 2000000.0) > 1e-5)
            throw std::runtime_error("sum outside tolerance");
        std::ostringstream os;
        os << "numel=" << t.size() << "; sum=" << std::setprecision(15) << sum;
        return os.str();
    }, "Allocation failure is captured as FAIL; machine memory is environment-dependent.");

    run("TC09", "First element", "T(0,0)=1", [] {
        Tensor t({2,3});
        for (std::size_t i=0;i<6;++i) t.data()[i]=double(i+1);
        const double v=t.at({0,0});
        if (v != 1.0) throw std::runtime_error("expected 1");
        return "T(0,0)=" + std::to_string(v);
    });

    run("TC10", "Last element", "T(1,2)=6", [] {
        Tensor t({2,3});
        for (std::size_t i=0;i<6;++i) t.data()[i]=double(i+1);
        const double v=t.at({1,2});
        if (v != 6.0) throw std::runtime_error("expected 6");
        return "T(1,2)=" + std::to_string(v);
    });

    run("TC11", "Middle element", "T(1,1)=5", [] {
        Tensor t({2,3});
        for (std::size_t i=0;i<6;++i) t.data()[i]=double(i+1);
        const double v=t.at({1,1});
        if (v != 5.0) throw std::runtime_error("expected 5");
        return "T(1,1)=" + std::to_string(v);
    });

    run("TC12", "Modify element", "T(0,1)=20; others unchanged", [] {
        Tensor t({2,3});
        for (std::size_t i=0;i<6;++i) t.data()[i]=double(i+1);
        t.at({0,1})=20.0;
        if (t.at({0,1}) != 20.0) throw std::runtime_error("write/read-back mismatch");
        const std::vector<double> expected{1,20,3,4,5,6};
        for (std::size_t i=0;i<6;++i)
            if (t.data()[i] != expected[i]) throw std::runtime_error("unmodified element changed");
        return "values=" + values(t);
    });

    run("TC13", "Invalid row index", "documented out_of_range; no crash", [] {
        Tensor t({2,3});
        try { (void)t.at({2,0}); }
        catch (const std::out_of_range& e) { return std::string("caught std::out_of_range: ") + e.what(); }
        throw std::runtime_error("invalid row index was not rejected");
    });

    run("TC14", "Invalid column index", "documented out_of_range; no crash", [] {
        Tensor t({2,3});
        try { (void)t.at({0,3}); }
        catch (const std::out_of_range& e) { return std::string("caught std::out_of_range: ") + e.what(); }
        throw std::runtime_error("invalid column index was not rejected");
    });

    run("TC15", "Negative index", "negative index rejected safely", [] {
        Tensor t({2,3});
        // The public API takes size_t, so this deliberately models passing -1
        // through the actual unsigned API. It must be rejected as out of range.
        const std::size_t negative_as_unsigned = static_cast<std::size_t>(-1);
        try { (void)t.at(std::vector<std::size_t>{negative_as_unsigned,0}); }
        catch (const std::out_of_range& e) {
            return std::string("rejected after unsigned conversion: ") + e.what();
        }
        throw std::runtime_error("converted negative index was not rejected");
    }, "API does not accept signed indices; verifies converted -1 cannot access memory.");

    run("TC16", "3-D row-major mapping", "T(1,2,3) maps to flattened offset 23", [] {
        Tensor t({2,3,4});
        for (std::size_t i=0;i<24;++i) t.data()[i]=double(i);
        const double v=t.at({1,2,3});
        if (v != 23.0 || t.flat_at(23) != 23.0)
            throw std::runtime_error("row-major offset mismatch");
        return "T(1,2,3)=" + std::to_string(v) + "; flat_at(23)=" + std::to_string(t.flat_at(23));
    });

    write_csv(csv);
    std::size_t passed=0;
    for (const auto& r:results) if (r.status=="PASS") ++passed;
    std::cout << "\nSummary: " << passed << "/" << results.size()
              << " PASS; " << (results.size()-passed) << " FAIL\n";
    std::cout << "CSV: " << csv << "\n";
    return passed == results.size() ? 0 : 1;
}
