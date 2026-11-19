#include "minitensor/Ops.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace minitensor;

struct Result {
    std::string id, name, status, expected, actual, error;
};
static std::vector<Result> results;

static void run(const std::string& id, const std::string& name,
                const std::string& expected, const std::function<std::string()>& test) {
    Result r{id, name, "FAIL", expected, "", ""};
    try {
        r.actual = test();
        r.status = "PASS";
    } catch (const std::exception& e) {
        r.error = e.what();
    } catch (...) {
        r.error = "unknown exception";
    }
    results.push_back(r);
    std::cout << r.id << " " << r.status << " - " << r.name;
    if (!r.error.empty()) std::cout << " (" << r.error << ")";
    std::cout << '\n';
}

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

static void near(double actual, double expected, const std::string& message) {
    if (std::abs(actual - expected) > 1e-9) throw std::runtime_error(message);
}

static void same(const Tensor& a, const Tensor& b) {
    require(a.shape().dimensions() == b.shape().dimensions(), "shape mismatch");
    require(a.size() == b.size(), "size mismatch");
    for (std::size_t i = 0; i < a.size(); ++i)
        near(a.data()[i], b.data()[i], "element mismatch at " + std::to_string(i));
}

template<class F>
static void throws_as(F&& f, const std::string& message) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    require(caught, message);
}

static Tensor make(const std::vector<std::size_t>& shape,
                   const std::vector<double>& values) {
    Tensor t(shape);
    require(t.size() == values.size(), "test fixture size mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) t.data()[i] = values[i];
    return t;
}

static std::string write_csv(const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open CSV output");
    auto q = [](const std::string& s) {
        std::string v = "\"";
        for (char c : s) { if (c == '"') v += '"'; v += c; }
        return v + "\"";
    };
    out << "test_id,test_name,status,expected,actual,error\n";
    for (const auto& r : results)
        out << q(r.id) << ',' << q(r.name) << ',' << q(r.status) << ','
            << q(r.expected) << ',' << q(r.actual) << ',' << q(r.error) << '\n';
    return path;
}

int main(int argc, char** argv) {
    const std::string csv = argc > 1 ? argv[1] : "tc17_tc50_results.csv";

    const Tensor a = make({2,3}, {1,2,3,4,5,6});
    const Tensor b = make({2,3}, {6,5,4,3,2,1});

    // TC17-TC24: addition and subtraction
    run("TC17", "Addition element values", "[7,7,7,7,7,7]", [&] {
        Tensor r = add(a,b); for (std::size_t i=0;i<6;++i) near(r.data()[i],7,"addition mismatch");
        return "all six elements equal 7";
    });
    run("TC18", "Addition operator", "operator+ matches add()", [&] {
        same(a+b, add(a,b)); return "operator+ matches named addition";
    });
    run("TC19", "Addition shape preservation", "result shape={2,3}", [&] {
        Tensor r=add(a,b); require(r.shape().dimensions()==std::vector<std::size_t>{2,3},"shape mismatch");
        return "shape={2,3}";
    });
    run("TC20", "Addition rejects mismatched shapes", "std::invalid_argument", [&] {
        Tensor x({3,2}); throws_as([&]{ (void)add(a,x); },"addition accepted mismatched shapes");
        return "std::invalid_argument caught";
    });
    run("TC21", "Subtraction element values", "[-5,-3,-1,1,3,5]", [&] {
        Tensor r=subtract(a,b); const double e[]={-5,-3,-1,1,3,5};
        for(std::size_t i=0;i<6;++i) near(r.data()[i],e[i],"subtraction mismatch");
        return "all six elements match expected values";
    });
    run("TC22", "Subtraction operator", "operator- matches subtract()", [&] {
        same(a-b,subtract(a,b)); return "operator- matches named subtraction";
    });
    run("TC23", "Subtraction input immutability", "inputs unchanged", [&] {
        require(a.data()[0]==1 && a.data()[5]==6 && b.data()[0]==6 && b.data()[5]==1,"input mutated");
        return "both inputs unchanged";
    });
    run("TC24", "Subtraction rejects mismatched shapes", "std::invalid_argument", [&] {
        Tensor x({6}); throws_as([&]{ (void)subtract(a,x); },"subtraction accepted mismatched shapes");
        return "std::invalid_argument caught";
    });

    // TC25-TC30: element-wise multiplication
    run("TC25", "Element-wise multiplication values", "[6,10,12,12,10,6]", [&] {
        Tensor r=multiply(a,b); const double e[]={6,10,12,12,10,6};
        for(std::size_t i=0;i<6;++i) near(r.data()[i],e[i],"multiply mismatch");
        return "all six elements match expected values";
    });
    run("TC26", "Multiplication operator", "operator* matches multiply()", [&] {
        same(a*b,multiply(a,b)); return "operator* matches named multiplication";
    });
    run("TC27", "Multiplication shape preservation", "result shape={2,3}", [&] {
        Tensor r=multiply(a,b); require(r.shape().dimensions()==std::vector<std::size_t>{2,3},"shape mismatch");
        return "shape={2,3}";
    });
    run("TC28", "Multiplication rejects mismatched shapes", "std::invalid_argument", [&] {
        Tensor x({3,2}); throws_as([&]{ (void)multiply(a,x); },"multiplication accepted mismatched shapes");
        return "std::invalid_argument caught";
    });
    run("TC29", "Multiplication with negative and zero values", "[-6,0,-8,0,-10,0]", [&] {
        Tensor x=make({2,3},{-2,0,-4,0,-5,0});
        Tensor y=make({2,3},{3,7,2,9,2,11});
        Tensor r=multiply(x,y); const double e[]={-6,0,-8,0,-10,0};
        for(std::size_t i=0;i<6;++i) near(r.data()[i],e[i],"negative/zero multiplication mismatch");
        return "negative and zero products correct";
    });
    run("TC30", "Multiplication input immutability", "inputs unchanged", [&] {
        require(a.data()[0]==1 && a.data()[5]==6 && b.data()[0]==6 && b.data()[5]==1,"input mutated");
        return "both inputs unchanged";
    });

    // TC31-TC38: matrix multiplication
    const Tensor ma=make({2,3},{1,2,3,4,5,6});
    const Tensor mb=make({3,2},{7,8,9,10,11,12});
    const Tensor expected=make({2,2},{58,64,139,154});
    run("TC31", "2x3 times 3x2 matrix multiplication", "2x2 [58,64,139,154]", [&] {
        Tensor r=matmul(ma,mb); same(r,expected); return "shape={2,2}; values=[58,64,139,154]";
    });
    run("TC32", "Matrix multiplication output shape", "shape={2,2}", [&] {
        Tensor r=matmul(ma,mb); require(r.shape().dimensions()==std::vector<std::size_t>{2,2},"wrong output shape");
        return "shape={2,2}";
    });
    run("TC33", "Matrix multiplication 1x1", "6*7=42", [&] {
        Tensor x=make({1,1},{6}), y=make({1,1},{7}); Tensor r=matmul(x,y);
        near(r.data()[0],42,"1x1 product mismatch"); return "result=42";
    });
    run("TC34", "Matrix multiplication identity", "A*I=A", [&] {
        Tensor x=make({2,2},{2,3,4,5}), i=make({2,2},{1,0,0,1});
        same(matmul(x,i),x); return "identity multiplication preserved matrix";
    });
    run("TC35", "Parallel matrix multiplication", "matches sequential", [&] {
        same(matmul_parallel(ma,mb),expected); return "parallel result matches reference";
    });
    run("TC36", "Blocked matrix multiplication", "matches sequential", [&] {
        for(std::size_t n : {1U,2U,4U,8U}) same(matmul_blocked(ma,mb,n),expected);
        return "block sizes 1,2,4,8 match reference";
    });
    run("TC37", "Parallel blocked matrix multiplication", "matches sequential", [&] {
        for(std::size_t n : {1U,2U,4U,8U}) same(matmul_parallel_blocked(ma,mb,n),expected);
        return "parallel block sizes 1,2,4,8 match reference";
    });
    run("TC38", "Matrix multiplication rejects invalid dimensions", "std::invalid_argument for incompatible/rank mismatch", [&] {
        Tensor wrong({4,2}); throws_as([&]{ (void)matmul(ma,wrong); },"incompatible dimensions accepted");
        Tensor vector({3}); throws_as([&]{ (void)matmul(vector,vector); },"non-2D operands accepted");
        return "incompatible dimensions and non-2D operands rejected";
    });

    // TC39-TC45: reshape
    run("TC39", "Reshape 2x3 to 3x2", "shape={3,2}", [&] {
        Tensor x=a; x.reshape({3,2}); require(x.shape().dimensions()==std::vector<std::size_t>{3,2},"reshape shape mismatch");
        return "shape={3,2}";
    });
    run("TC40", "Reshape preserves flattened data", "[1,2,3,4,5,6]", [&] {
        Tensor x=a; x.reshape({3,2}); for(std::size_t i=0;i<6;++i) near(x.flat_at(i),double(i+1),"reshape changed data");
        return "flattened values preserved";
    });
    run("TC41", "Reshape preserves element count", "size remains 6", [&] {
        Tensor x=a; x.reshape({1,2,3}); require(x.size()==6,"element count changed");
        return "size=6";
    });
    run("TC42", "Reshape rejects changed element count", "std::invalid_argument and original shape retained", [&] {
        Tensor x=a; throws_as([&]{ x.reshape({4,2}); },"invalid reshape accepted");
        require(x.shape().dimensions()==std::vector<std::size_t>{2,3},"shape changed after failed reshape");
        return "invalid reshape rejected; original shape retained";
    });
    run("TC43", "Reshape initializer-list overload", "shape={6}", [&] {
        Tensor x=a; x.reshape({6}); require(x.rank()==1 && x.size()==6,"initializer-list reshape failed");
        return "shape={6}";
    });
    run("TC44", "Reshape zero-sized tensor", "zero elements retained", [&] {
        Tensor x({2,0,3}); x.reshape({0,6}); require(x.size()==0 && x.shape().dimensions()==std::vector<std::size_t>{0,6},"empty reshape mismatch");
        return "shape={0,6}; size=0";
    });
    run("TC45", "Reshape scalar tensor", "rank=0; value preserved", [&] {
        Tensor x({},9.0); x.reshape({}); require(x.rank()==0 && x.size()==1,"scalar reshape metadata mismatch");
        near(x.flat_at(0),9,"scalar value changed"); return "rank=0; size=1; value=9";
    });

    // TC46-TC50: ownership and memory robustness
    run("TC46", "Copy construction is deep", "copy data independent", [&] {
        Tensor original=a; Tensor copy(original); copy.data()[0]=999;
        require(original.data()[0]==1 && copy.data()[0]==999,"copy shares storage");
        return "copy mutation does not affect source";
    });
    run("TC47", "Copy assignment is deep", "assigned tensor independent", [&] {
        Tensor source=a, destination({1},-1); destination=source; destination.data()[1]=777;
        require(source.data()[1]==2 && destination.data()[1]==777,"copy assignment shares storage");
        return "copy assignment independent";
    });
    run("TC48", "Move construction preserves tensor contents", "moved-to tensor retains values", [&] {
        Tensor source=a; Tensor destination(std::move(source)); same(destination,a);
        return "moved-to tensor preserves shape and values";
    });
    run("TC49", "Move assignment preserves tensor contents", "destination receives shape and values", [&] {
        Tensor source=a, destination({1},-1); destination=std::move(source); same(destination,a);
        return "move-assigned tensor preserves shape and values";
    });
    run("TC50", "Aligned allocation and overflow guard", "64-byte alignment; dimension overflow rejected", [&] {
        Tensor x({1024},1.0);
        require(reinterpret_cast<std::uintptr_t>(x.data())%64==0,"data pointer is not 64-byte aligned");
        bool overflow=false;
        try { Tensor too_large({static_cast<std::size_t>(-1),2}); }
        catch(const std::overflow_error&) { overflow=true; }
        require(overflow,"dimension overflow was not rejected");
        return "64-byte alignment verified; overflow rejected";
    });

    write_csv(csv);
    std::size_t passed=0;
    for(const auto& r:results) if(r.status=="PASS") ++passed;
    std::cout << "\nSummary: " << passed << "/" << results.size()
              << " PASS; " << (results.size()-passed) << " FAIL\nCSV: " << csv << '\n';
    return passed==results.size()?0:1;
}
