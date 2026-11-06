#include "minitensor/Ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <vector>

using namespace minitensor;
using Dims = std::vector<std::size_t>;

namespace {
void require(bool condition) { if (!condition) throw std::runtime_error("Operation reference check failed"); }
void equal(double actual, double expected)
{
    if (std::isnan(expected)) { require(std::isnan(actual)); return; }
    if (std::isinf(expected)) { require(actual == expected); return; }
    require(std::isfinite(actual));
    require(std::abs(actual - expected) <= 1e-10 + std::abs(expected) * 1e-10);
    if (actual == 0 && expected == 0) require(std::signbit(actual) == std::signbit(expected));
}
template<class Function> void invalid(Function function)
{
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("Expected invalid_argument");
}
Dims broadcast_dims(const Dims& a, const Dims& b)
{
    Dims out(std::max(a.size(), b.size()), 1);
    for (std::size_t i = 0; i < out.size(); ++i)
    {
        const auto x = i < out.size()-a.size() ? 1 : a[i-(out.size()-a.size())];
        const auto y = i < out.size()-b.size() ? 1 : b[i-(out.size()-b.size())];
        if (x != y && x != 1 && y != 1) throw std::invalid_argument("Incompatible reference");
        out[i] = x == 1 ? y : x;
    }
    return out;
}
// Convert explicit coordinates to each input's own row-major offset.
double broadcast_value(const Tensor& input, const Dims& coords)
{
    Dims indices(input.rank());
    for (std::size_t i = 0; i < input.rank(); ++i)
        indices[i] = input.shape().dimensions()[i] == 1 ? 0 : coords[coords.size()-input.rank()+i];
    return input.at(indices);
}
void check_broadcast(const Dims& ad, const Dims& bd, std::mt19937& rng)
{
    Tensor a(ad), b(bd);
    std::uniform_int_distribution<int> random(-16, 16);
    for (std::size_t i=0; i<a.size(); ++i) a.data()[i] = random(rng)/8.0;
    for (std::size_t i=0; i<b.size(); ++i) b.data()[i] = random(rng)/8.0;
    Dims dims;
    try { dims = broadcast_dims(ad,bd); }
    catch (const std::invalid_argument&)
    {
        invalid([&] { add_broadcast(a,b); });
        invalid([&] { subtract_broadcast(a,b); });
        invalid([&] { multiply_broadcast(a,b); });
        return;
    }
    const Tensor outputs[] = {add_broadcast(a,b),subtract_broadcast(a,b),multiply_broadcast(a,b)};
    for (const auto& out : outputs)
    {
        require(out.shape().dimensions() == dims);
        if (out.size()) require(reinterpret_cast<std::uintptr_t>(out.data()) % 64 == 0);
    }
    for (std::size_t flat=0; flat<outputs[0].size(); ++flat)
    {
        Dims coords(dims.size());
        auto remainder=flat;
        for (std::size_t axis=dims.size(); axis>0; --axis)
        { coords[axis-1]=remainder%dims[axis-1]; remainder/=dims[axis-1]; }
        const double x=broadcast_value(a,coords), y=broadcast_value(b,coords);
        equal(outputs[0].data()[flat],x+y);
        equal(outputs[1].data()[flat],x-y);
        equal(outputs[2].data()[flat],x*y);
    }
    if (ad==bd)
    {
        for (int threads : {1,2,4})
        {
            omp_set_num_threads(threads);
            const Tensor regular[] = {add(a,b),subtract(a,b),multiply(a,b),
                                     add_parallel(a,b),subtract_parallel(a,b),multiply_parallel(a,b)};
            for (int op=0; op<6; ++op)
                for (std::size_t i=0; i<regular[op].size(); ++i)
                    equal(regular[op].data()[i],outputs[op%3].data()[i]);
        }
    }
}
void check_reductions(const Tensor& a)
{
    long double total=0, magnitude=0;
    double lo=a.size()?a.data()[0]:0, hi=lo;
    for (std::size_t i=0; i<a.size(); ++i)
    {
        total+=a.data()[i]; magnitude+=std::abs(a.data()[i]);
        if (a.data()[i]<lo) lo=a.data()[i];
        if (a.data()[i]>hi) hi=a.data()[i];
    }
    for (int threads : {1,2,4,8})
    {
        omp_set_num_threads(threads);
        const double actual=sum(a), expected=static_cast<double>(total);
        if (!std::isfinite(expected)) equal(actual,expected);
        else require(std::isfinite(actual) && std::abs(actual-expected)<=1e-10+static_cast<double>(magnitude)*1e-12);
        if (!a.size())
        {
            invalid([&] { mean(a); }); invalid([&] { min(a); }); invalid([&] { max(a); });
        }
        else
        {
            equal(min(a),lo); equal(max(a),hi);
            const double expected_mean=static_cast<double>(total/a.size());
            if (!std::isfinite(expected_mean)) equal(mean(a),expected_mean);
            else require(std::abs(mean(a)-expected_mean)<=1e-10+static_cast<double>(magnitude/a.size())*1e-12);
        }
    }
}
} // namespace

int main()
{
    try
    {
        omp_set_dynamic(0);
        std::mt19937 rng(20261106);
        // Empty-brace initial values remain valid public zero initialization.
        Storage storage(4, {});
        Tensor initialized(Dims{2,3}, {});
        for (std::size_t i=0;i<storage.size();++i) equal(storage.data()[i],0);
        for (std::size_t i=0;i<initialized.size();++i) equal(initialized.data()[i],0);
        Tensor shared({65537});
        for(std::size_t i=0;i<shared.size();++i) shared.data()[i]=(static_cast<int>(i%31)-15)/8.0;
        for(int threads:{1,2,4,8})
        {
            omp_set_num_threads(threads);
            const Tensor doubled=add_parallel(shared,shared);
            const Tensor squared=multiply_parallel(shared,shared);
            for(std::size_t i=0;i<shared.size();++i)
            {equal(doubled.data()[i],2*shared.data()[i]);equal(squared.data()[i],shared.data()[i]*shared.data()[i]);}
        }
        const std::vector<Dims> shapes = {{},{0},{1},{3},{2,3},{1,3},{2,1},{2,1,5},
            {1,3,1},{3,5},{5},{1,1,1,1},{2,0,3},{1,0,1},{2,3,5,7}};
        for (const auto& a: shapes) for (const auto& b: shapes) check_broadcast(a,b,rng);
        for (int sample=0; sample<200; ++sample)
        {
            Dims a(1+rng()%5), b(a.size());
            for (std::size_t i=0; i<a.size(); ++i)
            {
                const auto d=1+rng()%5;
                a[i]=rng()%2?1:d; b[i]=rng()%2?d:1;
            }
            check_broadcast(a,b,rng);
        }
        check_broadcast({7,1,19,1},{1,5,1,11},rng);
        check_broadcast({5,17,33},{33},rng);
        for (std::size_t n : {65535U,65536U,65537U}) check_broadcast({n},{n},rng);
        for (const auto& dims : std::vector<Dims>{{0,3},{3,0},{1,513},{513,1},{17,19},{513,517}})
        {
            Tensor a(dims);
            for(std::size_t i=0;i<a.size();++i) a.data()[i]=static_cast<double>(i)-17.5;
            const Tensor out=transpose(a);
            require(out.shape().dimensions()==Dims({dims[1],dims[0]}));
            for(std::size_t i=0;i<dims[0];++i) for(std::size_t j=0;j<dims[1];++j)
                equal(out.at({j,i}),a.at({i,j}));
        }
        for (std::size_t count : {0U,1U,31U,4096U,262143U,262144U,262145U,1000003U})
        {
            Tensor a({count});
            for(std::size_t i=0;i<count;++i) { equal(a.data()[i],0); a.data()[i]=(static_cast<int>(i%31)-15)/8.0; }
            check_reductions(a);
        }
        const double nan=std::numeric_limits<double>::quiet_NaN();
        const double inf=std::numeric_limits<double>::infinity();
        for (const auto& values: std::vector<std::vector<double>>{
                {nan,1,-1},{1,nan,-2,3},{inf,1,-inf},{-0.0,+0.0},{+0.0,-0.0},
                {1,+0.0,-0.0},{-1,-0.0,+0.0},{1e16,1,-1e16,3}})
        {
            Tensor a({values.size()});
            std::copy(values.begin(),values.end(),a.data()); check_reductions(a);
        }
        for (double zero : {+0.0,-0.0})
        {
            Tensor a({1000003},1.0);
            a.data()[123]=zero; a.data()[700001]=-zero; a.data()[500002]=nan;
            // Skip sum/mean's NaN check only through the same reference logic.
            check_reductions(a);
            a.data()[0]=nan; check_reductions(a);
        }
        std::cout << "Broadcast, elementwise, transpose, reduction and initialization reference checks passed\n";
    }
    catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
