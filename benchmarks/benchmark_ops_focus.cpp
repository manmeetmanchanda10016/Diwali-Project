#include "minitensor/Ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minitensor;
using Clock=std::chrono::steady_clock;
volatile double ops_sink=0;

template<class Function>
void measure(const std::string& op,const std::string& shape,int threads,
             std::size_t elements,Function function)
{
    // Single tiny operations fall below this platform's clock resolution.
    // Keep 3 warmup batches and 10 measured batches; report per-call times.
    const int batch=elements<=4096?256:1;
    for(int i=0;i<3;++i)
    {
        double value=0;
        for(int call=0;call<batch;++call) value+=function();
        ops_sink+=value;
    }
    std::vector<double> samples;
    double total=0;
    for(int i=0;i<10;++i)
    {
        const auto start=Clock::now();
        double value=0;
        for(int call=0;call<batch;++call) value+=function();
        const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count()/batch;
        ops_sink+=value; samples.push_back(ms); total+=ms;
    }
    auto sorted=samples; std::sort(sorted.begin(),sorted.end());
    std::cout<<op<<','<<shape<<','<<threads<<','<<elements<<','<<batch<<",3,10,"
             <<total/10<<','<<(sorted[4]+sorted[5])/2<<",PASS";
    for(double ms:samples) std::cout<<','<<ms;
    std::cout<<std::endl;
}

template<class Function>
void tensor_case(const std::string& op,const std::string& label,int threads,
                 double expected,Function function)
{
    const Tensor check=function();
    for(std::size_t i=0;i<check.size();++i)
        if(!std::isfinite(check.data()[i]) || check.data()[i]!=expected)
            throw std::runtime_error("Tensor benchmark correctness failure: "+op);
    measure(op,label,threads,check.size(),[&] {
        const Tensor out=function();
        #if defined(__GNUC__) || defined(__clang__)
        // Escape the whole buffer so unused output stores cannot be removed.
        asm volatile("" : : "g"(out.data()) : "memory");
        #endif
        return out.size()?out.data()[out.size()/2]:0.0;
    });
}

int main()
{
    try
    {
        omp_set_dynamic(0); omp_set_num_threads(1);
        std::cout<<"Operation,Shape,Threads,Elements,BatchCalls,Warmups,Measurements,Average_ms,Median_ms,Correctness";
        for(int i=0;i<10;++i) std::cout<<",Sample"<<i<<"_ms";
        std::cout<<'\n'<<std::fixed<<std::setprecision(9);
        const std::vector<Tensor(*)(const Tensor&,const Tensor&)> sequential={add,subtract,multiply};
        const std::vector<Tensor(*)(const Tensor&,const Tensor&)> parallel={add_parallel,subtract_parallel,multiply_parallel};
        const std::vector<Tensor(*)(const Tensor&,const Tensor&)> broadcast={add_broadcast,subtract_broadcast,multiply_broadcast};
        const std::vector<std::string> names={"add","subtract","multiply"};
        const double expected[]={3.5,-0.5,3.0};
        for(std::size_t count:{31U,4096U,1000000U,4000000U})
        {
            Tensor a({count},1.5),b({count},2.0);
            for(int op=0;op<3;++op)
            {
                omp_set_num_threads(1);
                tensor_case(names[op],std::to_string(count),1,expected[op],[&]{return sequential[op](a,b);});
                for(int threads:{1,2,4,8})
                {
                    omp_set_num_threads(threads);
                    tensor_case(names[op]+"_parallel",std::to_string(count),threads,expected[op],[&]{return parallel[op](a,b);});
                }
            }
        }
        struct BroadcastCase { const char* name; std::vector<std::size_t> a,b; };
        for(const auto& item:std::vector<BroadcastCase>{
                {"same",{1024,1024},{1024,1024}},
                {"row",{4096,257},{1,257}}, {"column",{4096,257},{4096,1}},
                {"scalar",{1000000},{}}, {"row3d",{16,257,129},{129}},
                {"general4d",{8,1,257,1},{1,16,1,33}},
                {"leading-singletons",{1,1,1024,1024},{1024,1024}}})
        {
            Tensor a(item.a,1.5),b(item.b,2);
            omp_set_num_threads(1);
            for(int op=0;op<3;++op)
                tensor_case(names[op]+"_broadcast",item.name,1,expected[op],[&]{return broadcast[op](a,b);});
        }
        for(const auto& dims:std::vector<std::vector<std::size_t>>{{64,64},{512,512},{2048,2048},{513,517}})
        {
            omp_set_num_threads(1);
            Tensor a(dims,1.5);
            tensor_case("transpose",std::to_string(dims[0])+"x"+std::to_string(dims[1]),1,1.5,[&]{return transpose(a);});
        }
        const std::vector<double(*)(const Tensor&)> reductions={sum,mean,min,max};
        const std::vector<std::string> reduction_names={"sum","mean","min","max"};
        for(std::size_t count:{4096U,1000000U,4000000U})
        {
            Tensor a({count});
            long double total=0;
            for(std::size_t i=0;i<count;++i) {a.data()[i]=(static_cast<int>(i%31)-15)/8.0; total+=a.data()[i];}
            const double values[]={static_cast<double>(total),static_cast<double>(total/count),-15/8.0,15/8.0};
            for(int threads:{1,2,4,8})
            {
                omp_set_num_threads(threads);
                for(int op=0;op<4;++op)
                {
                    const double actual=reductions[op](a);
                    if(!std::isfinite(actual)||std::abs(actual-values[op])>1e-10)
                        throw std::runtime_error("Reduction benchmark correctness failure");
                    measure(reduction_names[op],std::to_string(count),threads,count,[&]{return reductions[op](a);});
                }
            }
        }
        omp_set_num_threads(1);
        {
            Tensor a({512,512});
            double expected_total=0;
            for(std::size_t i=0;i<a.size();++i) {a.data()[i]=(i%31)/8.0;expected_total+=a.data()[i];}
            for(int kind=0;kind<3;++kind)
            {
                const auto function=[&] {
                    double total=0;
                    std::vector<std::size_t> index(2);
                    for(std::size_t i=0;i<512;++i) for(std::size_t j=0;j<512;++j)
                    {
                        if(kind==0) total+=a.flat_at(i*512+j);
                        else if(kind==1) total+=a.at({i,j});
                        else {index[0]=i;index[1]=j;total+=a.at(index);}
                    }
                    return total;
                };
                if(function()!=expected_total) throw std::runtime_error("Index correctness failure");
                measure(std::vector<std::string>{"index_flat","index_list","index_vector"}[kind],"512x512",1,a.size(),function);
            }
        }
        for(std::size_t count:{4096U,1000000U,4000000U})
        {
            tensor_case("construct",std::to_string(count),1,1.5,[&]{return Tensor({count},1.5);});
            const Tensor a({count},1.5);
            tensor_case("copy",std::to_string(count),1,1.5,[&]{return Tensor(a);});
        }
        std::cerr<<"Checksum: "<<ops_sink<<'\n';
    }
    catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
