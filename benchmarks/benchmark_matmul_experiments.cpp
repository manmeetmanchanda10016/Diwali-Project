// Isolated experiments: production dispatch is unchanged by this program.
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
volatile double experiment_sink=0;
// Copy of the production row-group kernel, changing only schedule(static) to
// schedule(runtime), so scheduling policies can be compared without confounds.
Tensor scheduled_matmul(const Tensor& a, const Tensor& b)
{
    if (a.rank() != 2 || b.rank() != 2)
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors");

    const auto& as = a.shape().dimensions();
    const auto& bs = b.shape().dimensions();

    const std::size_t rows_a = as[0];
    const std::size_t cols_a = as[1];
    const std::size_t rows_b = bs[0];
    const std::size_t cols_b = bs[1];

    if (cols_a != rows_b)
        throw std::invalid_argument("Invalid matrix dimensions");

    Tensor result({rows_a, cols_b}, 0.0);

    const double* A = a.data();
    const double* B = b.data();
    double* C = result.data();

    // Empty products need no pointer arithmetic or parallel region.
    if (rows_a == 0 || cols_a == 0 || cols_b == 0)
        return result;

    // Reuse each B value across four rows. Each worker owns complete output
    // rows, so no synchronization is needed while accumulating over k.
    // Keep one-row tasks when grouping would leave requested workers idle.
    const std::size_t row_group =
        rows_a / 4 >= static_cast<std::size_t>(omp_get_max_threads()) ? 4 : 1;
    const std::size_t groups = rows_a / row_group + (rows_a % row_group != 0);

    #pragma omp parallel for schedule(runtime)
    for (long long group = 0; group < static_cast<long long>(groups); ++group)
    {
        const std::size_t i = static_cast<std::size_t>(group) * row_group;
        if (row_group == 4 && rows_a - i >= row_group)
        {
            double* c0 = C + i * cols_b;
            double* c1 = c0 + cols_b;
            double* c2 = c1 + cols_b;
            double* c3 = c2 + cols_b;
            for (std::size_t k = 0; k < cols_a; ++k)
            {
                const double a0 = A[i * cols_a + k];
                const double a1 = A[(i + 1) * cols_a + k];
                const double a2 = A[(i + 2) * cols_a + k];
                const double a3 = A[(i + 3) * cols_a + k];
                const double* b_row = B + k * cols_b;
                #pragma omp simd
                for (std::size_t j = 0; j < cols_b; ++j)
                {
                    const double value = b_row[j];
                    c0[j] += a0 * value;
                    c1[j] += a1 * value;
                    c2[j] += a2 * value;
                    c3[j] += a3 * value;
                }
            }
        }
        else
        {
            // Single-row tasks or the final one to three rows.
            const std::size_t end = i + std::min(row_group, rows_a - i);
            for (std::size_t row = i; row < end; ++row)
                for (std::size_t k = 0; k < cols_a; ++k)
                {
                    const double value = A[row * cols_a + k];
                    #pragma omp simd
                    for (std::size_t j = 0; j < cols_b; ++j)
                        C[row * cols_b + j] += value * B[k * cols_b + j];
                }
        }
    }

    return result;
}


Tensor transposed_b_matmul(const Tensor& a,const Tensor& b)
{
    const auto m=a.shape().dimensions()[0],k=a.shape().dimensions()[1],n=b.shape().dimensions()[1];
    // Packing/transposition, temporary allocation and destruction are timed.
    const Tensor bt=transpose(b);
    Tensor out({m,n});
    const double* A=a.data(); const double* BT=bt.data(); double* C=out.data();
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<static_cast<long long>(m);++i)
        for(std::size_t j=0;j<n;++j)
        {
            double value=0;
            #pragma omp simd reduction(+:value)
            for(std::size_t p=0;p<k;++p) value+=A[i*k+p]*BT[j*k+p];
            C[i*n+j]=value;
        }
    return out;
}

void run(const std::string& name,std::size_t n,int threads,std::size_t tile,
         const Tensor& expected,const std::function<Tensor()>& function)
{
    const Tensor actual=function();
    if(actual.shape().dimensions()!=expected.shape().dimensions()) throw std::runtime_error("Shape failure");
    for(std::size_t i=0;i<actual.size();++i)
        if(!std::isfinite(actual.data()[i]) ||
           std::abs(actual.data()[i]-expected.data()[i])>1e-10+1e-10*std::abs(expected.data()[i]))
            throw std::runtime_error("Experiment correctness failure: "+name);
    for(int i=0;i<3;++i) {const Tensor out=function();experiment_sink+=out.data()[out.size()/2];}
    std::vector<double> samples; double total=0;
    for(int i=0;i<10;++i)
    {
        double value=0; const auto start=Clock::now();
        { const Tensor out=function();value=out.data()[out.size()/2]; }
        const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        samples.push_back(ms);total+=ms;experiment_sink+=value;
    }
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    const double median=(sorted[4]+sorted[5])/2;
    std::cout<<n<<','<<name<<','<<threads<<','<<tile<<",3,10,"<<total/10<<','<<median
             <<','<<2.0*n*n*n/(median*1e6)<<",PASS";
    for(double ms:samples)std::cout<<','<<ms;
    std::cout<<std::endl;
}

int main()
{
    try
    {
        omp_set_dynamic(0);
        std::cout<<"MatrixSize,Algorithm,Threads,BlockSize,Warmups,Measurements,Average_ms,Median_ms,GFLOPS,Correctness";
        for(int i=0;i<10;++i)std::cout<<",Sample"<<i<<"_ms";
        std::cout<<'\n'<<std::fixed<<std::setprecision(9);
        omp_sched_t saved;int saved_chunk;omp_get_schedule(&saved,&saved_chunk);
        for(std::size_t n:{257U,512U,1024U})
        {
            Tensor a({n,n}),b({n,n});
            for(std::size_t i=0;i<a.size();++i)
            { a.data()[i]=(static_cast<int>(i%17)-8)/9.0;b.data()[i]=(static_cast<int>(i%23)-11)/12.0; }
            const Tensor expected=matmul_blocked(a,b,128);
            omp_set_num_threads(1);
            for(std::size_t tile:{32U,64U,96U,128U,192U,256U})
                run("sequential-blocked",n,1,tile,expected,[&]{return matmul_blocked(a,b,tile);});
            for(int threads:{1,2})
            {
                omp_set_num_threads(threads);
                run("parallel-current",n,threads,0,expected,[&]{return matmul_parallel(a,b);});
                run("transposed-B-end-to-end",n,threads,0,expected,[&]{return transposed_b_matmul(a,b);});
                for(const auto& policy:std::vector<std::pair<omp_sched_t,std::string>>{
                        {omp_sched_static,"static-runtime"},{omp_sched_dynamic,"dynamic1-runtime"},
                        {omp_sched_guided,"guided1-runtime"}})
                {
                    omp_set_schedule(policy.first,1);
                    run(policy.second,n,threads,0,expected,[&]{return scheduled_matmul(a,b);});
                }
                for(std::size_t tile:{32U,64U,96U,128U,192U,256U})
                    run("parallel-blocked",n,threads,tile,expected,[&]{return matmul_parallel_blocked(a,b,tile);});
            }
        }
        omp_set_schedule(saved,saved_chunk);
        std::cerr<<"Checksum: "<<experiment_sink<<'\n';
    }
    catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
