#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <random>
#include <numbers>
#include <chrono>

#include "lbfgsb.h"
#include "gradient_strategy.h"

#if defined(USING_EIGEN)
#define EIGEN_NO_DEBUG  // Disable debug mode for Eigen (faster)
#define EIGEN_DONT_PARALLELIZE  // Disable multi-threading if the overhead is too large for small matrices
#define EIGEN_VECTORIZE  // Enable vectorization
#endif

// Example usage

using namespace std;
using namespace gradients;
using namespace LBFGSB;

static int bdexp_test(double x0, int n, double l, double u, bool debug = false)
{
    cout << "in bdexp \n";
    VectorXd g(n);
    auto f = [&n](const VectorXd& x_) -> double
        {
            double sum = 0.0;
            for (int i = 0; i < n - 2; ++i)
            {
                sum += (x_[i] + x_[i + 1]) * exp((x_[i] + x_[i + 1]) * -x_[i + 2]);
            }
            return sum;
        };

    auto grad = [&g, n](const VectorXd& x_) -> VectorXd
        {
            auto g1 = [&x_](int i) -> double
                {
                    // Derivative of the generalized function with respect to x_i 
                    //Term involving x_i, x_{ i + 1 }, and x_{ i + 2 }
                    double term = (x_[i] + x_[i + 1]);
                    double exponent = -(x_[i] + x_[i + 1]) * x_[i + 2];
                    double derivative = (1 - x_[i + 2] * term) * exp(exponent);
                    return derivative;
                };
            auto g2 = [&x_](int i) -> double
                {
                    double term = (x_[i] + x_[i + 1]);
                    double exponent = -(x_[i] + x_[i + 1]) * x_[i + 2];
                    double derivative = (1 - x_[i + 2] * term) * exp(exponent);
                    return derivative;
                };
            auto g3 = [&x_](int i) ->double
                {
                    double term = (x_[i] + x_[i + 1]) * (x_[i] + x_[i + 1]);
                    double exponent = -(x_[i] + x_[i + 1]) * x_[i + 2];
                    double derivative = -term * exp(exponent);
                    return derivative;
                };
            for (int i = 0; i < n; ++i)
            {
                g[i] = 0.0;
            }
            for (int i = 0; i < n - 2; ++i)
            {
                g[i] += g1(i);
                g[i + 1] += g2(i);
                g[i + 2] += g3(i);
            }
            return g;
        };
#if defined(USING_EIGEN)
    VectorXd x = VectorXd::Constant(n, x0);
    VectorXd lb = VectorXd::Constant(n, l);
    VectorXd ub = VectorXd::Constant(n, u);
#else
    VectorXd x(n, x0), lb(n, l), ub(n, u);
#endif
    if (debug)
    {
        cout << "debug on\n";
    }
    std::chrono::time_point<std::chrono::system_clock> st = std::chrono::system_clock::now();
    bool status = optimize(f, grad, x, lb, ub, 5, 100, 10, 1e-7, 1e-4, 0.9, 2.5, 1e12, debug);
    std::chrono::time_point<std::chrono::system_clock> et = std::chrono::system_clock::now();
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count(); 
    cout << "opmised finished in " << dt << " milliseconds\n";
    cout << "Optimized position: (";
    for (int i = 0; i < min(n, 10); ++i)
    {
        cout << x[i] << " ";
    }
    cout << "...), function = " << f(x) << "\n";
    return 1;
}
static int quadratic(double x0, double l, double u)
{
    auto f = [](const VectorXd& x_) -> double
        {
            double x = x_[0];
            double y = x_[1];
            double f = x * x + y * y;
            return f;
        };

    auto g = [](const VectorXd& x_) -> VectorXd
        {
            double x = x_[0];
            double y = x_[1];
            VectorXd result(2);  
            result[0] = 2 * x;
            result[1] = 2 * y;   
            return result;
        };
    cout << "quadratic test \n";
#if defined(USING_EIGEN)
    VectorXd x = VectorXd::Constant(2, x0);
    VectorXd lb = VectorXd::Constant(2, l);
    VectorXd ub = VectorXd::Constant(2, u);
#else
    VectorXd x(2, x0), lb(2, l), ub(2, u);
#endif
    bool status = optimize(f, g, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }

    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";

    NumericalGradient n_grad(f);
    std::cout << "quadratic test - numerical gradient\n";
#if defined(USING_EIGEN)
    x = VectorXd::Constant(2, x0);
#else
    x = VectorXd(2, x0);
#endif
    status = optimize(f, n_grad, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";

    return 1;
}

static int rosenbrock(double l, double u)
{
    std::cout << "rosenbrock test \n";
    auto f = [](const VectorXd&x_) -> double
        {
            double x = x_[0];
            double y = x_[1];
            double f = (1. - x) * (1. - x) + 100.0 * (y - x * x) * (y - x * x);
            return f;
        };
    auto g = [](const VectorXd& x_) -> VectorXd
        {
            double x = x_[0];
            double y = x_[1];
            VectorXd g(2);
            g[0] = -2 * (1 - x) - 400 * x * (y - x * x);
            g[1] = 200.0 * (y - x * x);
            return g;
        };
#if defined(USING_EIGEN)
    VectorXd x(2);
    x << -1.0, 1.0;
    VectorXd lb = VectorXd::Constant(2, l);
    VectorXd ub = VectorXd::Constant(2, u);
#else
    VectorXd x = { -1.2, 1.0 };
    VectorXd lb = { l, l };
    VectorXd ub = { u, u };
#endif
    bool status = optimize(f, g, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";
    NumericalGradient n_grad(f, 1e-5);
    std::cout << "rosenbrock test - numerical gradient\n";
#if defined(USING_EIGEN)
    x << -1., 1.0;
#else
    x = { -1.2, 1.0 };
#endif
    status = optimize(f, n_grad, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";

    return 1;
}
static int rastrigin(double l, double u, size_t n = 2, double a = 10, double x0 = 1.)
{
    std::cout << "rastrigin test, dimension = " << n << "\n";
    auto f = [&a](const VectorXd& x_)
        {
            const double pi = 3.1415927;
            size_t n = x_.size();
            double f = a * n;
            for (int i = 0; i < n; ++i)
            {
                f += x_[i] * x_[i] - 10 * cos(2 * pi * x_[i]);
            }
            return f;
        };
    auto g = [&a](const VectorXd& x_) -> VectorXd
        {
            const double pi = 3.1415927;
            size_t n = x_.size();
            VectorXd g(n);
            for (size_t i = 0; i < n; ++i)
            {
                g[i] = 2 * x_[i] + 2 * pi * a * sin(2 * pi * x_[i]);
            }
            return g;
        };
#if defined(USING_EIGEN)
    VectorXd x = VectorXd::Constant(n, x0);
    VectorXd lb = VectorXd::Constant(n, l);
    VectorXd ub = VectorXd::Constant(n, u);
#else
    VectorXd x(n, x0);
    VectorXd lb(n, l);
    VectorXd ub(n, u);
#endif
    bool status = optimize(f, g, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 200 iterations \n";
    }
    cout << "Optimized position: (" << x[0];
    for (int i = 1; i < n; ++i)
    {
        cout << "," << x[i];
    }
    cout << " f = " << f(x) << "\n";
    NumericalGradient n_grad(f);
    std::cout << "rastrigin test - numerical gradient\n";
#if defined(USING_EIGEN)
    x = VectorXd::Constant(n, x0);
#else
    x = VectorXd (n, x0);
#endif
    status = optimize(f, n_grad, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0];
    for (int i = 1; i < n; ++i)
    {
        cout << "," << x[i];
    }
    cout << " f = " << f(x) << "\n";
    return 1;
}
static int himmelblau(double l, double u)
{
    std::cout << "himmelblau test \n";
    auto f = [](const VectorXd& x_) -> double
        {
            double x = x_[0];
            double y = x_[1];
            double f = (x * x + y - 11) * (x * x + y - 11) + (x + y * y - 7) * (x + y * y - 7);
            return f;
        };
    auto g = [](const VectorXd& x_) -> VectorXd
        {
            double x = x_[0];
            double y = x_[1];
            VectorXd g(2);
            g[0] = 2 * 2 * x * (x * x + y - 11) + 2 * (x + y * y - 7);
            g[1] = 2 * (x * x + y - 11) + 2 * y * (x + y * y - 7);
            return g;
        };
#if defined(USING_EIGEN)
    VectorXd x(2);
    x << 0, 1.0 ;
    VectorXd lb = VectorXd::Constant(2, l);
    VectorXd ub = VectorXd::Constant(2, u);
#else
    VectorXd x = { 0, 1.0 };
    VectorXd lb = { l, l };
    VectorXd ub = { u, u };
#endif
    bool status = optimize(f, g, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";

    std::cout << "himmelblau test - numerical gradient\n";
    NumericalGradient n_grad(f);
#if defined(USING_EIGEN)
    x << 0, 1.;
#else
    x = { 0, 1.0 };
#endif
    status = optimize(f, n_grad, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";

    return 1;
}
static int beale()
{
    std::cout << "beale test\n";
    auto  f = [](const VectorXd& x_)
        {
            double x = x_[0];
            double y = x_[1];
            double f = pow(1.5 - x + x * y, 2) + pow(2.25 - x + x * y * y, 2) + pow(2.625 - x + x * y * y * y, 2);
            return f;
        };
    auto g = [](const VectorXd& x_) -> VectorXd
        {
            double x = x_[0];
            double y = x_[1];
            VectorXd g(2);
            g[0] = 2 * (1.5 - x + x * y) * (-1. + y) + 2 * (2.25 - x + x * y * y) * (-1 + y * y) +
                2 * (2.625 - x + x * y * y * y) * (-1 + y * y * y);
            g[1] = 2 * (1.5 - x + x * y) * (x)+2. * (2.25 - x + x * y * y) * 2 * x * y +
                2 * (2.625 - x + x * y * y * y) * 3 * x * y * y;
            return g;
        };
#ifdef USING_EIGEN
    VectorXd x(2);
    x << 0, 0;
    VectorXd lb = VectorXd::Constant(2, -4.5);
    VectorXd ub = VectorXd::Constant(2, 4.5);
#else
    VectorXd x = { 0, 0 };
    VectorXd lb = { -4.5, -4.5 };
    VectorXd ub = { 4.5, 4.5 };
#endif

    bool status = optimize(f, g, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position using AnalyticalGradient: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";
    return 1;
}
static int threehumpcamel()
{
    std::cout << "three hump camel test\n";
    auto f = [](const VectorXd&x_) -> double
        {
            double x = x_[0];
            double y = x_[1];
            double t = x * x;
            double f = 2. * t - 1.05 * t * t + t * t * t / 6 + x * y + y * y;
            return f;
        };
   
    auto g = [](const VectorXd& x_) -> VectorXd
        {
            double x = x_[0];
            double y = x_[1];
            VectorXd g(2);
            g[0] = 4 * x - 4.2 * x * x * x + x * x * x * x * x + y;
            g[1] = x + 2 * y;
            return g;
        };
#ifdef USING_EIGEN
    VectorXd x(2);
    VectorXd lb = VectorXd::Constant(2, -5); 
    VectorXd ub = VectorXd::Constant(2, 5);
    x << -2, 2;
#else
    VectorXd x = { -2, 2 };
    VectorXd lb = { -5, -5 };
    VectorXd ub = { 5, 5 };
#endif
   
    bool status = optimize(f, g, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }

    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";
    return 1;
}
static int easom()
{
    std::cout << "easom function test\n";
    auto  f = [](const VectorXd& x_)
        {
            const double pi = 3.1415927;
            double x = x_[0];
            double y = x_[1];
            double h = exp(-((x - pi) * (x - pi) + (y - pi) * (y - pi)));
            double f = -cos(x) * cos(y) * h;
            return f;
        };
    auto g = [](const VectorXd& x_) -> VectorXd
        {
            const double pi = 3.1415927;
            double x = x_[0];
            double y = x_[1];
            double h = exp(-((x - pi) * (x - pi) + (y - pi) * (y - pi)));
            VectorXd g(2);
            g[0] = h * (sin(x) * cos(y) + cos(x) * cos(y) * 2 * (x-pi));
            g[1] = h * (cos(x) * sin(y) + cos(x) * cos(y) * 2 * (y-pi));
            return g;
        };
#ifdef USING_EIGEN
    VectorXd x(2);
    x << 4, 4;
    VectorXd lb = VectorXd::Constant(2, -100);
    VectorXd ub = VectorXd::Constant(2, 100);
#else
    VectorXd x = {4, 4};
    VectorXd lb = { -100, -100 };
    VectorXd ub = { 100, 100 };
#endif
    bool status = optimize(f, g, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";
    return 1;
}
static int booth()
{
    std::cout << "booth test\n";
    auto  f = [](const VectorXd& x_)
        {
            double x = x_[0];
            double y = x_[1];
            double f = pow(x + 2 *y - 7, 2) + pow(2*x + y -5, 2);
            return f;
        };
    auto g = [](const VectorXd& x_) -> VectorXd
        {
            double x = x_[0];
            double y = x_[1];
            VectorXd g(2);
            g[0] = 2 * (x + 2 * y - 7) + 2 * 2 * (2 * x + y - 5);
            g[1] = 2 * 2 * (x + 2 * y - 7) + 2 * (2 * x + y - 5);
            return g;
        };
#ifdef USING_EIGEN
    VectorXd x(2);
    x << 2, 0;
    VectorXd lb = VectorXd::Constant(2, -10);
    VectorXd ub = VectorXd::Constant(2, 10);
#else
    VectorXd x = { 0, 0 };
    VectorXd lb = { -10, -10 };
    VectorXd ub = { 10, 10 };
#endif
    bool status = optimize(f, g, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";
    return 1;
}
static int egg_holder()
{
    std::cout << "eggholder test\n";
    auto f = [](const VectorXd& p) -> double
        {
            double x1 = p[0];
            double x2 = p[1];
            double sum = -(x2 + 47) * sin(sqrt(fabs(x2 + x1 / 2 + 47))) - x1 * sin(sqrt(fabs(x1 - (x2 + 47))));
            return sum;
        };
#ifdef USING_EIGEN
    VectorXd x(2);
    x << 20, 0;
    VectorXd lb = VectorXd::Constant(2, -512);
    VectorXd ub = VectorXd::Constant(2, 512);
#else
    VectorXd x = { 10, 10 };
    VectorXd lb = { -512, -512 };
    VectorXd ub = { 512, 512 };
#endif
    NumericalGradient n_grad(f, 1e-4);
    bool status = optimize(f, n_grad, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";
    return 1;
}
static int mccormick()
{
    std::cout << "mccormick test\n";
    auto f = [](const VectorXd& p) -> double
        {
            double x = p[0];
            double y = p[1];
            return sin(x+y) + (x-y)*(x-y) - 1.5*x + 2.5*y + 1;
        };
    auto g = [](const VectorXd& p) -> VectorXd
        {
            double x = p[0];
            double y = p[1];
            VectorXd grad(2);
            grad[0] = cos(x + y) + 2 * (x - y) - 1.5;
            grad[1] = cos(x + y) - 2 * (x - y) + 2.5;
            return grad;
        };
#ifdef USING_EIGEN
    VectorXd x(2);
    x << 0., 0.;
    VectorXd lb(2), ub(2);
    lb << -1.5, -2;
    ub << 4, 4;
#else
    VectorXd x = { 0, 0 };
    VectorXd lb = { -1.5, -2 };
    VectorXd ub = { 4, 4 };
#endif
    bool status = optimize(f, g, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";
    return 1;
}
static int schwefel(int d)
{
    cout << "schwefel test, dimension = " << d << "\n";
    auto f = [&d](const VectorXd& x) -> double
        {
            double sum = 418.9829 * d;
            for (int i = 0; i < d; ++i)
            {
                sum -= x[i] * sin(sqrt(fabs(x[i])));
            }
            return sum;
        };
    auto g = [&d](const VectorXd& x) -> VectorXd
        {
            VectorXd grad(d);
            for (int i = 0; i < d; ++i)
            {
                double x_ = sqrt(fabs(x[i]));
                grad[i] = -sin(x_) - 0.5 * x_ * cos(x_);
            }
            return grad;
        };
#ifdef USING_EIGEN
    VectorXd x = VectorXd::Constant(d, 100);
    VectorXd lb = VectorXd::Constant(d, -500);
    VectorXd ub = VectorXd::Constant(d, 500);
#else
    VectorXd x(d, 100);
    VectorXd lb(d, -500);
    VectorXd ub(d, 500);
#endif
    NumericalGradient n_grad(f);
    bool status = optimize(f, g, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (";
    for (size_t i = 0; i < d; ++i)
    {
        std::cout << x[i] << ", ";
    }
    std::cout << "f = " << f(x) << "\n";
    return 1;
}
static int styblinski_tang(size_t n = 2, double l=-5, double u=5)
{
    std::cout << "styblinski-tang test\n";
    auto f = [](const VectorXd& x) -> double
        {
            double sum = 0.0;
            for (size_t i = 0; i < x.size(); ++i)
            {
                double xi = x[i];
                sum += xi * xi * xi * xi - 16 * xi * xi + 5 * xi;
            }
            return 0.5 * sum;
        };
    auto g = [](const VectorXd& x) -> VectorXd
        {
            size_t n = x.size();
            VectorXd grad(n);
            for (size_t i = 0; i < n; ++i)
            {
                double xi = x[i];
                grad[i] = 0.5 * (4 * xi * xi * xi - 32 * xi + 5);
            }
            return grad;
        };
#ifdef USING_EIGEN
    VectorXd x = VectorXd::Constant(n, 0);
    VectorXd lb  = VectorXd::Constant(n, l);
    VectorXd ub = VectorXd::Constant(n, u);
#else
    VectorXd x(n, 0);
    VectorXd lb(n, l);
    VectorXd ub(n, u);
#endif
    bool status = optimize(f, g, x, lb, ub);
    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (";
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << x[i] << ", ";
    }
    std::cout << "f = " << f(x) << "\n";

    return 1;
}
static void linear_regession(double a, double b, double x_low, double x_high, size_t n, double error=0.1)
{
    std::cout << "linear regression tests\n";
    std::random_device rnd{};
    std::mt19937 gen{ static_cast<long unsigned int>(time(0)) };
    std::uniform_real_distribution<double> uniform(x_low, x_high);
    auto norm = std::normal_distribution<double >{ 0.0, error };
    VectorXd x(n), y(n);
    for (int i = 0; i < n; ++i)
    {
        x[i] = uniform(gen);
        y[i] = a * x[i] + b + norm(gen);
    }

    auto f = [&x, &y](const VectorXd& p) -> double
        {
            double a = p[0];
            double b = p[1];
            double sum = 0.0;
            for (int i = 0; i < x.size(); ++i)
            {
                double y_ = a * x[i] + b;
                sum += (y_ - y[i]) * (y_ - y[i]);
            }
            return sum;
        };
    auto g = [&x, &y](const VectorXd& p) -> VectorXd
        {
            double a = p[0];
            double b = p[1];
            VectorXd grads(2);
#if defined(USING_EIGEN)
            grads << 0.0, 0.0;
#else
            VectorXd(2, 0.0);
#endif
            for (int i = 0; i < x.size(); ++i)
            {
                double y_ = a * x[i] + b;
                grads[0] += 2 * (y_ - y[i]) * x[i];
                grads[1] += 2 * (y_ - y[i]);
            }
            return grads;
        };
    constexpr double inf = numeric_limits<double>::infinity();
#ifdef USING_EIGEN
    VectorXd params(2), lb(2), ub(2);
    params << 1., 1.;
    lb << -inf, -inf;
    ub << inf, inf;
#else
    VectorXd lb = { -inf, -inf };
    VectorXd ub = { inf, inf };
    VectorXd params = { 0.0, 0.0 };
#endif
    bool status = optimize(f, g, params, lb, ub);

    if (!status)
    {
        std::cout << "no convergence after 100 iterations \n";
    }
    std::cout << "ground truth a = " << a << " b = " << b << "\n";
    std::cout << "analytical gradient a = " << params[0] << ", b = " << params[1] << "\n";
    std::cout << "error = " << std::sqrt(f(params)/n) << "\n";
}
static int matyas()
{
    constexpr double inf = numeric_limits<double>::infinity();

    std::cout << "matyas test  - f(0,0) = 0 \n";
    auto f = [](const VectorXd& x_) -> double
        {
            double x = x_[0];
            double y = x_[1];
            double f = 0.26 * (x * x + y * y) - 0.48 * x * y;
            return f;
        };
    auto g = [](const VectorXd& x_) -> VectorXd
        {
            double x = x_[0];
            double y = x_[1];
            VectorXd g(2);
            g[0] = 0.52 * x - 0.48 * y;
            g[1] = 0.52 * y - 0.48 * x;
            return g;
        };
#if defined(USING_EIGEN)
    VectorXd x(2);
    x << -10, -1.0;
    VectorXd lb = VectorXd::Constant(2, -inf);
    VectorXd ub = VectorXd::Constant(2, inf);
#else
    VectorXd x = { -10, -1.0 };
    VectorXd lb = { -inf, -inf };
    VectorXd ub = { inf, inf };
#endif

    bool status = optimize(f, g, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";

    std::cout << "matyas test - numerical gradient\n";
    NumericalGradient n_grad(f);
#if defined(USING_EIGEN)
    x << 10, -1.;
#else
    x = { 10, -1.0 };
#endif
    status = optimize(f, n_grad, x, lb, ub);

    if (!status)
    {
        cout << "no convergence after 100 iterations \n";
    }
    cout << "Optimized position: (" << x[0] << ", " << x[1] << ", " << f(x) << ")\n";

    return 1;
}

int main()
{
    constexpr double inf = numeric_limits<double>::infinity();
  
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    rosenbrock(-inf, inf);

    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    quadratic(-20, -10, 10);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    quadratic(5, -10, 10);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    quadratic(5, 1, 10);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double x = 8. + uniform(generator);
    quadratic(x, 1, 10);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    quadratic(5., std::sin(3.1415 / 2), 10);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";


    rastrigin(-5.12, 5.12);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    rastrigin(-5.12, 5.12, 3);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    rastrigin(-5.12, 5.12, 5);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    rastrigin(-5.12, 5.12, 25);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    rastrigin(-5.12, 5.12, 100);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    himmelblau(-inf, inf);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    beale();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    threehumpcamel();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    easom();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    linear_regession(2, 3, -10, 10, 100, 0.01);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    linear_regession(-2, 3, -10, 10, 100, 0.01);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    linear_regession(2, 0, -5, 5, 100, 0.01);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    linear_regession(0, 2, -5, 5, 100, 0.01);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    linear_regession(0, 0, -5, 5, 100, 0.01);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    egg_holder();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    booth();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    mccormick();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    styblinski_tang();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    styblinski_tang(3);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    styblinski_tang(10);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    styblinski_tang(100);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    matyas();
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    schwefel(20);
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
    
    bdexp_test(2.0, 5000, -inf, inf, false);
}