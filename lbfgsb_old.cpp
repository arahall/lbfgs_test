#include <iostream>

#include <vector>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include "lbfgsb.h"

using namespace std;

static vector<vector<double>> identity(size_t n)
{
    vector<vector<double>> a(n);
    for (size_t i = 0; i < n; ++i)
    {
        a[i].resize(n, 0.0);
        a[i][i] = 1.0;
    }
    return a;
}
static vector<vector<double>> tril(const vector<vector<double>>& a, int k = 0)
{
    // returns lower triangle of a
    size_t n = a.size();
    size_t m = a[0].size();
    vector<vector<double>> l(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            if (j <= i + k)
            {
                l[i][j] = a[i][j];
            }
        }
    }
    return l;
}
static vector<double> diag(const vector<vector<double>>& a)
{
    size_t n = a.size();
    vector<double> d = vector<double>(n);
    for (size_t i = 0; i < n; ++i)
    {
        d[i] = a[i][i];
    }
    return d;
}
static vector<vector<double>> diag(const vector<double>& d)
{
    size_t n = d.size();
    vector<vector<double>> dm(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
    {
        dm[i][i] = d[i];
    }
    return dm;
}
static void initialise(vector<vector<double>>& a, const vector<double>& v)
{
    if (!a.empty())
    {
        throw runtime_error("initialise matrix - a must be empty");
    }
    a = vector<vector<double>>(v.size(), vector<double>(1));
    for (size_t i = 0; i < v.size(); ++i)
    {
        a[i][0] = v[i];
    }
}
static void add_end_column(vector<vector<double>>& a, const vector<double>& v)
{
    if (a.empty())
    {
        initialise(a, v);
    }
    else
    {
        size_t num_rows = a.size();
        if (a.size() != v.size())
        {
            throw runtime_error("add_end_column: number of rows in a must be the same as size of v");
        }
        for (size_t i = 0; i < num_rows; ++i)
        {
            a[i].push_back(v[i]);
        }
    }
}
static size_t num_columns(const vector<vector<double>>& a)
{
    return a.empty() ? 0 : a[0].size();
}
static size_t num_rows(const vector<vector<double>>& a)
{
    return a.empty() ? 0 : a.size();
}
static void remove_end_column(vector<vector<double>>& a)
{
    if (a.empty() || a[0].empty())
    {
        throw runtime_error("matrix a is empty or has no columns");
    }
    for (auto& row : a)
    {
        row.pop_back();
    }
}
static vector<double> scale(const vector<double>& x, double a)
{
    vector<double> y(x);
    transform(y.begin(), y.end(), y.begin(), [&a](auto& c) {return c * a; });
    return y;
}
static vector<vector<double>> scale(const vector<vector<double>>& x, double a)
{
    size_t n = x.size();
    vector<vector<double>> s(n);
    for (size_t i = 0; i < n; ++i)
    {
        s[i] = scale(x[i], a);
    }
    return s;
}
static vector<vector<double>> copy(const vector<vector<double>>& a_)
{
    if (a_.empty())
    {
        throw runtime_error("copy: matrix a is empty");
    }
    size_t n = a_.size();
    vector<vector<double>> a(n);
    for (size_t i = 0; i < n; ++i)
    {
        a[i].resize(a_[i].size());
        for (size_t j = 0; j < n; ++j)
        {
            a[i][j] = a_[i][j];
        }
    }
    return a;
}
static vector<double> gaussian_elimination(vector<vector<double>> &a, vector<double> &b)
{
    size_t n = a.size();

    // Forward elimination
    for (size_t i = 0; i < n; ++i) 
    {
        // Pivot: find the row with the largest element in the current column
        for (size_t k = i + 1; k < n; ++k) 
        {
            double factor = a[k][i] / a[i][i];
            for (size_t j = i; j < n; ++j) 
            {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) 
    {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) 
        {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }

    return x;
}
static vector<vector<double>> vconcat(const vector<vector<double>>& a, const vector<vector<double>>& b)
{
    vector<vector<double>> ab = a;
    ab.insert(ab.end(), b.begin(), b.end());
    return ab;
}
static vector<vector<double>> hconcat(const vector<vector<double>>& a, const vector<vector<double>>& b) 
{
    size_t n = a.size();
    vector<vector<double>> result(n);
    for (size_t i = 0; i < n; ++i) 
    {
        result[i].insert(result[i].end(), a[i].begin(), a[i].end());
        result[i].insert(result[i].end(), b[i].begin(), b[i].end());
    }
    return result;
}
static vector<double> get_col(const vector<vector<double>>& a, size_t col)
{
    if (a.empty() || col < 0 || col >= a[0].size()) 
    {
        throw out_of_range("Invalid column index!");
    }

    vector<double> column;
    for (const auto& row : a) 
    {
        column.push_back(row[col]);
    }

    return column;
}
static vector<double> get_row(const vector<vector<double>>& a, size_t row)
{
    if (row < 0 || row >= a.size())
    {
        throw out_of_range("invalid row index");
    }
    return a[row];
}
static vector<vector<double>> transpose(const vector<vector<double>>& a)
{
    size_t n = a.size(), m = a[0].size();
    vector<vector<double>> t(m);
    for (size_t i = 0; i < m; ++i)
    {
        t[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            t[i][j] = a[j][i];
        }
    }
    return t;
}
static vector<double> multiply(const vector<vector<double>>& a, const vector<double>& x)
{
    // mulliply a matrix of size (m, n) by a vector of size n
    size_t m = a.size();
    if (a.empty() || m == 0)
    {
        throw runtime_error("multiply - matrix a is empty");
    }
    size_t n = a[0].size();
    if (n != x.size())
    {
        throw runtime_error("multiply - number of cols in a must be the same as the number of rows in x");
    }
    vector<double> b(m);
    for (size_t i = 0; i < m; ++i)
    {
        b[i] = 0.0;
        for (size_t j = 0; j < n; ++j)
        {
            b[i] += a[i][j] * x[j];
        }
    }
    return b;
}
static vector<vector<double>> multiply(const vector<vector<double>>& a, const vector<vector<double>>& b)
{
    // multiply two matrices
    size_t rows_a = a.size(), cols_a = a[0].size();
    size_t rows_b = b.size(), cols_b = b[0].size();
    if (rows_a != cols_b)
    {
        throw invalid_argument("Number of columns in a must equal the number of rows in b");
    }
    vector<vector<double>> c(rows_a);
    for (size_t i = 0; i < rows_a; ++i)
    {
        c[i].resize(cols_b, 0.0);
        for (size_t j = 0; j < cols_b; ++j)
        {
            for (size_t k = 0; k < cols_a; ++k)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}
static vector<double> subtract(const vector<double>& a, const vector<double>& b)
{
    size_t n = a.size();
    vector<double> c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = a[i] - b[i];
    }
    return c;
}
static vector<vector<double>> subtract(const vector<vector<double>>& a, const vector<vector<double>>& b)
{
    // subtract two matrices
    size_t m = a.size(); // number of rows
    size_t n = a[0].size(); // number of columns
    vector<vector<double>> c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    return c;
}
static vector<double> add(const vector<double>& a, const vector<double>& b)
{
    size_t n = a.size();
    vector<double> c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
    return c;
}
static vector<vector<double>> add(const vector<vector<double>>& a, const vector<vector<double>>& b)
{
    // add two matrices
    size_t m = a.size(); // number of rows
    size_t n = a[0].size(); // number of columns
    vector<vector<double>> c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i].resize(n);
        for (size_t j = 0; j < n; ++j)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}
static double dot(const vector<double>& a, const vector<double>& b)
{
    double sum = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}
static vector<size_t> argsort(const vector<double>& x)
{
    vector <size_t> idx(x.size());
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&x](size_t i, size_t j) { return x[i] < x[j]; });
    return idx;
}
//
// LBFGSB 
//
static tuple<vector<double>, vector<double>> get_break_points(const vector<double>& x, const vector<double>& g, 
                                                              const vector<double>& l, const vector<double>& u)
{
    // returns the break point vector and the search direction
    size_t n = x.size();
    vector<double> t(n), d(n);

    d = scale(g, -1.0);
    for (int i = 0; i < n; ++i)
    {
        if (g[i] < 0.0)
        {
            t[i] = (x[i] - u[i]) / g[i];
        }
        else if (g[i] > 0.)
        {
            t[i] = (x[i] - l[i]) / g[i];
        }
        else
        {
            t[i] = numeric_limits<double>::max();
        }
        if (t[i] < numeric_limits<double>::epsilon())
        {
            d[i] = 0.0;
        }
    }
    return { t, d };
}

static tuple<vector<double>, vector<double>> get_cauchy_point(const vector<double>& x, const vector<double> &g, 
                            const vector<double>& lb, const vector<double>& ub,
                            double theta, vector<vector<double>> &w, 
                            const vector<vector<double>> &m)
{
    // w(n, 2l)
    // m(2l, 2l)
    size_t n = x.size();
    tuple<vector<double>, vector<double>> bp = get_break_points(x, g, lb, ub);
    vector<double> tt = get<0>(bp);
    vector<double> d = get<1>(bp);
    //
    vector<size_t> indices = argsort(tt);
    vector<double> xc(x);

    vector<double> p = multiply(transpose(w), d);
    size_t cols = w[0].size();
    vector<double> c(cols, 0.0);
    double fp = -dot(d, d);
    double fpp = -theta * fp - dot(p, multiply(m, p));
    double fpp0 = -theta * fp;
    double dt_min = -fp / fpp;
    double t_old = 0.0;
    int i = 0;
    size_t b = indices[i];
    double t = tt[b];
    double dt = t - t_old;
    // examine the rest of the segments
    while (dt_min > dt && i < n)
    {
        if (d[b] > 0.0)
        {
            xc[b] = ub[b];
        }
        else if (d[b] < 0.0)
        {
            xc[b] = lb[b];
        }
        double zb = xc[b] - x[b];
        c = add(c, scale(p, dt));
        double gb = g[b];
        vector<double> wb = get_row(w, b);
        fp += dt * fpp + gb * gb + theta * gb * zb - gb * dot(wb, multiply(m, c));
        fpp -= theta * gb * gb + 2.0 * gb * dot(wb, multiply(m, p)) + gb * gb * dot(wb, multiply(m, wb));
        fpp = max(numeric_limits<double>::epsilon() * fpp0, fpp);
        p = add(p, scale(wb, gb));
        d[b] = 0.0;
        dt_min = -fp / fpp;
        t_old = t;
        ++i;
        if (i < n)
        {
            b = indices[i];
            t = tt[b];
            dt = t - t_old;
        }
    }
    // perform final updates
    dt_min = max(dt_min, 0.0);
    t_old += dt_min;
    for (size_t j = i; j < xc.size(); ++j)
    {
        size_t idx = indices[j];
        xc[idx] += t_old * d[idx];
    }
    c = add(c, scale(p, dt_min));
    return make_tuple(xc, c);
}
static double get_optimality(const vector<double>& x, const vector<double> &g, 
                                const vector<double>& l, const vector<double>& u)
{
    size_t n = x.size();
    vector<double> projected_g(n);
    for (int i = 0; i < n; ++i)
    {
        projected_g[i] = fabs(min(max(l[i], x[i] - g[i]), u[i]) - x[i]);
    }
    auto it = max_element(projected_g.begin(), projected_g.end());
    return *it;
}
static double find_alpha(const vector<double>& l, const vector<double>& u,
                            const vector<double>& xc, const vector<double>& du,
                            const vector<size_t>& free_vars_idx)
{
    double alpha_star = 1.0;
    size_t n = free_vars_idx.size();
    for (size_t i = 0; i < n; ++i)
    {
        size_t idx = free_vars_idx[i];
        if (du[i] > 0)
        {
            alpha_star = min(alpha_star, (u[idx] - xc[idx]) / du[i]);
        }
        else
        {
            alpha_star = min(alpha_star, (l[idx] - xc[idx]) / du[i]);
        }
    }
    return alpha_star;
}
static tuple<bool, vector<double>> subspace_minimisation(const vector<double>& x, const vector<double>& g,
                                    const vector<double>& l, const vector<double>& u,
                                    const vector<double>& xc, const vector<double>& c,
                                    double theta, const vector<vector<double>>& w,
                                    const vector<vector<double>>& m)
{
    size_t n = x.size();
    vector<size_t> free_vars_index;
    for (size_t i = 0; i < n; ++i)
    {
        if (xc[i] > l[i] && xc[i] < u[i])
        {
            free_vars_index.push_back(i);
        }
    }
    size_t num_free_vars = free_vars_index.size();
    if (num_free_vars == 0)
    {
        vector<double> xbar(xc);
        return {false, xbar}; // line search not required on return
    }
    vector<vector<double>> wz(num_free_vars);
    for (size_t i = 0; i < num_free_vars; ++i)
    {
        wz[i].resize(c.size());
        size_t idx = free_vars_index[i];
            for (size_t j = 0; j < c.size(); ++j)
        {
            wz[i][j] = w[idx][j];
        }
    }
    vector<vector<double>> wtz = transpose(wz);
    // compute the reduced gradient of mk restricted to free variables
    // rr = g + theta * (xc - x) - w *(m*c)
    vector<double> temp1 = scale(subtract(xc, x), theta);
    vector<double> temp2 = multiply(w, multiply(m, c));
    vector<double> rr = add(g, subtract(temp1, temp2));
    vector<double> r(num_free_vars);
    for (int i = 0; i < num_free_vars; ++i)
    {
        r[i] = rr[free_vars_index[i]];
    }
    // form intermediate variables
    double one_over_theta = 1.0 / theta;
    vector<double> v = multiply(m, multiply(wtz, r));
    vector<vector<double>> big_n = scale(multiply(wtz, wz), one_over_theta);
    big_n = subtract(identity(big_n.size()), multiply(m, big_n));
    v = gaussian_elimination(big_n, v);
    vector<double> du = add(scale(r, -one_over_theta), scale(multiply(wz, v), -one_over_theta * one_over_theta));
    
    // find alpha star
    double alpha_star = find_alpha(l, u, xc, du, free_vars_index);
    
    // compute the subspace minimisation
    vector<double> xbar(xc);
    for (size_t i = 0; i < num_free_vars; ++i)
    {
        size_t idx = free_vars_index[i];
        xbar[idx] += alpha_star * du[i];
    }
    return { true, xbar };
}
static double alpha_zoom(function<double(const vector<double>&)> func, 
                            function<vector<double>(const vector<double>&)> grad_func,
                            const vector<double>& x0,
                            double f0, const vector<double>& g0, const vector<double>& p, double alpha_lo, double alpha_hi,
                            int max_iters = 20, double c1 = 1e-4, double c2 = 0.9)
{
    double dphi0 = dot(g0, p);
    double alpha;

    for (int i = 0; i < max_iters; ++i)
    {
        double alpha_i = 0.5 * (alpha_lo + alpha_hi);
        alpha = alpha_i;
        vector<double> x = add(x0, scale(p, alpha_i));
        double f_i = func(x);
        vector<double> g_i = grad_func(x);
        vector<double> x_lo = add(x0, scale(p, alpha_lo));
        double f_lo = func(x_lo);

        if ((f_i > f0 + c1 * alpha_i * dphi0) || (f_i >= f_lo))
        {
            alpha_hi = alpha_i;
        }
        else
        {
            double dphi = dot(g_i, p);
            if (fabs(dphi) <= -c2 * dphi0)
            {
                return alpha_i;
            }
            if (dphi * (alpha_hi - alpha_lo) >= 0)
            {
                alpha_hi = alpha_lo;
            }
            alpha_lo = alpha_i;
        }
    }

    // Final fallback if the loop exits without setting alpha
    return 0.5 * (alpha_lo + alpha_hi);
}
static double strong_wolfe(function<double(const vector<double>&)> func, 
                            function<vector<double>(const vector<double>&)> grad_func, 
                            const vector<double>& x0, double f0, const vector<double>& g0, 
                            const vector<double>& p, int max_iters=20, double c1 = 1e-4, double c2 = 0.9,
                           double alpha_max = 2.5)
{
    // compute line search satisfying strong Wolfe conditions
    double f_im1 = f0, alpha_im1 = 0.0, alpha_i = 1.0;
    double dphi0 = dot(g0, p);

    for (int iter = 0; iter < max_iters; ++iter)
    {
        vector<double> x = add(x0, scale(p, alpha_i));
        double f_i = func(x);
        vector<double> g_i = grad_func(x);
        if ((f_i > f0 + c1 * dphi0) || (iter > 1 && f_i >= f_im1) )
        {
            return alpha_zoom(func, grad_func, x0, f0, g0, p, alpha_im1, alpha_i);
        }
        double dphi = dot(g_i, p);
        if (fabs(dphi) <= -c2 * dphi0)
        {
            return alpha_i;
        }
        if (dphi >= 0.0)
        {
            return alpha_zoom(func, grad_func, x0, f0, g0, p, alpha_i, alpha_im1);
        }
        // update
        alpha_im1 = alpha_i;
        f_im1 = f_i;
        alpha_i += 0.8 * (alpha_max - alpha_i);
    }
    return alpha_i;
}
static vector<vector<double>> inverse(const vector<vector<double>>& a) 
{
    size_t  n = a.size();
    vector<vector<double>> augmented(n, vector<double>(2 * n, 0.0));

    // Create an augmented matrix [A | I]
    for (size_t i = 0; i < n; ++i) 
    {
        for (size_t j = 0; j < n; ++j) 
        {
            augmented[i][j] = a[i][j];    // Copy A to the left side
        }
        augmented[i][n + i] = 1.0;        // Add identity matrix on the right side
    }

    // Perform Gaussian elimination
    for (size_t i = 0; i < n; ++i) 
    {
        // Check for non-zero pivot element
        if (fabs(augmented[i][i]) < numeric_limits<double>::epsilon())
        {
            throw runtime_error("Matrix is singular and cannot be inverted.");
        }

        // Normalize the pivot row
        double pivot = augmented[i][i];
        for (size_t j = 0; j < 2 * n; ++j) 
        {
            augmented[i][j] /= pivot;
        }

        // Eliminate the other rows
        for (size_t k = 0; k < n; ++k) 
        {
            if (k != i) 
            {
                double factor = augmented[k][i];
                for (size_t j = 0; j < 2 * n; ++j) 
                {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    vector<vector<double>> inverse(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) 
    {
        for (size_t j = 0; j < n; ++j) 
        {
            inverse[i][j] = augmented[i][n + j];  // Extract the right side
        }
    }

    return inverse;
}
static vector<vector<double>> hessian(const vector<vector<double>>& w, const vector<vector<double>>& s,
                                        const vector<vector<double>>&y,  double theta)
{
    vector<vector<double>> st = transpose(s);
    vector<vector<double>> a = multiply(st, y);
    vector<vector<double>> l = tril(a, -1);
    vector<vector<double>> d = scale(diag(diag(a)), -1);
    // form the upper part
    vector<vector<double>> lt = transpose(l);
    vector<vector<double>> sts = multiply(st, s);
    vector<vector<double>> top = hconcat(d, lt);
    vector<vector<double>> bottom = hconcat(l, scale(sts, theta));
    vector<vector<double>> mm = vconcat(top, bottom);
    vector<vector<double>> m = inverse(mm);
    return m;
}
static vector<double> gradient(const vector<double>& x, 
                               function<double(const vector<double>&)> func,
                               function<vector<double>(const vector<double>&)> grad_func, 
                               const double epsilon=1e-2)
{
    if (grad_func)
    {
        return grad_func(x);
    }
    else
    {
        vector<double> grad(x.size());
        vector<double> x_shift = x;

        for (size_t i = 0; i < x.size(); ++i) 
        {
            double original_value = x[i];
            x_shift[i] = original_value * (1 + epsilon);
            double f_plus = func(x_shift);

            x_shift[i] = original_value * (1 - epsilon);
            double f_minus = func(x_shift);

            grad[i] = (f_plus - f_minus) / (2 * epsilon * original_value);
            x_shift[i] = original_value;  // Restore the original value
        }
        return grad;
    }
}
bool LBFGSB::optimize(function<double(const vector<double>&)> func,
                function<vector<double>(const vector<double>&)> grad_func,
                vector<double>& x, const vector<double>& lb, const vector<double>& ub,
                int max_history, int max_iter, double tol, bool debug, double c1, double c2, double epsilon)
{
    // func - the function to be minimised
    // gfunc - the gradient of the function to be minimosed
    // x - the solution 
    // max_history - number of corrections used in the limited memeory matrix
    // max_history < 3 not recommended, large m not recommend
    // 3 <= m < 20 is the recommended range for me
    //
    size_t n = x.size(); // the problem dimension

    vector<vector<double>> y_history, s_history;
    vector<vector<double>> w(n, vector<double>(1,0.0)), m(1, vector<double>(1,0.0));


    double f = func(x);
    vector<double> g = grad_func(x);
    if (g.size() != n)
    {
        throw runtime_error("LBFGSB::optimise - len of gradient must be the same as the problem dimension");
    }
    double theta = 1.0;
    for (int iter = 0; iter < max_iter; ++iter)
    {
        double opt = get_optimality(x, g, lb, ub);
        if (debug)
        {
            cout << "optimality = " << opt << " func = " << f << "\n";
            for (const auto& x_ : x)
            {
                cout << x_ << " ";
            }
            cout << "\n";
        }
        if (opt < tol)
        {
            if (debug)
            {
                cout << "converged in " << iter << " iterations\n";
            }
            return true;
        }
        vector<double> x_old(x);
        vector<double> g_old(g);

        // compute new search directon
        tuple<vector<double>, vector<double>> cp = get_cauchy_point(x, g, lb, ub, theta, w, m);
        vector<double> xc = get<0>(cp);
        vector<double> c = get<1>(cp);
        tuple<bool, vector<double>> sm = subspace_minimisation(x, g, lb, ub, xc, c, theta, w, m);
        bool flag = get<0>(sm);
        vector<double> xbar = get<1>(sm);
        double alpha = flag ? strong_wolfe(func, grad_func, x, f, g, subtract(xbar, x), max_iter) : 1.0;
        x = add(x, scale(subtract(xbar,x), alpha));
        f = func(x);
        g = grad_func(x);
        vector<double> dx = subtract(x, x_old);
        vector<double> dg = subtract(g, g_old);
        double curv = dot(dx, dg);
        if (curv < 0)
        {
            // negative curvature, skip l-bfgs update
            continue;
        }
        if (num_columns(y_history) == max_history)
        {
            remove_end_column(y_history);
            remove_end_column(s_history);
        }
        add_end_column(y_history, dg);
        add_end_column(s_history, dx);
        theta = dot(dg, dg) / dot(dg, dx);
        w = hconcat(y_history, scale(s_history, theta));
        m = hessian(w, s_history, y_history, theta);
    }
    return false;
}