#if !defined(USING_EIGEN)
#include <stdexcept>
#include <algorithm>
#include "matrix_ops.h"

vector<vector<double>> matrix_ops::identity(size_t n)
{
    vector<vector<double>> a(n);
    for (size_t i = 0; i < n; ++i)
    {
        a[i].resize(n, 0.0);
        a[i][i] = 1.0;
    }
    return a;
}
vector<vector<double>> matrix_ops::tril(const vector<vector<double>>& a, int k)
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
vector<double> matrix_ops::diag(const vector<vector<double>>& a)
{
    size_t n = a.size();
    vector<double> d = vector<double>(n);
    for (size_t i = 0; i < n; ++i)
    {
        d[i] = a[i][i];
    }
    return d;
}
vector<vector<double>> matrix_ops::diag(const vector<double>& d)
{
    size_t n = d.size();
    vector<vector<double>> dm(n, vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
    {
        dm[i][i] = d[i];
    }
    return dm;
}
void matrix_ops::initialise(vector<vector<double>>& a, const vector<double>& v)
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
void matrix_ops::add_end_column(vector<vector<double>>& a, const vector<double>& v)
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
size_t matrix_ops::num_columns(const vector<vector<double>>& a)
{
    return a.empty() ? 0 : a[0].size();
}
size_t matrix_ops::num_rows(const vector<vector<double>>& a)
{
    return a.empty() ? 0 : a.size();
}
void matrix_ops::remove_end_column(vector<vector<double>>& a)
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
vector<double> matrix_ops::scale(const vector<double>& x, double a)
{
    vector<double> y(x);
    transform(y.begin(), y.end(), y.begin(), [&a](auto& c) {return c * a; });
    return y;
}
vector<vector<double>> matrix_ops::scale(const vector<vector<double>>& x, double a)
{
    size_t n = x.size();
    vector<vector<double>> s(n);
    for (size_t i = 0; i < n; ++i)
    {
        s[i] = scale(x[i], a);
    }
    return s;
}
vector<vector<double>> matrix_ops::copy(const vector<vector<double>>& a_)
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
vector<vector<double>> matrix_ops::vconcat(const vector<vector<double>>& a, const vector<vector<double>>& b)
{
    vector<vector<double>> ab = a;
    ab.insert(ab.end(), b.begin(), b.end());
    return ab;
}
vector<vector<double>> matrix_ops::hconcat(const vector<vector<double>>& a, const vector<vector<double>>& b)
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
vector<double> matrix_ops::get_col(const vector<vector<double>>& a, size_t col)
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
vector<double> matrix_ops::get_row(const vector<vector<double>>& a, size_t row)
{
    if (row < 0 || row >= a.size())
    {
        throw out_of_range("invalid row index");
    }
    return a[row];
}
vector<vector<double>> matrix_ops::transpose(const vector<vector<double>>& a)
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
vector<double> matrix_ops::multiply(const vector<vector<double>>& a, const vector<double>& x)
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
vector<vector<double>> matrix_ops::multiply(const vector<vector<double>>& a, const vector<vector<double>>& b)
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
vector<double> matrix_ops::subtract(const vector<double>& a, const vector<double>& b)
{
    size_t n = a.size();
    vector<double> c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = a[i] - b[i];
    }
    return c;
}
vector<vector<double>> matrix_ops::subtract(const vector<vector<double>>& a, const vector<vector<double>>& b)
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
vector<double> matrix_ops::add(const vector<double>& a, const vector<double>& b)
{
    size_t n = a.size();
    vector<double> c(n);
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
    return c;
}
vector<vector<double>> matrix_ops::add(const vector<vector<double>>& a, const vector<vector<double>>& b)
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
double matrix_ops::dot(const vector<double>& a, const vector<double>& b)
{
    double sum = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif