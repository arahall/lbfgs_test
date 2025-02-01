#if !defined(USING_EIGEN)
#include <stdexcept>
#include <cmath>

#include "lu.h"

using namespace std;
void lu::decompose(vector<vector<double>>& a, vector<size_t>& p)
{
	size_t n = a.size();
	if (p.size() != n + 1)
	{
		throw runtime_error("lu::decompose - length of p must n+1");
	}
	for (size_t i = 0; i < n+1; ++i) // p[n] is S+N where S is the number of row exchanges
	{
		p[i] = i;
	}
	for (size_t i = 0; i < n; ++i)
	{
		double max_a = 0.0;
		size_t imax = i;

		for (size_t k = i; k < n; ++k)
		{
			double abs_a = fabs(a[k][i]);
			if (abs_a > max_a)
			{
				max_a = abs_a;
				imax = k;
			}
		}
		if (max_a < numeric_limits<double>::epsilon())
		{
			throw runtime_error("matrix a is degenatate");
		}
		if (imax != i)
		{
			size_t j = p[i];
			p[i] = p[imax];
			p[imax] = j;
			vector<double> row = a[i];
			a[i] = a[imax];
			a[imax] = row;
			p[n]++;
		}
		for (size_t j = i + 1; j < n; ++j)
		{
			a[j][i] /= a[i][i];
			for (size_t k = i + 1; k < n; ++k)
			{
				a[j][k] -= a[j][i] * a[i][k];
			}
		}
	}
}
vector<double> lu::solve(const vector<vector<double>> &a, const vector<size_t> &p, const vector<double> &b)
{
	size_t n = a.size();
	vector<double> x(n);
	for (size_t i = 0; i < n; ++i)
	{
		x[i] = b[p[i]];
		for (size_t k = 0; k < i; ++k)
		{
			x[i] -= a[i][k] * x[k];
		}
	}
	for (int i = n - 1; i >= 0; --i)
	{
		for (int k = i + 1; k < n; ++k)
		{
			x[i] -= a[i][k] * x[k];
		}
		x[i] /= a[i][i];
	}
	return x;
}
vector<vector<double>> lu::inverse(vector<vector<double>>& a)
{
	size_t n = a.size();
	vector<size_t> p(n+1);
	
	decompose(a, p);

	vector<vector<double>> ia(n, vector<double>(n));

	for (size_t j = 0; j < n; ++j)
	{
		for (size_t i = 0; i < n; ++i)
		{
			ia[i][j] = p[i] == j ? 1.0 : 0.0;
			for (int k = 0; k < i; ++k)
			{
				ia[i][j] -= a[i][k] * ia[k][j];
			}
		}
		for (int i = n - 1; i >= 0; --i)
		{
			for (int k = i + 1; k < n; ++k)
			{
				ia[i][j] -= a[i][k] * ia[k][j];
			}
			ia[i][j] /= a[i][i];
		}
	}
	return ia;
}
double lu::determinant(const vector<vector<double>>& a, const vector<size_t>& p)
{
	double det = a[0][0];
	size_t n = a.size();
	for (size_t i = 0; i < n; ++i)
	{
		det *= a[i][i];
	}
	return (p[n] - n) % 2 == 0 ? det : -det;
}
#endif