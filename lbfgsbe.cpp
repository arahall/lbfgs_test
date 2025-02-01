#include <iostream>

#include <vector>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>

#include "lbfgsb.h"
#include "lu.h"
#include "matrix_ops.h"

using namespace std;
using namespace matrix_ops;

static vector<size_t> argsort(const vector<double>& x)
{
	vector <size_t> idx(x.size());
	iota(idx.begin(), idx.end(), 0);
	sort(idx.begin(), idx.end(), [&x](size_t i, size_t j) { return x[i] < x[j]; });
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

static tuple<vector<double>, vector<double>> get_cauchy_point(const vector<double>& x, const vector<double>& g,
															  const vector<double>& lb, const vector<double>& ub,
															  double theta, vector<vector<double>>& w,
															  const vector<vector<double>>& m)
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
static double get_optimality(const vector<double>& x, const vector<double>& g,
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
		return { false, xbar }; // line search not required on return
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
	vector<size_t> p(big_n.size() + 1);
	lu::decompose(big_n, p);
	v = lu::solve(big_n, p, v);
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
						 function<vector<double>(const vector<double>&)> gradient,
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
		vector<double> g_i = gradient(x);
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
						   function<vector<double>(const vector<double>&)> gradient,
						   const vector<double>& x0, double f0, const vector<double>& g0,
						   const vector<double>& p, int max_iters, double c1, double c2,
						   double alpha_max)
{
	// compute line search satisfying strong Wolfe conditions
	double f_im1 = f0, alpha_im1 = 0.0, alpha_i = 1.0;
	double dphi0 = dot(g0, p);

	for (int iter = 0; iter < max_iters; ++iter)
	{
		vector<double> x = add(x0, scale(p, alpha_i));
		double f_i = func(x);
		vector<double> g_i = gradient(x);
		if ((f_i > f0 + c1 * dphi0) || (iter > 1 && f_i >= f_im1))
		{
			return alpha_zoom(func, gradient, x0, f0, g0, p, alpha_im1, alpha_i);
		}
		double dphi = dot(g_i, p);
		if (fabs(dphi) <= -c2 * dphi0)
		{
			return alpha_i;
		}
		if (dphi >= 0.0)
		{
			return alpha_zoom(func, gradient, x0, f0, g0, p, alpha_i, alpha_im1);
		}
		// update
		alpha_im1 = alpha_i;
		f_im1 = f_i;
		alpha_i += 0.8 * (alpha_max - alpha_i);
	}
	return alpha_i;
}
static vector<vector<double>> hessian(const vector<vector<double>>& w, const vector<vector<double>>& s,
									  const vector<vector<double>>& y, double theta)
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
	try
	{
		vector<vector<double>> m = lu::inverse(mm);
		return m;
	}
	catch (std::runtime_error& e)
	{
		throw e;
	}
}
bool LBFGSB::optimize(function<double(const vector<double>&)> func,
					  function<vector<double>(const vector<double>&)> gradient,
					  vector<double>& x, const vector<double>& lb, const vector<double>& ub,
					  int max_history, int max_iter, double tol, bool debug, double c1, double c2)
{
	// func - the function to be minimised
	// gradient - the gradient of the function to be minimosed
	// x - the solution 
	// max_history - number of corrections used in the limited memeory matrix
	// max_history < 3 not recommended, large m not recommend
	// 3 <= m < 20 is the recommended range for me
	//
	size_t n = x.size(); // the problem dimension

	vector<vector<double>> y_history, s_history;
	vector<vector<double>> w(n, vector<double>(1, 0.0)), m(1, vector<double>(1, 0.0));


	double f = func(x);
	vector<double> g = gradient(x);
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
		double alpha = flag ? strong_wolfe(func, gradient, x, f, g, subtract(xbar, x), max_iter) : 1.0;
		x = add(x, scale(subtract(xbar, x), alpha));
		f = func(x);
		g = gradient(x);
		vector<double> dx = subtract(x, x_old);
		vector<double> dg = subtract(g, g_old);
		double curv = dot(dx, dg);
		if (curv < numeric_limits<double>::epsilon())
		{
			if (debug)
			{
				cout << "negative curvature detected - skipping BFGS update\n";
			}
			continue; // skip BFGS update
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
		try
		{
			m = hessian(w, s_history, y_history, theta);
		}
		catch (...)
		{
			if (debug)
			{
				cout << "hessian update failed\n";
			}
			return false;
		}
	}
	return false;
}