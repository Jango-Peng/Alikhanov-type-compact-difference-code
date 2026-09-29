# Alikhanov-type-compact-difference-code
Alikhanov-type compact difference method for variable-exponent time-fractional Burgers' equation

# Example 1: Variable-Order Fractional Burgers Solver

This folder contains MATLAB implementations of an Alikhanov-type compact-difference method for a variable-exponent fractional Burgers problem. Two time-history treatments are provided:

- **Direct scheme**: evaluates the history terms by explicit summation.
- **Fast scheme**: uses a positive-weight SOE for the constant-order Caputo kernel and a signed-SOE approximation for the perturbation-kernel history.

The spatial discretization uses a compact second-derivative relation together with a centered first-derivative matrix. The nonlinear algebraic system at each time level is solved by a Newton method.


## Quick start

The following example uses the parameter set used by the CPU comparison script.

```matlab
clear;

L = 1;
T = 1;
M = 128;
N = 256;
alpha0 = 0.5;
r = 2/alpha0;
nu = 0.01;
show_progress = false;

[U_direct,h_direct,t_direct,cpu_direct,newton_direct,residual_direct] = ...
    Direct_Main(L,T,M,N,alpha0,r,nu,show_progress);

[U_fast,h_fast,t_fast,cpu_fast,newton_fast,residual_fast,...
    Q_alpha,Q_kappa,caputo_error,kappa_error,...
    fast_alikhanov_bound,fast_alikhanov_ok] = ...
    Fast_Main(L,T,M,N,alpha0,r,nu,1e-10,1e-8,show_progress);
```

The solution array uses the convention

```matlab
U(:,n+1) = U^n
```

where each column contains the solution at all interior spatial nodes. The fourth output is the measured CPU time. Newton iteration counts and final residual norms are returned as the fifth and sixth outputs.

## General variable-order profile

`Fast_Main` accepts optional function handles after the `show_progress` argument. If they are omitted, the default profile is

```matlab
alpha(t) = alpha0 + 0.2*t^(1+alpha0)
```

with its analytical derivative. A user-defined profile can be supplied as follows:

```matlab
alpha_function = @(q) alpha0 + 0.2*q.^(1+alpha0);
dalpha_function = @(q) 0.2*(1+alpha0)*q.^alpha0;

[U,h,t,cpu_time,newton_iterations,newton_residuals,...
    Q_alpha,Q_kappa,caputo_error,kappa_error,...
    fast_alikhanov_bound,fast_alikhanov_ok] = ...
    Fast_Main(L,T,M,N,alpha0,r,nu,1e-10,1e-8,false,...
    alpha_function,dalpha_function);
```

The code checks that the variable order remains strictly between zero and one on its sampling points.

## File organization

### Direct scheme

- `Direct_Main.m`: direct-history time stepping and compact spatial discretization.
- `Direct_Newton.m`: standard Newton solver for one time level.
- `Direct_Residual_Jacobian.m`: nonlinear residual and analytical Jacobian.
- `Direct_Time_Weights.m`: direct nonuniform L2-1-sigma and perturbation-history weights.
- `Direct_Time_Order.m`: temporal self-convergence experiment.
- `Direct_Space_Order.m`: spatial self-convergence experiment.

### Fast scheme

- `Fast_Main.m`: fast-history time stepping.
- `Fast_Prepare_Time.m`: time grid, SOE coefficients, kernel data, and recurrence arrays.
- `Fast_History_Begin.m`: assemble the known history contribution at the current level.
- `Fast_History_Commit.m`: update the Caputo-SOE and signed-SOE history states.
- `Fast_Newton.m`: Newton solver for the fast scheme.
- `Fast_Residual_Jacobian.m`: nonlinear residual and analytical Jacobian for the fast scheme.
- `Fast_Time_Order.m`: temporal self-convergence experiment.
- `Fast_Space_Order.m`: spatial self-convergence experiment.

### Kernel and problem-data utilities

- `Alpha_Kernels.m`: evaluate the variable-order perturbation kernel and its derivative.
- `Fixed_SOE.m`: construct the positive-weight SOE for the constant-order Caputo kernel.
- `Signed_SOE.m`: fit a signed sum of exponentials to a real-valued perturbation kernel.
- `G_Tilde.m`: evaluate the example perturbation kernel for the default order profile.
- `Source_Term.m`: evaluate the source term associated with the example solution.
- `Exact_Solution.m`: provide the solution used by the convergence-data setup.

### Comparison script

- `Direct_Fast_CPU_Comparison.m`: compare direct and fast CPU times as the number of time levels increases.

## Signed-SOE interpretation

The signed-SOE routine approximates the perturbation kernel on a prescribed interval `[delta_min,T]` by

```text
kappa(q) ~= sum_l omega_kappa(l)*exp(-s_kappa(l)*q).
```

The rates `s_kappa` are nonnegative, but the coefficients `omega_kappa` may be positive or negative. Therefore, signed-SOE must not be treated as a positive-weight SOE: positivity-based energy, monotonicity, stability, or convergence conclusions do not follow automatically.

`Signed_SOE.m` uses separate training and validation point sets. Candidate exponential counts are tested using a truncated SVD, and the reported fitting error is the maximum absolute error on the independent validation grid. This numerical check is a finite-grid diagnostic; it does not by itself establish a uniform approximation theorem or an unconditional stability result for the complete fast solver.

## Convergence and performance scripts

Run a script from this directory, for example:

```matlab
run('Direct_Time_Order.m');
run('Fast_Time_Order.m');
run('Direct_Space_Order.m');
run('Fast_Space_Order.m');
run('Direct_Fast_CPU_Comparison.m');
```

The convergence scripts compare solutions on nested grids. Unless an independently verified exact error is supplied, the resulting differences should be interpreted as self-convergence diagnostics rather than exact discretization errors.

The CPU comparison includes SOE construction and fast-history initialization in the fast timing. Total runtime also depends on Newton iterations, spatial linear algebra, and the selected SOE tolerances.

## Numerical assumptions

- `0 < alpha0 < 1`.
- The variable-order function must satisfy `0 < alpha(t) < 1` over the relevant time interval.
- The graded-mesh parameter is normally selected as `r = 2/alpha0` for this example.
- `epsilon_alpha` controls the positive Caputo SOE approximation.
- `epsilon_kappa` controls the signed-SOE validation target for the perturbation kernel.
- The fast scheme keeps the local contribution separate from the recursively updated far history.
