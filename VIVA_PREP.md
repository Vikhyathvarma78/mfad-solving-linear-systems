# Viva preparation – answer every question as **Concept → Purpose → Outcome**

## 1. What does your data matrix represent?
Each **row is a car**, each **column is a feature** (1 for the intercept, cylinders, displacement, horsepower, weight in lb, acceleration, model year, weight in kg). `b` is the measured mpg. So `A` is 392×8, and `Ax = b` has 392 equations in 8 unknowns.

## 2. Why is it a system of linear equations?
We assume mpg ≈ x₀ + x₁·cyl + … + x₆·year. One car gives one linear equation in the unknown coefficients x.

## 3. Part A – how can you solve Ax = b, and which is best?
| Method | Idea | Verdict (our measurements) |
|---|---|---|
| `np.linalg.solve` | PLU (Gaussian elimination with pivoting) | Fastest, accurate – **use this** |
| PLU by hand | `A = PLU`; solve `Ly = Pᵀb` forward, `Ux = y` backward | Identical result (error ~1e-16) |
| `lstsq` | SVD, minimises ‖Ax−b‖ | Same on a square system; works for any shape |
| Inverse `A⁻¹b` | Compute the full inverse | ≈4× slower at n=500; ≈150× less accurate on Hilbert n=10 |
| RREF `[A|b]` | Gauss–Jordan | ≈100× slower at n=500; accuracy similar to `solve` here (we use pivoting) |

## 4. What is the residual and why use it?
`r = Ax − b`. The exact solution is unknown in practice; a tiny ‖r‖ (≈1e-14) shows the equations are satisfied. (Caveat: on ill-conditioned problems a small residual does not guarantee a small error – the Hilbert experiment shows this.)

## 5. Why is the inverse a bad way to solve?
Computing A⁻¹ costs several times more than one LU solve and amplifies rounding error. "A⁻¹" in a textbook usually means "solve a system here".

## 6. What is PLU?
`A = PLU`: P permutation (row swaps for stability), L lower triangular, U upper triangular. Forward + backward substitution then costs only O(n²).

## 7. Over-determined vs under-determined?
* **m > n (over-determined, e.g. 392 cars, 7 unknowns):** usually inconsistent → least squares; normal equations `AᵀAx = Aᵀb`; residual ⟂ Col(A) (we verified Aᵀr ≈ 0).
* **m < n (under-determined):** infinitely many solutions = particular solution + Null(A). `pinv` gives the **minimum-norm** one. Our 3×4 example: rank 2 → nullity 2.

## 8. RREF / matrix simplification – what did it tell you?
RREF of A (392×8) had **7 pivot columns**; the `weight_kg` column was not a pivot and its RREF column read 0.4536 × weight_lb. → one redundant column.

## 9. Rank and nullity?
rank = 7, nullity = 8 − 7 = 1. The null-space vector (1, −2.2046) on (weight_lb, weight_kg) says `weight_lb − 2.2046·weight_kg = 0`. Fundamental subspaces: dim Col = 7, dim Row = 7, dim Null = 1, dim Null(Aᵀ) = 385.

## 10. Why remove the redundant column? (basis selection)
With dependent columns `AᵀA` is singular (cond ≈ 2e16), so there is no unique least-squares solution. Keeping the 7 pivot columns gives a basis of Col(A): unique solution.

## 11. Gram–Schmidt – what and why?
Turns independent columns into orthonormal columns: `qⱼ = (aⱼ − Σᵢ<ⱼ (qᵢᵀaⱼ)qᵢ)/‖·‖`, giving `A = QR`. Orthonormal columns make projection easy (`QQᵀ`) and the fit `Rx = Qᵀb` numerically stable. We compared classical (‖QᵀQ−I‖ = 2.6e-13), modified (1.7e-14) and Householder (4.7e-15): modified is better than classical.

## 12. Projection?
`b̂ = Pb`, `P = QQᵀ = A(AᵀA)⁻¹Aᵀ`. We verified: P symmetric, P² = P, trace P = 7 = rank, the error `e = b − b̂` is orthogonal to every column of A, and ‖b‖² = ‖b̂‖² + ‖e‖² (Pythagoras). b̂ is the best prediction a linear model can make.

## 13. Least squares – how and why?
`x̂` minimises ‖Ax − b‖. It is the coefficient vector of the projection. We computed it six ways (normal equations with PLU / inverse / RREF, QR, SVD, pinv); all agree to ~1e-13.

## 14. How good are the predictions?
80/20 split (313 train, 79 test): **test R² = 0.769, RMSE = 3.61 mpg, MAE = 2.69 mpg** (train R² 0.817). Matches scikit-learn to 4e-15. Typical error ≈ 3–4 mpg.

## 15. Why is the horsepower coefficient unstable?
It is −0.0004 on all data and +0.0083 on the training split: features are strongly correlated (big engines are heavy and powerful), so individual coefficients are fragile while predictions stay stable. This is explained by the eigenvalues (Q16).

## 16. Eigenvalues and eigenvectors – what did they show?
Eigen-analysis of the 6×6 feature correlation matrix: eigenvalues 4.26, 0.84, 0.67, 0.13, 0.06, 0.04. The first carries **71 %** of the variance; its eigenvector loads (same sign) on cylinders, displacement, horsepower, weight → one "size/power" factor. Condition number λmax/λmin ≈ 117.

## 17. Diagonalization – what did it simplify?
Symmetric `C = VΛVᵀ` (spectral theorem). The coupled normal equations become independent scalar equations λᵢβ̃ᵢ = ρ̃ᵢ, so `β = VΛ⁻¹Vᵀρ`. Using all 6 directions reproduces the QR solution exactly; keeping only the 4 largest (principal component regression) gives test RMSE 3.57 vs 3.61 – a simpler model, same accuracy.

## 18. Why is cond(AᵀA) a problem?
cond(AᵀA) = cond(A)² (here 7.3e9 vs 8.5e4). Forming normal equations squares the condition number, so QR/SVD are safer than the normal equations on ill-conditioned data.

## 19. How does each step connect to the next?
Data → matrix → **RREF** finds redundancy → **rank/nullity** quantifies it → **basis** removes it → **Gram–Schmidt** orthogonalises the basis → **projection** onto it → **least squares** = coefficients of that projection → **eigenvalues** explain collinearity → **diagonalization** simplifies the model → **prediction**.

## 20. What if asked about limitations?
Linear model only (mpg is not exactly linear in weight); 1970–82 cars; collinear features; R² 0.77 leaves room for non-linear or additional features.
