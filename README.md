# Solving Linear Systems in Python

**MFAD Mini Project – UE25MA242A (Mathematical Foundation for AI & Data Science), PES University**
**Application Problem 4:** *Solving linear systems in Python*

**Team:** Varun C B · Vikhyath Varma D · Varshith Bhashyam · Ruthvik Reddy | Section: H

## What this project does

| Part | Description |
|---|---|
| **Part A – Core Project 4** | Solves `Ax = b` with `np.linalg.solve`, hand-written PLU (forward/back substitution), `lstsq`, the inverse and RREF. Compares **accuracy** (residuals, Hilbert matrices) and **efficiency** (timing, n = 50…500). Also covers over-determined (least squares, normal equations) and under-determined systems (pseudoinverse, null space). |
| **Part B – Real-world application** | **Predicts car fuel efficiency (mpg)** from the *Auto-MPG* dataset (392 cars). The data is turned into a linear system and taken through the department workflow: RREF/LU → rank & nullity → basis selection → Gram–Schmidt (QR) → projection → least squares → eigenvalues → diagonalization → predictions. |

### Workflow coverage (as required by the Mini Project Guidelines)

| Workflow stage | Linear-algebra component | Notebook location |
|---|---|---|
| Real-world data | Auto-MPG (`data/auto-mpg.csv`) | Part B, Step 1 |
| Matrix representation | System of linear equations `Ax = b` | Part B, Step 2 |
| Matrix simplification | Gaussian elimination / RREF / LU | Part A Tasks 4, 7 · Part B Step 3 |
| Structure of the space | Rank & nullity, fundamental subspaces | Part B, Step 4 |
| Remove redundancy | Linear independence → basis selection | Part B, Step 5 |
| Orthogonalization | Gram–Schmidt (classical & modified) → QR | Part B, Step 6 |
| Projection | Orthogonal projection onto Col(A) | Part B, Step 7 |
| Prediction / approximation | Least squares (6 methods compared) | Part A Task 9 · Part B Step 8 |
| Pattern discovery | Eigenvalues & eigenvectors | Part B, Step 9 |
| System simplification | Diagonalization of a symmetric matrix (PCR) | Part B, Step 10 |
| Final output | mpg predictions + `predict_mpg()` | Part B, Final |

## Key results
* On a well-conditioned 5×5 system all five methods agree to ~1e-13.
* n = 500: inverse is ≈ 4× and RREF ≈ 100× slower than `solve`. On the ill-conditioned Hilbert matrix (n = 10) the inverse is ≈ 150× less accurate than `solve` (3.6e-3 vs 2.3e-5).
* Auto-MPG: the design matrix (392×8) has **rank 7, nullity 1** – the null-space vector reveals `weight_lb = 2.2046 × weight_kg`. After removing it, QR least squares gives **test R² ≈ 0.77, RMSE ≈ 3.6 mpg** on 79 unseen cars (matches scikit-learn to 1e-15).
* The feature correlation matrix has one dominant eigenvalue (71 % of variance); diagonalization / PCR with 4 directions matches the full model.

## Repository structure
```
├── Project4_Solving_Linear_Systems.ipynb   # main notebook (already executed, outputs visible)
├── project4_solving_linear_systems.py      # same code as a plain script
├── data/auto-mpg.csv                       # dataset (UCI Auto-MPG)
├── figures/                                # plots saved by the notebook
├── requirements.txt
├── VIVA_PREP.md                            # Concept → Purpose → Outcome answers
└── README.md
```

## Setup and run
Requires **Python 3.9+**.

```bash
# 1. clone
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (recommended) virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4a. run the notebook (for the live demo)
jupyter notebook Project4_Solving_Linear_Systems.ipynb      # then Kernel → Restart & Run All

# 4b. or run the script
python project4_solving_linear_systems.py
```
Runtime is under a minute (the RREF timing at n = 500 is the slowest cell). Figures are written to `figures/`.

## Notes
* Dataset: Auto-MPG, UCI Machine Learning Repository (also distributed with seaborn-data). Six rows with missing horsepower are dropped (398 → 392).
* The `weight_kg` column is added deliberately to demonstrate redundancy detection (same quantity in two units).
* Corrections to the textbook's Project 4 code: Task 4 now actually uses L and U (the book calls `solve` again); Task 10 uses a 3×4 matrix (the book's 4×3 matrix with a length-3 vector raises a shape error); RREF uses partial pivoting and a tolerance.
