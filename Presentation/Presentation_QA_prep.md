# SGD vs R-SGD — likely examiner questions after the 5-minute talk

Rep. 2 = Section 2 report (implementation), Rep. 3 = Section 3 report (simulations).

**1. Prove the 1(e) inequality. Does it guarantee that R-SGD decreases f?**
By L-smoothness, f(θ+d) ≤ f(θ) + ∇fᵀd + (L/2)‖d‖². If ∇f(θₖ) ∈ Gₖ, then ∇fᵀd ≤ max over h ∈ Gₖ of hᵀd = gᵀd + ρ√(dᵀSd). Taking the inner product of the 1(d) optimality condition with d gives gᵀd + ρ√(dᵀSd) = −‖d‖²/α. Hence f(θₖ₊₁) ≤ f(θₖ) − (1/α − L/2)‖d‖², which is a strict decrease for α < 2/L whenever d ≠ 0. The guarantee is conditional: whether the true gradient lies in Gₖ is random and cannot be checked.

**2. How is the robust step computed, and why does Newton converge?**
With diagonal S, each coordinate is dⱼ(τ) = −ατgⱼ/(τ + αρsⱼ), where τ = √(dᵀSd) solves the scalar equation Σⱼ aⱼ/(τ + cⱼ)² = 1. That equation has a root exactly when gᵀS⁻¹g > ρ², so the zero-step rule is built in. Newton's method is applied to ψ = q^(−1/2) − 1, which is concave by Cauchy–Schwarz. Started from τ = 0 it increases monotonically to the root, and it is exact in one step when S is isotropic. The cost is O(p) per iteration. On 20 000 test problems it took 4.4 iterations on average and 9 at most, with residuals ≤ 1.0·10⁻¹⁴, and it was cross-checked against CVXPY (Rep. 2).

**3. R-SGD searched 36 settings against 6 for SGD, and the settings were fixed after tuning at c = 1. Is that fair?**
The search budget favours R-SGD, because the best of many noisy estimates is optimistic (the "winner's curse"). The network shows this. Tuning predicted that R-SGD would win (0.0104 vs 0.0138), but at evaluation, at the same c = 1, it lost: validation loss 0.0230 vs 0.0114, accuracy 0.9946 vs 0.9980. Even with that advantage it did not beat step-matched SGD. Fixing the settings works against R-SGD: it carries α₀ = 10 into c ≥ 4. Hence the next experiment: tune each method at every noise level with equal budgets.

**4. Does gᵀS⁻¹g ≤ ρ² mean ∇f = 0?**
No. It only means the gradient cannot be told apart from zero at this batch size. On the noisy quadratic, zero steps occurred at f up to 2.94 (starting from f = 10), because noise hides the θ₁-gradient for |θ₁| up to ρ·2σ/√b ≈ 6.9 (Rep. 2). Under injected noise, gᵀS⁻¹g has mean ≈ p(b−1)/(b−3) whatever θ is. So once ρ ≥ √p, the network takes 35–100% zero steps and stalls at accuracy 0.50–0.92 (Rep. 3).

**5. R-SGD reached 5.8× lower training loss on the network at c = 0. Why is accuracy no higher?**
Both methods are already at about 99.8% accuracy (0.9982 vs 0.9977). The lower cross-entropy mostly means more confident outputs on points that are already classified correctly; the loss ranking predicted the accuracy ranking in only 3 of 10 seeds. Logistic regression sits at the Bayes accuracy (0.923), so no optimiser can improve it. At high noise all rankings agree, with SGD better.

**6. How did you build the step-length control, and what did it show?**
The control is SGD with α₀′ = α₀ × the mean of ‖dₖ‖/(αₖ‖gₖ‖) over the matching R-SGD run, on the same seeds; zero steps count as 0. On the network at c = 0 the ratio was 0.593, so α₀′ = 5.93. Step-matched SGD reached test loss 0.0065 against R-SGD's 0.0082, and R-SGD won only 4 of 10 seeds. The one genuine R-SGD effect is in logistic regression at low noise: it beat matched SGD by 0.0027 test loss at c = 1 (9 of 10 seeds), because its shrinkage acts as an adaptive step-size decay. That lowers the loss but not the error rate.

**7. Why divide by b(b−1), and what happens with very small or very large batches?**
With sampling with replacement, ŝⱼ is unbiased for Var(gⱼ) = Σⱼⱼ/b, the variance of the batch mean, which is exactly what Gₖ describes. Its relative spread is √(2/(b−1)): 141% at b = 2 and 9% at b = 256. At a minimiser, the mean of gᵀS⁻¹g is infinite for b ≤ 3. The robust correction shrinks like ρ/√b: it halved from 0.237 to 0.117 between b = 64 and b = 256. Sampling without replacement would overestimate S by a factor n/(n−b). Hence b = 16 here (Rep. 2).

**8. What is the main limitation, and what result would change your conclusion?**
The main limitation is tuning: 3 seeds at a single noise level, with unequal search grids. The network's R-SGD choice did not reproduce, and the logistic-regression tuning criterion was flat. The next experiment is to tune each method separately at every c with equal budgets and more seeds. The conclusion would change if R-SGD tuned at c = 16 beat SGD tuned at c = 16. The radius sweep suggests it will not: there, step-matched SGD was better (0.246 vs 0.275 in logistic regression, 0.086 vs 0.098 in the network). A second limitation is that the diagonal S ignores correlations between parameters.
