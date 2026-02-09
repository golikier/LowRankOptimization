# LowRankOptimization

This repository contains Matlab implementations of the following low-rank optimization methods:
- monotone PGD [OW25, Algorithm 4.2 with l = 0 or p = 1];
- P2GD [SU15, Algorithm 3];
- P2GDR [OGA26, Definition 6.1];
- P2GD-PGD [OGA26, Definition 7.1];
- four classes of ERFD (whose iteration map is [OA24, Algorithm 3.1]), namely RFD [SU15, Algorithm 4] and CRFD [OA24, Definition 6.3] with each of the three cones from [OA24, Table 6.1];
- four classes of ERFDR [OA24, Algorithm 4.2], namely RFDR [OA23, Algorithm 3] and CRFDR [OA24, Definition 6.3] with each of the three cones from [OA24, Table 6.1];
- HRTR [LKB23, Algorithm 1].

It also contains Matlab code enabling to check the Hessian and the Matlab scripts used to realize the numerical experiments presented in [OGA26, sections 8.3 and 8.4], which compare monotone PGD, P2GD, P2GDR, P2GD-PGD, RFD, and RFDR on two weighted low-rank approximation (WLRA) problems.

Every Matlab function implements:
- the iteration map if its name contains “map”;
- the iterative method returning all iterates and function values generated over a prefixed number of iterations if its name contains “iterinfo”;
- the running time needed to bring the function value below a prefixed threshold if its name contains “time”.

These functions are called by the scripts that generated the results presented in [OGA26, sections 8.3 and 8.4]. The experiment described in [OGA26, section 8.3] focuses on the apocalyptic WLRA problem from [OGA26, section 8.2]. The results were obtained by running the scripts TimeWLRAapo.m and PlotWLRAapo.m. The experiment described in [OGA26, section 8.4] focuses on a matrix completion problem similar to that presented in [SU15, section 3.4]. The results were obtained by running the scripts TimeMatrixCompletion.m and PlotMatrixCompletion.m. 

The script CheckHessianLKB23.m calls the functions CheckHessian.m and CheckHessianLift.m.

References

[SU15] https://doi.org/10.1137/140957822

[LKB23] https://link.springer.com/article/10.1007/s10107-022-01851

[OA23] https://doi.org/10.1137/22M1518256

[OW25] https://doi.org/10.1137/24M1692782

[OGA26] https://arxiv.org/abs/2201.03962v3

[OA24] https://arxiv.org/abs/2409.12298
