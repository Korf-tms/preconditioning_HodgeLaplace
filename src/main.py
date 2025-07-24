import numpy as np
import pygeon as pg
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from pygeon.numerics.differentials import exterior_derivative as diff
from pygeon.numerics.innerproducts import mass_matrix, default_discr
from pygeon.numerics.stiffness import stiff_matrix


def generate_rhs(mdg: pg.MixedDimensionalGrid):
    sd = mdg.subdomains()[0]

    def f_func(x):
        return np.sum([np.sin(2 * np.pi * x_i) for x_i in x])

    def f_func_vec(x):
        return np.tile(f_func(x), 3)

    np.random.seed(0)

    f_0 = default_discr(sd, sd.dim).interpolate(sd, f_func)
    f_1 = default_discr(sd, sd.dim - 1).interpolate(sd, f_func_vec)
    f_n = default_discr(sd, 0).interpolate(sd, f_func)

    if sd.dim == 2:
        f = [f_0, f_1, f_n]
    else:
        f_2 = default_discr(sd, 1).interpolate(sd, f_func_vec)
        f = [f_0, f_1, f_2, f_n]

    return f


def assemble_block_matrices(mdg: pg.MixedDimensionalGrid):
    dim = mdg.dim_max()

    # Assemble mass and stiffness matrices
    Mass = [mass_matrix(mdg, dim - k, None) for k in range(dim + 1)]
    D = [diff(mdg, dim - k) for k in range(dim)]
    Diff = [Mass[k + 1] @ D[k] for k in range(dim)]
    Stiff = [D[k].T @ Diff[k] for k in range(dim)]
    Stiff.append(stiff_matrix(mdg, 0, None))

    return Mass, Diff, Stiff


def assemble_hodge_laplace(k, alpha, Mass, Diff, Stiff):
    A = sps.block_array(
        [[alpha * Mass[k - 1], -Diff[k - 1].T], [-Diff[k - 1], -Stiff[k]]]
    )

    return A


def assemble_rhs(mdg, Mass, k):
    f = generate_rhs(mdg)
    b = np.hstack((Mass[k - 1] @ f[k - 1], Mass[k] @ f[k]))

    return b


def solve(A, b, P):
    num_iterations = 0

    def nonlocal_iterate(arr):
        nonlocal num_iterations
        num_iterations += 1

    spla.minres(A, b, M=P, callback=nonlocal_iterate)

    return num_iterations


def create_preconditioner(k, alpha, Mass, Stiff):
    Bp = alpha * Mass[k - 1] + (alpha + 1) * Stiff[k - 1]
    Bu = Mass[k] / (alpha + 1) + Stiff[k]

    Pp = spla.splu(Bp)
    Pu = spla.splu(Bu)

    def P(r):
        rp, ru = np.split(r, [Bp.shape[0]])
        xp = Pp.solve(rp)
        xu = Pu.solve(ru)

        return np.hstack((xp, xu))

    shape = (Pp.shape[0] + Pu.shape[0], Pp.shape[0] + Pu.shape[0])

    return spla.LinearOperator(matvec=P, shape=shape)


def create_preconditioner_flipped(k, alpha, Mass, Stiff):
    Bp = alpha * Mass[k - 1] + Stiff[k - 1]
    Bu = Mass[k] + Stiff[k]

    Pp = spla.splu(Bp)
    Pu = spla.splu(Bu)

    def P(r):
        rp, ru = np.split(r, [Bp.shape[0]])
        xp = Pp.solve(rp)
        xu = Pu.solve(ru)

        return np.hstack((xp, xu))

    shape = (Pp.shape[0] + Pu.shape[0], Pp.shape[0] + Pu.shape[0])

    return spla.LinearOperator(matvec=P, shape=shape)



def make_table(h_list, alpha_list, iters, k_list):
    latex_str = "\\begin{table}[h]\n\n"
    latex_str += "\\begin{tabular}{r|"
    latex_str += "|".join(["r" * alpha_list.shape[0] for _ in k_list])
    latex_str += "}\n\\hline\n"

    latex_str += "& "

    head_start = "\\multicolumn{"
    head_start += "{}".format(alpha_list.shape[0])
    head_start += "}{|c}{"
    header = [head_start + "$k = {:}$".format(k) + "}" for k in k_list]

    latex_str += " & ".join(header)
    latex_str += "\\\\\n $h \\backslash \\log_{10}(\\alpha)$ & "

    latex_str += " & ".join(
        [
            " & ".join([str(int(np.log10(alpha))) for alpha in alpha_list])
            for _ in k_list
        ]
    )
    latex_str += " \\\\\n\\hline\n"

    formatted_rows = [" & ".join(str(num) for num in row) for row in iters]
    formatted_rows = [
        "{:.2e} & ".format(h) + row for h, row in zip(h_list, formatted_rows)
    ]
    latex_str += " \\\\\n".join(formatted_rows)
    latex_str += " \n\\end{tabular}\n\\end{table}"

    return latex_str


def get_iteration_counts(dim, k_list, alpha_list, h_list):
    iters = [np.empty((h_list.size, alpha_list.size), dtype=int) for _ in k_list]

    for i_h, h in enumerate(h_list):
        mdg = pg.unit_grid(dim, h)
        mdg.compute_geometry()

        sd = mdg.subdomains()[0]
        h_list[i_h] = np.mean(sd.cell_diameters())

        Mass, Diff, Stiff = assemble_block_matrices(mdg)

        for i_k, k in enumerate(k_list):
            b = assemble_rhs(mdg, Mass, k)
            print("k = {}, n = {}".format(k, b.size))

            for i_a, alpha in enumerate(alpha_list):
                A = assemble_hodge_laplace(k, alpha, Mass, Diff, Stiff)
                P = create_preconditioner_flipped(k, alpha, Mass, Stiff)

                iters[i_k][i_h, i_a] = solve(A, b, P)

    return np.hstack(iters)


if __name__ == "__main__":
    # Input
    dim = 3
    k_list = np.arange(1, dim + 1)
    alpha_list = 10.0 ** np.arange(-4, 12, 2)

    if dim == 2:
        # h_list = 2.0 ** (-np.arange(4, 9))
        h_list = 2.0 ** (-np.arange(4, 11))
        print(h_list)
    else:
        # h_list = 1.5 ** (-np.arange(3, 8))
        h_list = 1.5 ** (-np.arange(3, 9))

    iters = get_iteration_counts(dim, k_list, alpha_list, h_list)

    string = make_table(h_list, alpha_list, iters, k_list)

    with open("table{}D.txt".format(dim), "w") as f:
        print(string, file=f)
