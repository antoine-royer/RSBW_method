""" ComPer QUEST 3.2 """

""" Combined Perturbations-based Quantum Evaluator for State Thresholds """

""" Packages """
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display
import warnings

""" WARNING PLEASE REMOVE OR USE # UNTIL YOU THINK THE CODE IS PERFECT FOR YOU """
""" REMOVE ERROR NOTIFICATIONS """
warnings.filterwarnings("ignore")

# %matplotlib inline


def ExactHamiltonian(E1, E2, V11, h12, V22, t1, t2, U):
    """
    Diagonalization of the Exact Hamiltonian

    Parameters
    ----------
    E1, E2
        energy of the unperturbed states
    V11, V22
        internal coupling of the model states
    h12
        coupling between the unperturbed states
    U
        energy of the perturbator
    t1, t2
        coupling between the unperturbed states and the perturbator
    """

    exact_hamiltonian = np.array(
        [[E1 + V11, h12, t1], [h12, E2 + V22, t2], [t1, t2, U]]
    )

    # returns the eigenvalues and the eigenvectors respectively (eigh means that we impose the
    # argument to be hermitian)
    exact_energy, exact_eigenket = sci.linalg.eigh(exact_hamiltonian)

    perturbation = np.array([[V11, h12, t1], [h12, V22, t2], [t1, t2, 0]])
    zero_order_hamiltonian = np.array([[E1, 0, 0], [0, E2, 0], [0, 0, U]])
    # defines the rho parameter for the Rayleigh-Schrödinger procedure
    rho_RS = np.linalg.norm(perturbation, ord=None) / np.linalg.norm(
        zero_order_hamiltonian, ord=None
    )

    return exact_hamiltonian, exact_energy, exact_eigenket, rho_RS


def EffectiveHamiltonian(E1, E2, V11, h12, V22, t1, t2, U):
    """
    Diagonalization of the Effective Hamiltonian / RS perturbation

    Parameters
    ----------
    E1, E2
        energy of the unperturbed states
    V11, V22
        internal coupling of the model states
    h12
        coupling between the unperturbed states
    U
        energy of the perturbator
    t1, t2
        coupling between the unperturbed states and the perturbator
    """

    # construction of the Effective Hamiltonian operating on the model space
    energy_mean = (E1 + E2) / 2
    cell11 = E1 + V11 + (t1**2 / (energy_mean - U))
    cell22 = E2 + V22 + (t2**2 / (energy_mean - U))
    cell12 = h12 + (t1 * t2 / (energy_mean - U))

    effective_hamiltonian = np.array([[cell11, cell12], [cell12, cell22]])

    # obtention of the eigenvalues and eigenvectors respectively (eigh means that we impose the
    # argument to be hermitian)
    model_energy, model_ket = sci.linalg.eigh(effective_hamiltonian)

    # construction of the RS Hamiltonian operating on the total space
    RS_hamiltonian = np.zeros((3, 3))
    RS_hamiltonian[:2, :2] = effective_hamiltonian
    RS_hamiltonian[-1, -1] = U

    return RS_hamiltonian, model_energy, model_ket


def HuckelHamiltonian(E1, E2, V11, h12, V22, t1, t2, U):
    """
    Diagonalization of the Hückel Hamiltonian

    Parameters
    ----------
    E1, E2
        energy of the unperturbed states
    V11, V22
        internal coupling of the model states
    h12
        coupling between the unperturbed states
    U
        energy of the perturbator
    t1, t2
        coupling between the unperturbed states and the perturbator
    """

    # construction of the Hückel Hamiltonian operating on the model space
    cell11 = E1 + V11
    cell22 = E2 + V22
    cell12 = h12

    restricted_huckel_hamiltonian = np.array([[cell11, cell12], [cell12, cell22]])

    # obtention of the eigenvalues and eigenvectors respectively (eigh means that we impose the
    # argument to be hermitian)
    huckel_energy, huckel_ket = sci.linalg.eigh(restricted_huckel_hamiltonian)

    # construction of the Hückel Hamiltonian operating on the total space
    extended_huckel_hamiltonian = np.zeros((3, 3))
    extended_huckel_hamiltonian[:2, :2] = restricted_huckel_hamiltonian
    extended_huckel_hamiltonian[-1, -1] = U

    return extended_huckel_hamiltonian, huckel_energy, huckel_ket


def BasisTransformation(hamiltonian, transition_submatrix):
    """
    Obtention of the total Hamiltonian in the model kets basis

    Parameters
    ----------
    hamiltonian
        the total Hamiltonian, expressed in the unperturbed basis
    transition_submatrix
        transition matrix on the model space from the unperturbed basis to the
    hückel or model kets basis

    transition_matrix
        total transition matrix from the unperturbed basis to the hückel or model
    kets basis
    """

    transition_matrix = np.zeros((3, 3))
    transition_matrix[:2, :2] = transition_submatrix
    transition_matrix[-1, -1] = 1

    return np.matmul(
        np.matmul(np.transpose(transition_matrix), hamiltonian), transition_matrix
    )


def BWPerturbation(exact_hamiltonian, model_energy, E1, E2, U, t1, t2, nb_iterations=1):
    """
    Computation of the energies after the BW perturbation

    Parameters
    ----------
    exact_hamiltonian
        the total hamiltonian, expressed in the eigenbasis of the Zeroth-Order
    Hamiltonian
    model_energy
        list of the eigenvalues of the Zeroth-Order Hamiltonian
    U
        energy of the perturbator
    nb_iterations
        number of iteration to optimize the energy

    second_order_perturbation
        list containing, for each state, a list containing, for each
    iteration of the BW self-consistent procedure, the different second-order corrections to the
    energy
    """

    nb_states = len(model_energy)
    perturbed_energy = []
    perturbation_contributions = []
    self_consistency_warning = []

    for state_loop in range(
        nb_states
    ):  # computes the energy of each state, one for each value of state_loop

        sc_energy = model_energy[state_loop]
        # the variable whose value will be updated through each iteration of the BW
        # self-consistent procedure, and whose final value will be the correct BW energy
        perturbation_per_iteration = []

        for BW_iteration in range(
            nb_iterations
        ):  # iterates the self-consistent procedure to solve the BW perturbation problem

            # second-order coupling with the perturbator
            betacoupling = 0
            diff = sc_energy - U
            matrix_element_beta = exact_hamiltonian[state_loop][-1]
            if abs(diff / matrix_element_beta) > 0.15:
                betacoupling = matrix_element_beta**2 / diff
                # print(f"Perturbator coupling ratio (must be low): {matrix_element_beta/diff}")
            else:
                self_consistency_warning.append(
                    f"Warning state {state_loop+1}: low ratio energy gap to matrix element :"
                    f"{diff/matrix_element_beta}. Coupling was ignored in this iteration"
                )

            # second-order coupling with the other model state
            model_state_coupling = 0
            for coupling_loop in range(nb_states):
                # for 2-dimensional model space, this loop is useless. Could be used as a basis to
                # improve or generalize current program
                matrix_element_mixing = exact_hamiltonian[state_loop][coupling_loop]
                if (t1 != t2 or E1 != E2) and state_loop != coupling_loop:
                    diff = sc_energy - model_energy[coupling_loop]
                    # print(
                    #     f"Model space coupling ratio (must be > 0.15): "\
                    #     f"{diff/matrix_element_mixing}"
                    # )
                    if abs(diff / matrix_element_mixing) > 0.15:
                        model_state_coupling += matrix_element_mixing**2 / diff
                    else:
                        self_consistency_warning.append(
                            f"Warning coupling {state_loop+1}-{coupling_loop+1}: low ratio energy "
                            f"gap to matrix element: {diff/matrix_element_mixing}. Coupling was "
                            f"ignored in this iteration"
                        )

            # self-consistent energy
            sc_energy = (
                exact_hamiltonian[state_loop][state_loop]
                + betacoupling
                + model_state_coupling
            )
            check_energy = (
                sc_energy - exact_hamiltonian[state_loop][state_loop]
            ) / exact_hamiltonian[state_loop][state_loop]
            if check_energy > 3:
                self_consistency_warning.append(
                    f"Warning iteration {BW_iteration}: energy correction seems big. Relative gap:"
                    f" {check_energy}"
                )

            perturbation_per_iteration.append(
                [
                    sc_energy,
                    exact_hamiltonian[state_loop][state_loop],
                    matrix_element_beta,
                    matrix_element_mixing,
                    betacoupling,
                    model_state_coupling,
                ]
            )

        perturbed_energy.append(sc_energy)
        perturbation_contributions.append(perturbation_per_iteration)

    return perturbed_energy, perturbation_contributions, self_consistency_warning


def CheckPerturbation(exact_hamiltonian, zero_order_hamiltonian):
    """
    Obtention of the rho parameters

    Parameters
    ----------
    exact_hamiltonian
        total hamiltonian operating on the total space
    zero_order_hamiltonian
        any zeroth order Hamiltonian operating on the total space
    """

    rho = np.linalg.norm(
        exact_hamiltonian - zero_order_hamiltonian, ord=None
    ) / np.linalg.norm(zero_order_hamiltonian, ord=None)

    warning = []

    if rho > 1:
        warning.append(f"Error: perturbation bigger than zeroth order. rho = {rho}")

    elif rho > 0.9:
        warning.append(f"Warning: perturbation seems big. rho = {rho}")

    return rho, warning


def Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations=1):
    """
    Recuperation of all the data

    Parameters
    ----------
    E1, E2
        energy of the unperturbed states
    V11, V22
        internal coupling of the model states
    h12
        coupling between the unperturbed states
    U
        energy of the perturbator
    t1, t2
        coupling between the unperturbed states and the perturbator
    nb_iterations
        number of iteration to optimize the energy

    model_energy
        list of the eigenvalues of the Effective Hamiltonian
    model_ket
        list of the eigenkets of the Effective Hamiltonian, each represented in the
    unperturbed basis
    huckel_energy
        list of the eigenvalues of the Hückel Hamiltonian
    huckel_ket
        list of the eigenkets of the Hückel Hamiltonian
    """

    ### Diagonalization of the Exact Hamiltonian ###
    exact_hamiltonian_UBasis, exact_energy, exact_state, rho_RS = ExactHamiltonian(
        E1, E2, V11, h12, V22, t1, t2, U
    )

    ### Diagonalization of the Hückel Hamiltonian ###
    huckel_hamiltonian, huckel_energy, huckel_ket = HuckelHamiltonian(
        E1, E2, V11, h12, V22, t1, t2, U
    )

    ### Diagonalization of the Effective Hamiltonian ###
    effective_hamiltonian, RS_perturbed_energy, model_ket = EffectiveHamiltonian(
        E1, E2, V11, h12, V22, t1, t2, U
    )

    ### Hamiltonian in the Hückel basis ###

    exact_hamiltonian_HBasis = BasisTransformation(exact_hamiltonian_UBasis, huckel_ket)

    ### Hamiltonian in the model kets basis ###

    exact_hamiltonian_MBasis = BasisTransformation(exact_hamiltonian_UBasis, model_ket)

    ### Brillouin-Wigner Perturbation for Hückel Hamiltonian ###

    # print("HBW procedure")
    (
        HBW_perturbed_energy,
        HBW_perturbation_contributions,
        self_consistency_warning_HBW,
    ) = BWPerturbation(
        exact_hamiltonian_HBasis, huckel_energy, E1, E2, U, t1, t2, nb_iterations
    )

    ### Brillouin-Wigner Perturbation for Effective Hamiltonian ###

    # print("RSBW procedure")
    (
        RSBW_perturbed_energy,
        RSBW_perturbation_contributions,
        self_consistency_warning_RSBW,
    ) = BWPerturbation(
        exact_hamiltonian_MBasis, RS_perturbed_energy, E1, E2, U, t1, t2, nb_iterations
    )

    ### Checks and procedure validity ###

    perturbation_status_HBW = CheckPerturbation(
        exact_hamiltonian_UBasis, huckel_hamiltonian
    )
    perturbation_status_RSBW = CheckPerturbation(
        exact_hamiltonian_UBasis, effective_hamiltonian
    )
    rho_list = [rho_RS, perturbation_status_HBW[0], perturbation_status_RSBW[0]]
    warning_list = (
        ["HBW procedure"]
        + perturbation_status_HBW[1]
        + self_consistency_warning_HBW
        + ["RSBW procedure"]
        + perturbation_status_RSBW[1]
        + self_consistency_warning_RSBW
    )

    perturbation_contributions = (
        HBW_perturbation_contributions,
        RSBW_perturbation_contributions,
    )

    return (
        exact_energy,
        RSBW_perturbed_energy,
        HBW_perturbed_energy,
        RS_perturbed_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    )


""" Plotting functions of U """


def PlotUEnergy(E1, E2, V11, V22, h12, t1, t2, U_min, U_max, nb_iterations):

    xaxis = []
    RSBW_yaxis1 = []  # list containing the RSBW energy of state 1 as a function of U
    RSBW_yaxis2 = []  # list containing the RSBW energy of state 2 as a function of U
    HBW_yaxis1 = []  # list containing the HBW energy of state 1 as a function of U
    HBW_yaxis2 = []  # list containing the HBW energy of state 2 as a function of U
    RS_yaxis1 = []  # list containing the RS energy of state 1 as a function of U
    RS_yaxis2 = []  # list containing the RS energy of state 2 as a function of U
    exact_yaxis1 = []  # list containing the exact energy of state 1 as a function of U
    exact_yaxis2 = []  # list containing the exact energy of state 2 as a function of U

    step = (U_max - U_min) / 20

    for index in range(21):
        U = step * index + U_min
        (
            exact_energy,
            RSBW_perturbed_energy,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
        xaxis.append(U)
        RSBW_yaxis1.append(RSBW_perturbed_energy[0])
        RSBW_yaxis2.append(RSBW_perturbed_energy[1])
        HBW_yaxis1.append(HBW_perturbed_energy[0])
        HBW_yaxis2.append(HBW_perturbed_energy[1])
        RS_yaxis1.append(RS_perturbed_energy[0])
        RS_yaxis2.append(RS_perturbed_energy[1])
        exact_yaxis1.append(exact_energy[0])
        exact_yaxis2.append(exact_energy[1])

        # print(f"U = {U}, {huckelBW_perturbed_energy[1]}")
        # print(f"U = {U}, {exact_energy[0]}")
        # print(f"U = {U}, {exact_energy[1]}")

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"U = {U}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    plt.rc("font", family="serif")  # Type of font
    plt.rc("xtick", labelsize="20")  # Size of the xtick label
    plt.rc("ytick", labelsize="20")  # Size of the ytick label
    plt.rc("lines", linewidth="3")  # Width of the curves
    plt.rc("legend", framealpha="1")  # Transparency of the legend frame
    plt.rc("legend", fontsize="23")  # Size of the legend
    plt.rc("grid", linestyle="--")  # Grid formed by dashed lines
    plt.rcParams.update(
        {"text.usetex": True}
    )  # Using LaTex style for text and equation

    """ First graph """

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, exact_yaxis1, "k-", label="Exact energy")
    fig1.plot(xaxis, RS_yaxis1, "g:", label="RS energy")
    fig1.plot(xaxis, HBW_yaxis1, "r--", label="BW energy")
    fig1.plot(xaxis, RSBW_yaxis1, "b-.", label="RSBW energy")

    fig1.set_xlabel("U", fontsize=23)
    fig1.set_ylabel("Energy of ground state", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, exact_yaxis2, "k-", label="Exact energy")
    fig2.plot(xaxis, RS_yaxis2, "g:", label="RS energy")
    fig2.plot(xaxis, HBW_yaxis2, "r--", label="BW energy")
    fig2.plot(xaxis, RSBW_yaxis2, "b-.", label="RSBW energy")

    fig2.set_xlabel("U", fontsize=23)
    fig2.set_ylabel("Energy of excited state", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()
    plt.show()


def PlotURelGap(E1, E2, V11, V22, h12, t1, t2, U_min, U_max, nb_iterations):

    xaxis = []

    # list containing the relative gap between RSBW and exact energy of state 1, as a function of U
    reldiff_RSBW_1 = []

    # list containing the relative gap between RSBW and exact energy of state 2, as a function of U
    reldiff_RSBW_2 = []

    # list containing the relative gap between HBW and exact energy of state 1, as a function of U
    reldiff_HBW_1 = []

    # list containing the relative gap between HBW and exact energy of state 2, as a function of U
    reldiff_HBW_2 = []

    # list containing the relative gap between RS and exact energy of state 1, as a function of U
    reldiff_RS_1 = []

    # list containing the relative gap between RS and exact energy of state 2, as a function of U
    reldiff_RS_2 = []
    step = (U_max - U_min) / 20

    for index in range(21):
        U = step * index + U_min
        (
            exact_energy,
            RSBW_perturbed_energy,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
        xaxis.append(U)
        reldiff_RSBW_1.append(
            abs((RSBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        reldiff_RSBW_2.append(
            abs((RSBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )
        reldiff_HBW_1.append(
            abs((HBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        reldiff_HBW_2.append(
            abs((HBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )
        reldiff_RS_1.append(
            abs((RS_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        reldiff_RS_2.append(
            abs((RS_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"U = {U}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    print("Relative gap RSBW state 1, Relative gap RSBW state 2")
    for index in range(21):
        print(
            f"U = {step*index + U_min} {reldiff_RSBW_1[index]}, {reldiff_RSBW_2[index]}"
        )

    plt.rc("font", family="serif")  # Type of font
    plt.rc("xtick", labelsize="20")  # Size of the xtick label
    plt.rc("ytick", labelsize="20")  # Size of the ytick label
    plt.rc("lines", linewidth="3")  # Width of the curves
    plt.rc("legend", framealpha="1")  # Transparency of the legend frame
    plt.rc("legend", fontsize="23")  # Size of the legend
    plt.rc("grid", linestyle="--")  # Grid formed by dashed lines
    plt.rcParams.update(
        {"text.usetex": True}
    )  # Using LaTex style for text and equation

    """ First graph """

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, reldiff_RS_1, "g:", label="RS procedure")
    fig1.plot(xaxis, reldiff_HBW_1, "r--", label="BW procedure")
    fig1.plot(xaxis, reldiff_RSBW_1, "b-.", label="RSBW procedure")

    fig1.set_xlabel("U", fontsize=23)
    fig1.set_ylabel(
        "Relative gap between approximate\nand exact energy of ground state",
        fontsize=23,
    )
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, reldiff_RS_2, "g:", label="RS procedure")
    fig2.plot(xaxis, reldiff_HBW_2, "r--", label="BW procedure")
    fig2.plot(xaxis, reldiff_RSBW_2, "b-.", label="RSBW procedure")

    fig2.set_xlabel("U", fontsize=23)
    fig2.set_ylabel(
        "Relative gap between approximate\nand exact energy of excited state",
        fontsize=23,
    )
    fig2.legend(loc="best")

    plt.tight_layout()
    plt.show()


def PlotURho(E1, E2, V11, V22, h12, t1, t2, U_min, U_max, nb_iterations):

    xaxis = []
    rho_RSBW = []  # list containing the RSBW rho parameter, as a function of U
    rho_HBW = []  # list containing the HBW rho parameter, as a function of U
    rho_RS = []  # list containing the RS rho parameter, as a function of U

    step = (U_max - U_min) / 20

    for index in range(21):
        U = step * index + U_min
        (
            exact_energy,
            RSBW_perturbed_energy,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
        xaxis.append(U)
        rho_RS.append(rho_list[0])
        rho_HBW.append(rho_list[1])
        rho_RSBW.append(rho_list[2])

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"U = {U}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    plt.rc("font", family="serif")  # Type of font
    plt.rc("xtick", labelsize="20")  # Size of the xtick label
    plt.rc("ytick", labelsize="20")  # Size of the ytick label
    plt.rc("lines", linewidth="3")  # Width of the curves
    plt.rc("legend", framealpha="1")  # Transparency of the legend frame
    plt.rc("legend", fontsize="23")  # Size of the legend
    plt.rc("grid", linestyle="--")  # Grid formed by dashed lines
    plt.rcParams.update(
        {"text.usetex": True}
    )  # Using LaTex style for text and equation

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, rho_RS, "g:", label="RS procedure")
    fig1.plot(xaxis, rho_HBW, "r--", label="BW procedure")
    fig1.plot(xaxis, rho_RSBW, "b-.", label="RSBW procedure")

    fig1.set_xlabel("U", fontsize=23)
    fig1.set_ylabel("$\\rho$ parameter", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()
    plt.show()


def DisplayFunctionU(displayed_info):

    E1 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="E1")
    E2 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="E2")
    V11 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V11")
    V22 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V22")
    h12 = widgets.FloatSlider(min=-4, max=0, value=-1, step=0.05, description="h12")
    t1 = widgets.FloatSlider(min=-3, max=0, value=-1, step=0.1, description="t1")
    t2 = widgets.FloatSlider(min=-3, max=0, value=-1, step=0.1, description="t2")
    # Number of iterations for BW self-consistent procedure
    nb_iterations = widgets.IntSlider(
        min=1, max=10, value=1, step=1, description="iterations"
    )
    # Changes the displayed window
    U_min = widgets.IntSlider(min=2, max=20, value=3, step=1, description="U_min")
    U_max = widgets.IntSlider(min=4, max=100, value=10, step=1, description="U_max")

    energy = widgets.HBox([E1, E2])
    vinterne = widgets.HBox([V11, V22])
    couplage = widgets.HBox([h12, t1, t2])
    optimisation = widgets.HBox([nb_iterations, U_min, U_max])

    if displayed_info == 0:
        out = widgets.interactive_output(
            PlotUEnergy,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t1": t1,
                "t2": t2,
                "U_min": U_min,
                "U_max": U_max,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 1:
        out = widgets.interactive_output(
            PlotURelGap,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t1": t1,
                "t2": t2,
                "U_min": U_min,
                "U_max": U_max,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 2:
        out = widgets.interactive_output(
            PlotURho,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t1": t1,
                "t2": t2,
                "U_min": U_min,
                "U_max": U_max,
                "nb_iterations": nb_iterations,
            },
        )

    display(energy, vinterne, couplage, optimisation, out)


""" Plotting functions of h12 """


def PlotH12Energy(E1, E2, U, V11, V22, t1, t2, h12_min, h12_max, nb_iterations):

    xaxis = []
    RSBW_yaxis1 = []  # list containing the RSBW energy of state 1 as a function of h12
    RSBW_yaxis2 = []  # list containing the RSBW energy of state 2 as a function of h12
    HBW_yaxis1 = []  # list containing the HBW energy of state 1 as a function of h12
    HBW_yaxis2 = []  # list containing the HBW energy of state 2 as a function of h12
    RS_yaxis1 = []  # list containing the RS energy of state 1 as a function of h12
    RS_yaxis2 = []  # list containing the RS energy of state 2 as a function of h12
    exact_yaxis1 = (
        []
    )  # list containing the exact energy of state 1 as a function of h12
    exact_yaxis2 = (
        []
    )  # list containing the exact energy of state 2 as a function of U h12

    step = (h12_max - h12_min) / 20

    for index in range(21):
        h12 = step * index + h12_min
        (
            exact_energy,
            RSBW_perturbed_energy,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
        xaxis.append(h12)
        RSBW_yaxis1.append(RSBW_perturbed_energy[0])
        RSBW_yaxis2.append(RSBW_perturbed_energy[1])
        HBW_yaxis1.append(HBW_perturbed_energy[0])
        HBW_yaxis2.append(HBW_perturbed_energy[1])
        RS_yaxis1.append(RS_perturbed_energy[0])
        RS_yaxis2.append(RS_perturbed_energy[1])
        exact_yaxis1.append(exact_energy[0])
        exact_yaxis2.append(exact_energy[1])

        # print(f"h12 = {h12}, {huckelBW_perturbed_energy[1]}")
        # print(f"h12 = {h12}, {exact_energy[0]}")
        # print(f"h12 = {h12}, {exact_energy[1]}")

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"h12 = {h12}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    # np.save('data6.npy', [xaxis, exact_yaxis1, RSBW_yaxis1, exact_yaxis2, RSBW_yaxis2])

    plt.rc("font", family="serif")  # Type of font
    plt.rc("xtick", labelsize="20")  # Size of the xtick label
    plt.rc("ytick", labelsize="20")  # Size of the ytick label
    plt.rc("lines", linewidth="3")  # Width of the curves
    plt.rc("legend", framealpha="1")  # Transparency of the legend frame
    plt.rc("legend", fontsize="23")  # Size of the legend
    plt.rc("grid", linestyle="--")  # Grid formed by dashed lines
    plt.rcParams.update(
        {"text.usetex": True}
    )  # Using LaTex style for text and equation

    """ First graph """

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, exact_yaxis1, "k-", label="Exact energy")
    fig1.plot(xaxis, RS_yaxis1, "g:", label="RS energy")
    fig1.plot(xaxis, HBW_yaxis1, "r--", label="BW energy")
    fig1.plot(xaxis, RSBW_yaxis1, "b-.", label="RSBW energy")

    fig1.set_xlabel("K", fontsize=23)
    fig1.set_ylabel("Energy of ground state", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, exact_yaxis2, "k-", label="Exact energy")
    fig2.plot(xaxis, RS_yaxis2, "g:", label="RS energy")
    fig2.plot(xaxis, HBW_yaxis2, "r--", label="BW energy")
    fig2.plot(xaxis, RSBW_yaxis2, "b-.", label="RSBW energy")

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel("Energy of excited state", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()

    plt.show()


def PlotH12RelGap(E1, E2, U, V11, V22, t1, t2, h12_min, h12_max, nb_iterations):

    xaxis = []
    # list containing the relative gap between RSBW and exact energy of state 1, as a function of
    # h12
    reldiff_RSBW_1 = []

    # list containing the relative gap between RSBW and exact energy of state 2, as a function of
    # h12
    reldiff_RSBW_2 = []

    # list containing the relative gap between HBW and exact energy of state 1, as a function of h12
    reldiff_HBW_1 = []

    # list containing the relative gap between HBW and exact energy of state 2, as a function of h12
    reldiff_HBW_2 = []

    # list containing the relative gap between RS and exact energy of state 1, as a function of h12
    reldiff_RS_1 = []

    # list containing the relative gap between RS and exact energy of state 2, as a function of h12
    reldiff_RS_2 = []

    step = (h12_max - h12_min) / 20

    for index in range(21):
        h12 = step * index + h12_min
        (
            exact_energy,
            RSBW_perturbed_energy,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
        xaxis.append(h12)
        reldiff_RSBW_1.append(
            abs((RSBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        reldiff_RSBW_2.append(
            abs((RSBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )
        reldiff_HBW_1.append(
            abs((HBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        reldiff_HBW_2.append(
            abs((HBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )
        reldiff_RS_1.append(
            abs((RS_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        reldiff_RS_2.append(
            abs((RS_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )

        if abs(exact_energy[0]) < 0.05:
            print(
                f"h12 = {h12} ERROR state 1: exact energy too low for the relative gap to be "
                f"significant\n"
            )
        if abs(exact_energy[1]) < 0.05:
            print(
                f"h12 = {h12} ERROR state 2: exact energy too low for the relative gap to be "
                f"significant\n"
            )

        # [0] for HBW, [1] for RSBW, [n][0] for ground state, [n][1] for excited state
        print(f"h12 = {h12} perturbation: {perturbation_contributions[0][0]}")

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"h12 = {h12}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    print("Relative gap RSBW state 1, Relative gap RSBW state 2")
    for index in range(21):
        print(
            f"h12 = {step*index + h12_min} {reldiff_RSBW_1[index]}, {reldiff_RSBW_2[index]}"
        )

    # np.save('data9.npy', [xaxis, reldiff_RSBW_1])

    plt.rc("font", family="serif")  # Type of font
    plt.rc("xtick", labelsize="20")  # Size of the xtick label
    plt.rc("ytick", labelsize="20")  # Size of the ytick label
    plt.rc("lines", linewidth="3")  # Width of the curves
    plt.rc("legend", framealpha="1")  # Transparency of the legend frame
    plt.rc("legend", fontsize="23")  # Size of the legend
    plt.rc("grid", linestyle="--")  # Grid formed by dashed lines
    plt.rcParams.update(
        {"text.usetex": True}
    )  # Using LaTex style for text and equation

    """ First graph """

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, reldiff_RS_1, "g:", label="RS procedure")
    fig1.plot(xaxis, reldiff_HBW_1, "r--", label="BW procedure")
    fig1.plot(xaxis, reldiff_RSBW_1, "b-.", label="RSBW procedure")

    fig1.set_xlabel("K", fontsize=23)
    fig1.set_ylabel(
        "Relative gap between approximate\nand exact energy of ground state",
        fontsize=23,
    )
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, reldiff_RS_2, "g:", label="RS procedure")
    fig2.plot(xaxis, reldiff_HBW_2, "r--", label="BW procedure")
    fig2.plot(xaxis, reldiff_RSBW_2, "b-.", label="RSBW procedure")

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel(
        "Relative gap between approximate\nand exact energy of excited state",
        fontsize=23,
    )
    fig2.legend(loc="best")

    plt.tight_layout()

    plt.show()


def PlotH12Rho(E1, E2, U, V11, V22, t1, t2, h12_min, h12_max, nb_iterations):

    xaxis = []
    rho_RS = []  # list containing the RS rho parameter, as a function of h12
    rho_HBW = []  # list containing the HBW rho parameter, as a function of h12
    rho_RSBW = []  # list containing the RSBW rho parameter, as a function of h12

    step = (h12_max - h12_min) / 20

    for index in range(21):
        h12 = step * index + h12_min
        (
            exact_energy,
            RSBW_perturbed_energy,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
        xaxis.append(h12)
        rho_RS.append(rho_list[0])
        rho_HBW.append(rho_list[1])
        rho_RSBW.append(rho_list[2])

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"h12 = {h12}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    plt.rc("font", family="serif")  # Type of font
    plt.rc("xtick", labelsize="20")  # Size of the xtick label
    plt.rc("ytick", labelsize="20")  # Size of the ytick label
    plt.rc("lines", linewidth="3")  # Width of the curves
    plt.rc("legend", framealpha="1")  # Transparency of the legend frame
    plt.rc("legend", fontsize="23")  # Size of the legend
    plt.rc("grid", linestyle="--")  # Grid formed by dashed lines
    plt.rcParams.update(
        {"text.usetex": True}
    )  # Using LaTex style for text and equation

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, rho_RS, "g:", label="RS procedure")
    fig1.plot(xaxis, rho_HBW, "r--", label="BW procedure")
    fig1.plot(xaxis, rho_RSBW, "b-.", label="RSBW procedure")

    fig1.set_xlabel("K", fontsize=23)
    fig1.set_ylabel("$\\rho$ parameter", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()

    plt.show()


def DisplayFunctionH12(displayed_info):

    E1 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="E1")
    E2 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="E2")
    U = widgets.FloatSlider(min=0, max=10, value=2, step=0.1, description="U")
    V11 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V11")
    V22 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V22")
    t1 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t1")
    t2 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t2")
    # Number of iterations for BW self-consistent procedure
    nb_iterations = widgets.IntSlider(
        min=1, max=10, value=1, step=1, description="iterations"
    )
    # Changes the information displayed
    h12_min = widgets.FloatSlider(
        min=-5, max=-0.5, value=-1.5, step=0.1, description="h12_min"
    )
    h12_max = widgets.FloatSlider(
        min=-3, max=0, value=-0.5, step=0.1, description="h12_max"
    )

    energy = widgets.HBox([E1, E2, U])
    vinterne = widgets.HBox([V11, V22])
    couplage = widgets.HBox([t1, t2])
    optimisation = widgets.HBox([nb_iterations, h12_min, h12_max])

    if displayed_info == 0:
        out = widgets.interactive_output(
            PlotH12Energy,
            {
                "E1": E1,
                "E2": E2,
                "U": U,
                "V11": V11,
                "V22": V22,
                "t1": t1,
                "t2": t2,
                "h12_min": h12_min,
                "h12_max": h12_max,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 1:
        out = widgets.interactive_output(
            PlotH12RelGap,
            {
                "E1": E1,
                "E2": E2,
                "U": U,
                "V11": V11,
                "V22": V22,
                "t1": t1,
                "t2": t2,
                "h12_min": h12_min,
                "h12_max": h12_max,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 2:
        out = widgets.interactive_output(
            PlotH12Rho,
            {
                "E1": E1,
                "E2": E2,
                "U": U,
                "V11": V11,
                "V22": V22,
                "t1": t1,
                "t2": t2,
                "h12_min": h12_min,
                "h12_max": h12_max,
                "nb_iterations": nb_iterations,
            },
        )

    display(energy, vinterne, couplage, optimisation, out)


""" Display of miscellaneous values """


def PrintValuesEnergy(E1, E2, U, V11, V22, h12, t1, t2, nb_iterations):
    (
        exact_energy,
        RSBW_perturbed_energy,
        HBW_perturbed_energy,
        RS_perturbed_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
    print(f"Exact energy: {exact_energy}")
    print(f"RS energy: {RS_perturbed_energy}")
    print(f"HBW energy: {HBW_perturbed_energy}")
    print(f"RSBW energy: {RSBW_perturbed_energy}")


def PrintValuesRelGap(E1, E2, U, V11, V22, h12, t1, t2, nb_iterations):
    (
        exact_energy,
        RSBW_perturbed_energy,
        huckelBW_perturbed_energy,
        RS_perturbed_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)

    to_print = [
        abs((RS_perturbed_energy[0] - exact_energy[0]) / exact_energy[0]),
        abs((RS_perturbed_energy[1] - exact_energy[1]) / exact_energy[1]),
    ]
    print(f"RS relative gap: {to_print}")

    to_print = [
        abs((huckelBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0]),
        abs((huckelBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1]),
    ]
    print(f"HBW relative gap: {to_print}")

    to_print = [
        abs((RSBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0]),
        abs((RSBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1]),
    ]
    print(f"RSBW relative gap {to_print}")


def PrintValuesRho(E1, E2, U, V11, V22, h12, t1, t2, nb_iterations):
    (
        exact_energy,
        RSBW_perturbed_energy,
        huckelBW_perturbed_energy,
        RS_perturbed_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    ) = Results(E1, E2, V11, h12, V22, t1, t2, U, nb_iterations)
    print(f"rho RS: {rho_list[0]}")
    print(f"rho HBW: {rho_list[1]}")
    print(f"rho RSBW: {rho_list[2]}")
    print(f"rho RSBW / rho RS: {rho_list[2]/rho_list[0]}")
    print(f"rho RSBW / rho HBW: {rho_list[2]/rho_list[1]}")


def DisplayValues(displayed_info):

    E1 = widgets.FloatSlider(min=0, max=5, value=0, step=0.1, description="E1")
    E2 = widgets.FloatSlider(min=0, max=5, value=0, step=0.1, description="E2")
    U = widgets.FloatSlider(min=0, max=10, value=2, step=0.1, description="U")
    V11 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V11")
    V22 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V22")
    h12 = widgets.FloatSlider(min=-4, max=0, value=-1, step=0.05, description="h12")
    t1 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t1")
    t2 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t2")
    # Number of iterations for BW self-consistent procedure
    nb_iterations = widgets.IntSlider(
        min=1, max=10, value=1, step=1, description="iterations"
    )

    energy = widgets.HBox([E1, E2, U])
    vinterne = widgets.HBox([V11, V22])
    couplage = widgets.HBox([h12, t1, t2])
    optimisation = widgets.HBox([nb_iterations])

    if displayed_info == 0:
        out = widgets.interactive_output(
            PrintValuesEnergy,
            {
                "E1": E1,
                "E2": E2,
                "U": U,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t1": t1,
                "t2": t2,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 1:
        out = widgets.interactive_output(
            PrintValuesRelGap,
            {
                "E1": E1,
                "E2": E2,
                "U": U,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t1": t1,
                "t2": t2,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 2:
        out = widgets.interactive_output(
            PrintValuesRho,
            {
                "E1": E1,
                "E2": E2,
                "U": U,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t1": t1,
                "t2": t2,
                "nb_iterations": nb_iterations,
            },
        )

    display(energy, vinterne, couplage, optimisation, out)


def DisplayChoice(display_mode, displayed_info):
    """Displays the results"""
    if display_mode == 0:
        DisplayValues(displayed_info)
    elif display_mode == 1:
        DisplayFunctionH12(displayed_info)
    elif display_mode == 2:
        DisplayFunctionU(displayed_info)


print("display mode: Values/Functions of h12/Functions of U")
print("displayed info: Energy/Relative gap/Rho parameters")
display_mode = widgets.IntSlider(
    min=0, max=2, value=1, step=1, description="display mode"
)
displayed_info = widgets.IntSlider(
    min=0, max=2, value=1, step=1, description="displayed info"
)

out = widgets.interactive_output(
    DisplayChoice, {"display_mode": display_mode, "displayed_info": displayed_info}
)

display(display_mode, displayed_info, out)
