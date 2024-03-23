""" ComPer QUEST 3.4 """

""" Combined Perturbations-based Quantum Evaluator for State Thresholds """
""" TWO PERTURBERS SYSTEM """


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


def ExactHamiltonian(E1, E2, U1, U2, V11, h12, V22, t11, t12, t21, t22, Vbb):
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
    U1, U2
        energy of the perturbators
    t11, t12
        coupling between the unperturbed states and the perturbator |beta 1>
    t21, t22
        coupling between the unperturbed states and the perturbator |beta 2>
    Vbb
        coupling between the perturbators
    """

    energy_mean = (E1 + E2) / 2

    zero_order_hamiltonian = np.array(
        [[energy_mean, 0, 0, 0], [0, energy_mean, 0, 0], [0, 0, U1, 0], [0, 0, 0, U2]]
    )

    initial_perturbation = np.array(
        [
            [V11, h12, t11, t21],
            [h12, V22, t12, t22],
            [t11, t12, 0, Vbb],
            [t21, t22, Vbb, 0],
        ]
    )

    exact_hamiltonian = zero_order_hamiltonian + initial_perturbation

    # returns the eigenvalues and the eigenvectors respectively (eigh means that we impose the
    # argument to be hermitian)
    exact_energy, exact_eigenket = sci.linalg.eigh(exact_hamiltonian)

    # defines the rho parameter for the Rayleigh-Schrödinger procedure
    rho_RS = np.linalg.norm(initial_perturbation, ord=None) / np.linalg.norm(
        zero_order_hamiltonian, ord=None
    )

    return exact_hamiltonian, exact_energy, exact_eigenket, rho_RS


def EffectiveHamiltonian(E1, E2, U1, U2, V11, h12, V22, t11, t12, t21, t22, Vbb):
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
    U1, U2
        energy of the perturbators
    t11, t12
        coupling between the unperturbed states and the perturbator |beta 1>
    t21, t22
        coupling between the unperturbed states and the perturbator |beta 2>
    Vbb
        coupling between the perturbators
    """

    # construction of the Effective Hamiltonian operating on the model space
    energy_mean = (E1 + E2) / 2
    cell11 = E1 + V11 + t11**2 / (energy_mean - U1) + t21**2 / (energy_mean - U2)
    cell22 = E2 + V22 + t12**2 / (energy_mean - U1) + t22**2 / (energy_mean - U2)
    cell12 = h12 + t11 * t12 / (energy_mean - U1) + t21 * t22 / (energy_mean - U2)

    effective_hamiltonian = np.array([[cell11, cell12], [cell12, cell22]])

    # obtention of the eigenvalues and eigenvectors respectively (eigh means that we impose the
    # argument to be hermitian)
    model_energy, model_ket = sci.linalg.eigh(effective_hamiltonian)

    # construction of the RS Hamiltonian operating on the total space
    RS_hamiltonian = np.zeros((4, 4))
    RS_hamiltonian[:2, :2] = effective_hamiltonian
    RS_hamiltonian[2, 2] = U1
    RS_hamiltonian[3, 3] = U2

    RS_energy = [model_energy[0], model_energy[1], U1, U2]

    return RS_hamiltonian, RS_energy, model_ket


def HuckelHamiltonian(E1, E2, U1, U2, V11, h12, V22, t11, t12, t21, t22, Vbb):
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
    U1, U2
        energy of the perturbators
    t11, t12
        coupling between the unperturbed states and the perturbator |beta 1>
    t21, t22
        coupling between the unperturbed states and the perturbator |beta 2>
    Vbb
        coupling between the perturbators
    """

    # construction of the Hückel Hamiltonian operating on the model space
    cell11 = E1 + V11
    cell22 = E2 + V22
    cell12 = h12

    restricted_huckel_hamiltonian = np.array([[cell11, cell12], [cell12, cell22]])

    # obtention of the eigenvalues and eigenvectors respectively (eigh means that we impose the
    # argument to be hermitian)
    model_energy, huckel_ket = sci.linalg.eigh(restricted_huckel_hamiltonian)

    # construction of the Hückel Hamiltonian operating on the total space
    extended_huckel_hamiltonian = np.zeros((4, 4))
    extended_huckel_hamiltonian[:2, :2] = restricted_huckel_hamiltonian
    extended_huckel_hamiltonian[2, 2] = U1
    extended_huckel_hamiltonian[3, 3] = U2

    huckel_energy = [model_energy[0], model_energy[1], U1, U2]

    return extended_huckel_hamiltonian, huckel_energy, huckel_ket


def BasisTransformation(matrix, transition_submatrix):
    """
    Obtention of the total Hamiltonian in the model kets basis

    Parameters
    ----------
    matrix
        matrix expressed in the unperturbed basis
    transition_submatrix
        transition matrix on the model space from the unperturbed basis to the hückel or model kets
        basis

    transition_matrix
        total transition matrix from the unperturbed basis to the hückel or model kets basis
    """

    transition_matrix = np.zeros((4, 4))
    transition_matrix[:2, :2] = transition_submatrix
    for index in range(2, len(matrix)):
        transition_matrix[index, index] = 1

    return np.matmul(
        np.matmul(np.transpose(transition_matrix), matrix), transition_matrix
    )


def BWPerturbationOrder2(
    perturbation_matrix, energy_list, E1, E2, t11, t12, t21, t22, nb_iterations=1
):
    """
    Computation of the energies by second-order BW procedure

    Parameters
    ----------
    perturbation_matrix
        the BW perturbation matrix, expressed in the eigenbasis of the BW zeroth-order Hamiltonian
    energy_list
        list of the eigenvalues of the BW zeroth-order Hamiltonian
    E1, E2
        energy of the unperturbed states
    t11, t12
        coupling between the unperturbed states and the perturbator |beta 1>
    t21, t22
        coupling between the unperturbed states and the perturbator |beta 2>
    nb_iterations
        number of iterations to optimize the energy
    """

    nb_model_states = 2
    nb_states = len(energy_list)
    perturbed_energy = []
    perturbation_contributions = []
    self_consistency_warning = []

    # computes the energy of each model state, one for each value of state_loop
    for state_loop in range(nb_model_states):
        # the variable whose value will be updated through each iteration of the BW self-consistent
        # procedure, and whose final value will be the correct BW energy
        sc_energy = energy_list[state_loop]
        perturbation_per_iteration = []

        # iterates the self-consistent procedure to solve the BW perturbation problem
        for _ in range(nb_iterations):
            # second-order correction
            second_order_correction = 0
            for coupling_loop in range(nb_states):
                if (
                    state_loop != coupling_loop
                ):  # and (coupling_loop < nb_model_states or t11 != t12 or t21 != t22 or E1 != E2):
                    diff = sc_energy - energy_list[coupling_loop]
                    matrix_element = perturbation_matrix[state_loop, coupling_loop]
                    if abs(diff / matrix_element) > 0.15:
                        second_order_correction += matrix_element**2 / diff
                    else:
                        self_consistency_warning.append(
                            f"Warning coupling {state_loop+1}-{coupling_loop+1}: low ratio energy "
                            f"gap to matrix element: {diff/matrix_element}. Coupling was ignored "
                            f"in this iteration"
                        )

                    # print(f"({state_loop+1}{coupling_loop+1}) = {matrix_element**2/diff}")

            # self-consistent energy
            sc_energy = (
                energy_list[state_loop]
                + perturbation_matrix[state_loop, state_loop]
                + second_order_correction
            )

            perturbation_per_iteration.append(
                [
                    sc_energy,
                    energy_list[state_loop]
                    + perturbation_matrix[state_loop, state_loop],
                    second_order_correction,
                ]
            )

        perturbed_energy.append(sc_energy)
        perturbation_contributions.append(perturbation_per_iteration)

    return perturbed_energy, perturbation_contributions, self_consistency_warning


def BWPerturbationOrder3(
    perturbation_matrix, energy_list, E1, E2, t11, t12, t21, t22, nb_iterations=1
):
    """
    Computation of the energies by third-order BW procedure

    Parameters
    ----------
    perturbation_matrix
        the BW perturbation matrix, expressed in the eigenbasis of the BW zeroth-order Hamiltonian
    energy_list
        list of the eigenvalues of the BW zeroth-order Hamiltonian
    E1, E2
        energy of the unperturbed states
    t11, t12
        coupling between the unperturbed states and the perturbator |beta 1>
    t21, t22
        coupling between the unperturbed states and the perturbator |beta 2>
    nb_iterations
        number of iterations to optimize the energy
    """

    nb_model_states = 2
    nb_states = len(energy_list)
    perturbed_energy = []
    perturbation_contributions = []
    self_consistency_warning = []

    # computes the energy of each model state, one for each value of state_loop
    for state_loop in range(nb_model_states):

        # the variable whose value will be updated through each iteration of the BW self-consistent
        # procedure, and whose final value will be the correct BW energy
        sc_energy = energy_list[state_loop]
        perturbation_per_iteration = []

        # iterates the self-consistent procedure to solve the BW perturbation problem
        for _ in range(nb_iterations):
            # second-order correction
            second_order_correction = 0
            for coupling_loop in range(nb_states):
                if (
                    state_loop != coupling_loop
                ):  # and (coupling_loop < nb_model_states or t11 != t12 or t21 != t22 or E1 != E2):
                    diff = sc_energy - energy_list[coupling_loop]
                    matrix_element = perturbation_matrix[state_loop][coupling_loop]
                    if abs(diff / matrix_element) > 0.15:
                        second_order_correction += matrix_element**2 / diff
                    else:
                        self_consistency_warning.append(
                            f"Warning coupling {state_loop+1}-{coupling_loop+1}: low ratio energy "
                            f"gap to matrix element: {diff/matrix_element}. Coupling was ignored "
                            f"in this iteration"
                        )

                    # print(f"({state_loop+1}{coupling_loop+1}) = {matrix_element**2/diff}")

            # third-order correction
            third_order_correction = 0
            for coupling_loop in range(nb_states):
                for mixing_loop in range(nb_states):
                    if coupling_loop != state_loop and mixing_loop != state_loop:
                        # definition of useful matrix elements
                        mel1 = perturbation_matrix[state_loop][coupling_loop]
                        mel2 = perturbation_matrix[coupling_loop][mixing_loop]
                        mel3 = perturbation_matrix[mixing_loop][state_loop]
                        # numerator and denominator
                        mel_product = mel1 * mel2 * mel3
                        diff_product = (sc_energy - energy_list[mixing_loop]) * (
                            sc_energy - energy_list[coupling_loop]
                        )
                        # correction
                        if (
                            abs(diff_product / (mel1 * mel2)) > 0.15
                            and abs(diff_product / (mel2 * mel3)) > 0.15
                        ):
                            third_order_correction += mel_product / diff_product
                        else:
                            self_consistency_warning.append(
                                f"Warning state {state_loop+1}: low ratio energy gap to matrix "
                                f"element: {diff_product/(mel1*mel2)} and/or "
                                f"{diff_product/(mel2*mel3)}. Coupling was ignored in this "
                                f"iteration"
                            )

                        # print(f"<{state_loop+1}|W|{coupling_loop+1}> = {mel1}")
                        # print(f"<{coupling_loop+1}|W|{mixing_loop+1}> = {mel2}")
                        # print(f"<{mixing_loop+1}|W|{state_loop+1}> = {mel3}")
                        # print(
                        #     f"({state_loop+1}{coupling_loop+1}{mixing_loop+1}) "\
                        #     f"= {mel_product/diff_product}\n"
                        # )

            # self-consistent energy
            sc_energy = (
                energy_list[state_loop]
                + perturbation_matrix[state_loop, state_loop]
                + second_order_correction
                + third_order_correction
            )

            perturbation_per_iteration.append(
                [
                    sc_energy,
                    energy_list[state_loop]
                    + perturbation_matrix[state_loop, state_loop],
                    second_order_correction,
                    third_order_correction,
                ]
            )

        perturbed_energy.append(sc_energy)
        perturbation_contributions.append(perturbation_per_iteration)

    return perturbed_energy, perturbation_contributions, self_consistency_warning


def CheckPerturbation(exact_hamiltonian, perturbation_matrix):
    """
    Obtention of the rho parameters

    Parameters
    ----------
    exact_hamiltonian
        total hamiltonian operating on the total space, expressed in the unperturbed basis
    perturbation_matrix
        the BW perturbation matrix, expressed in the unperturbed basis
    """

    zero_order_hamiltonian = exact_hamiltonian - perturbation_matrix
    rho = np.linalg.norm(
        exact_hamiltonian - zero_order_hamiltonian, ord=None
    ) / np.linalg.norm(zero_order_hamiltonian, ord=None)

    warning = []

    if rho > 1:
        warning.append(f"Error: perturbation bigger than zeroth order. rho = {rho}")

    elif rho > 0.9:
        warning.append(f"Warning: perturbation seems big. rho = {rho}")

    return rho, warning


def Results(E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations=1):
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
    U1, U2
        energy of the perturbators
    t11, t12
        coupling between the unperturbed states and the perturbator |beta 1>
    t21, t22
        coupling between the unperturbed states and the perturbator |beta 2>
    Vbb
        coupling between the perturbators
    nb_iterations
        number of iterations to optimize the energy
    """

    ### Diagonalization of the Exact Hamiltonian ###
    exact_hamiltonian, exact_energy, exact_state, rho_RS = ExactHamiltonian(
        E1, E2, U1, U2, V11, h12, V22, t11, t12, t21, t22, Vbb
    )

    ### Diagonalization of the Hückel Hamiltonian ###
    huckel_hamiltonian, huckel_energy, huckel_ket = HuckelHamiltonian(
        E1, E2, U1, U2, V11, h12, V22, t11, t12, t21, t22, Vbb
    )
    HBW_perturbation_unperturbed_basis = exact_hamiltonian - huckel_hamiltonian

    ### Diagonalization of the Effective Hamiltonian ###
    RS_hamiltonian, RS_energy, model_ket = EffectiveHamiltonian(
        E1, E2, U1, U2, V11, h12, V22, t11, t12, t21, t22, Vbb
    )
    RSBW_perturbation_unperturbed_basis = exact_hamiltonian - RS_hamiltonian

    ### HBW perturbation in the Hückel basis ###
    HBW_perturbation = BasisTransformation(
        HBW_perturbation_unperturbed_basis, huckel_ket
    )

    ### RSBW perturbation in the model kets basis ###
    RSBW_perturbation = BasisTransformation(
        RSBW_perturbation_unperturbed_basis, model_ket
    )

    ### Brillouin-Wigner Perturbation for Hückel Hamiltonian ###
    # print("HBW procedure")
    (
        HBW_perturbed_energy,
        HBW_perturbation_contributions,
        HBW_self_consistency_warning,
    ) = BWPerturbationOrder2(
        HBW_perturbation, huckel_energy, E1, E2, t11, t12, t21, t22, nb_iterations
    )

    ### 2nd order Brillouin-Wigner Perturbation for Effective Hamiltonian ###
    # print("RSBW procedure for 2nd order")
    (
        RSBW_perturbed_energy_order2,
        RSBW_perturbation_contributions_order2,
        RSBW_self_consistency_warning_order2,
    ) = BWPerturbationOrder2(
        RSBW_perturbation, RS_energy, E1, E2, t11, t12, t21, t22, nb_iterations
    )

    ### 3rd order Brillouin-Wigner Perturbation for Effective Hamiltonian ###
    # print("RSBW procedure for 3rd order")
    (
        RSBW_perturbed_energy_order3,
        RSBW_perturbation_contributions_order3,
        RSBW_self_consistency_warning_order3,
    ) = BWPerturbationOrder3(
        RSBW_perturbation, RS_energy, E1, E2, t11, t12, t21, t22, nb_iterations
    )

    ### Checks and procedure validity ###
    HBW_perturbation_status = CheckPerturbation(
        exact_hamiltonian, HBW_perturbation_unperturbed_basis
    )
    RSBW_perturbation_status = CheckPerturbation(
        exact_hamiltonian, RSBW_perturbation_unperturbed_basis
    )
    rho_list = [rho_RS, HBW_perturbation_status[0], RSBW_perturbation_status[0]]
    warning_list = (
        ["HBW procedure"]
        + HBW_perturbation_status[1]
        + HBW_self_consistency_warning
        + ["RSBW procedure"]
        + RSBW_perturbation_status[1]
        + RSBW_self_consistency_warning_order2
        + RSBW_self_consistency_warning_order3
    )

    perturbation_contributions = (
        HBW_perturbation_contributions,
        RSBW_perturbation_contributions_order2,
        RSBW_perturbation_contributions_order3,
    )

    return (
        exact_energy,
        RSBW_perturbed_energy_order2,
        RSBW_perturbed_energy_order3,
        HBW_perturbed_energy,
        RS_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    )


""" Plotting functions of h12 """


def PlotH12Energy(
    E1, E2, V11, V22, t11, t12, t21, t22, U1, U2, Vbb, h12_min, h12_max, nb_iterations
):

    xaxis = []
    # list containing the RSBW energy corrected to the BW second order for state 1 as a function of
    # h12
    second_order_RSBW_yaxis1 = []

    # list containing the RSBW energy corrected to the BW second order for state 2 as a function of
    # h12
    second_order_RSBW_yaxis2 = []

    # list containing the RSBW energy corrected to the BW third order for state 1 as a function of
    # h12
    third_order_RSBW_yaxis1 = []

    # list containing the RSBW energy corrected to the BW third order for state 2 as a function of
    # h12
    third_order_RSBW_yaxis2 = []
    HBW_yaxis1 = []  # list containing the HBW energy of state 1 as a function of h12
    HBW_yaxis2 = []  # list containing the HBW energy of state 2 as a function of h12
    RS_yaxis1 = []  # list containing the RS energy of state 1 as a function of h12
    RS_yaxis2 = []  # list containing the RS energy of state 2 as a function of h12

    # list containing the exact energy of state 1 as a function of h12
    exact_yaxis1 = []

    # list containing the exact energy of state 2 as a function of h12
    exact_yaxis2 = []

    # list containing the RSBW energy corrected to the BW third order for state 1 as a function of
    # h12, with Vbb = 0
    third_order_RSBW_yaxis1_ni = []

    # list containing the RSBW energy corrected to the BW third order for state 2 as a function of
    # h12, with Vbb = 0
    third_order_RSBW_yaxis2_ni = []

    step = (h12_max - h12_min) / 20

    for index in range(21):

        h12 = step * index + h12_min
        (
            exact_energy,
            RSBW_perturbed_energy_order2,
            RSBW_perturbed_energy_order3,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(
            E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
        )
        xaxis.append(h12)
        second_order_RSBW_yaxis1.append(RSBW_perturbed_energy_order2[0])
        second_order_RSBW_yaxis2.append(RSBW_perturbed_energy_order2[1])
        third_order_RSBW_yaxis1.append(RSBW_perturbed_energy_order3[0])
        third_order_RSBW_yaxis2.append(RSBW_perturbed_energy_order3[1])
        HBW_yaxis1.append(HBW_perturbed_energy[0])
        HBW_yaxis2.append(HBW_perturbed_energy[1])
        RS_yaxis1.append(RS_perturbed_energy[0])
        RS_yaxis2.append(RS_perturbed_energy[1])
        exact_yaxis1.append(exact_energy[0])
        exact_yaxis2.append(exact_energy[1])

        # same but with no interaction (ni) between the perturbers

        (
            exact_energy_ni,
            RSBW_perturbed_energy_order2_ni,
            RSBW_perturbed_energy_order3_ni,
            HBW_perturbed_energy_ni,
            RS_perturbed_energy_ni,
            perturbation_contributions_ni,
            rho_list_ni,
            warning_list_ni,
        ) = Results(E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, 0, nb_iterations)
        third_order_RSBW_yaxis1_ni.append(RSBW_perturbed_energy_order3_ni[0])
        third_order_RSBW_yaxis2_ni.append(RSBW_perturbed_energy_order3_ni[1])

        # print(f"h12 = {h12}, {HBW_perturbed_energy[1]}")
        # print(f"h12 = {h12}, {exact_energy[0]}")
        # print(f"h12 = {h12}, {exact_energy[1]}")

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

    """ First graph """

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, exact_yaxis1, "k-", label="Exact energy")
    fig1.plot(xaxis, RS_yaxis1, "g:", label="RS energy")
    fig1.plot(xaxis, HBW_yaxis1, "r--", label="BW energy")
    fig1.plot(xaxis, second_order_RSBW_yaxis1, "b-.", label="RSBW energy")

    fig1.set_xlabel("K", fontsize=23)
    fig1.set_ylabel("Energy of ground state", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, exact_yaxis2, "k-", label="Exact energy")
    fig2.plot(xaxis, RS_yaxis2, "g:", label="RS energy")
    fig2.plot(xaxis, HBW_yaxis2, "r--", label="BW energy")
    fig2.plot(xaxis, second_order_RSBW_yaxis2, "b-.", label="RSBW energy")

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel("Energy of excited state", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()

    """ Comparison of correction orders - First graph """

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, exact_yaxis1, "k-", label="Exact energy")
    fig1.plot(xaxis, second_order_RSBW_yaxis1, "b:", label="2nd order")
    fig1.plot(
        xaxis,
        third_order_RSBW_yaxis1_ni,
        "c--",
        label="3rd order, $V_{\\beta \\beta'} = 0$",
    )
    fig1.plot(
        xaxis,
        third_order_RSBW_yaxis1,
        "m-.",
        label="3rd order, $V_{\\beta \\beta'} =$" + f" {Vbb}",
    )

    fig1.set_xlabel("K", fontsize=23)
    fig1.set_ylabel("Energy of ground state", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()
    # plt.savefig('Amr_RSBW_Vincent_Urgent/energy_ground_3rd_order_Vbb0.25.pdf')

    """ Comparison of correction orders - Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, exact_yaxis2, "k-", label="Exact energy")
    fig2.plot(xaxis, second_order_RSBW_yaxis2, "b:", label="2nd order")
    fig2.plot(
        xaxis,
        third_order_RSBW_yaxis2_ni,
        "c--",
        label="3rd order, $V_{\\beta \\beta'} = 0$",
    )
    fig2.plot(
        xaxis,
        third_order_RSBW_yaxis2,
        "m-.",
        label="3rd order, $V_{\\beta \\beta'} =$" + f" {Vbb}",
    )

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel("Energy of excited state", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()

    plt.show()
    # plt.savefig('Amr_RSBW_Vincent_Urgent/energy_excited_3rd_order_Vbb0.25.pdf')


def PlotH12RelGap(
    E1, E2, V11, V22, t11, t12, t21, t22, U1, U2, Vbb, h12_min, h12_max, nb_iterations
):

    xaxis = []
    # list containing the relative gap between RSBW and exact energy of state 1, as a function of
    # h12
    relative_gap_RSBW_1 = []

    # list containing the relative gap between second-order corrected RSBW and exact energy of state
    # 1, as a function of h12
    relative_gap_second_order_1 = []

    # list containing the relative gap between second-order corrected RSBW and exact energy of state
    # 2, as a function of h12
    relative_gap_second_order_2 = []

    # list containing the relative gap between third-order corrected RSBW and exact energy of state
    # 1, as a function of h12
    relative_gap_third_order_1 = []

    # list containing the relative gap between third-order corrected RSBW and exact energy of state
    # 2, as a function of h12
    relative_gap_third_order_2 = []

    # list containing the relative gap between HBW and exact energy of state 1, as a function of h12
    relative_gap_HBW_1 = []

    # list containing the relative gap between HBW and exact energy of state 2, as a function of h12
    relative_gap_HBW_2 = []

    # list containing the relative gap between RS and exact energy of state 1, as a function of h12
    relative_gap_RS_1 = []

    # list containing the relative gap between RS and exact energy of state 2, as a function of h12
    relative_gap_RS_2 = []

    # list containing the relative gap between third-order corrected RSBW and exact energy of state
    # 1, as a function of h12, third-order correction is reduced to the interaction between
    # perturbers
    relative_gap_third_order_1_oi = []

    # list containing the relative gap between third-order corrected RSBW and exact energy of state
    # 2, as a function of h12, third-order correction is reduced to the interaction between
    # perturbers
    relative_gap_third_order_2_oi = []

    exact_energy_gap = []
    energy_gap_third_order_oi = []
    error_energy_gap_truncated_third_order = []

    step = (h12_max - h12_min) / 20

    for index in range(21):

        h12 = step * index + h12_min
        (
            exact_energy,
            RSBW_perturbed_energy_order2,
            RSBW_perturbed_energy_order3,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(
            E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
        )
        xaxis.append(h12)
        relative_gap_second_order_1.append(
            abs((RSBW_perturbed_energy_order2[0] - exact_energy[0]) / exact_energy[0])
        )
        relative_gap_second_order_2.append(
            abs((RSBW_perturbed_energy_order2[1] - exact_energy[1]) / exact_energy[1])
        )
        relative_gap_third_order_1.append(
            abs((RSBW_perturbed_energy_order3[0] - exact_energy[0]) / exact_energy[0])
        )
        relative_gap_third_order_2.append(
            abs((RSBW_perturbed_energy_order3[1] - exact_energy[1]) / exact_energy[1])
        )
        relative_gap_HBW_1.append(
            abs((HBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        relative_gap_HBW_2.append(
            abs((HBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )
        relative_gap_RS_1.append(
            abs((RS_perturbed_energy[0] - exact_energy[0]) / exact_energy[0])
        )
        relative_gap_RS_2.append(
            abs((RS_perturbed_energy[1] - exact_energy[1]) / exact_energy[1])
        )

        if abs(exact_energy[0]) < 0.07:
            print(
                f"h12 = {h12} ERROR state 1: exact energy too low for the relative gap to be"
                f"significant\n"
            )
        if abs(exact_energy[1]) < 0.07:
            print(
                f"h12 = {h12} ERROR state 2: exact energy too low for the relative gap to be"
                f"significant\n"
            )

        # same but with no interaction (ni) between the perturbers // same with only the interaction
        # (oi) between perturbers

        (
            exact_energy_ni,
            RSBW_perturbed_energy_order2_ni,
            RSBW_perturbed_energy_order3_ni,
            HBW_perturbed_energy_ni,
            RS_perturbed_energy_ni,
            perturbation_contributions_ni,
            rho_list_ni,
            warning_list_ni,
        ) = Results(E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, 0, nb_iterations)

        RSBW_perturbed_energy_order3_oi_state1 = (
            RSBW_perturbed_energy_order2[0]
            + RSBW_perturbed_energy_order3[0]
            - RSBW_perturbed_energy_order3_ni[0]
        )
        RSBW_perturbed_energy_order3_oi_state2 = (
            RSBW_perturbed_energy_order2[1]
            + RSBW_perturbed_energy_order3[1]
            - RSBW_perturbed_energy_order3_ni[1]
        )
        relative_gap_third_order_1_oi.append(
            abs(
                (RSBW_perturbed_energy_order3_oi_state1 - exact_energy[0])
                / exact_energy[0]
            )
        )
        relative_gap_third_order_2_oi.append(
            abs(
                (RSBW_perturbed_energy_order3_oi_state2 - exact_energy[1])
                / exact_energy[1]
            )
        )
        exact_energy_gap.append(exact_energy[1] - exact_energy[0])
        energy_gap_third_order_oi.append(
            RSBW_perturbed_energy_order3_oi_state2
            - RSBW_perturbed_energy_order3_oi_state1
        )
        error_energy_gap_truncated_third_order.append(
            abs(
                (energy_gap_third_order_oi[-1] - exact_energy_gap[-1])
                / exact_energy_gap[-1]
            )
        )

        # print(f"h12 = {h12} perturbation: {perturbation_contributions[1][1]}")
        # [0] for HBW, [1] for RSBW, [n][0] for ground state, [n][1] for excited state

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"h12 = {h12}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    print("Relative gap RSBW state 1, Relative gap RSBW state 2")
    for index in range(21):
        print(
            f"h12 = {step*index + h12_min} {relative_gap_second_order_1[index]}, "
            f"{relative_gap_third_order_1[index]}"
        )

    # np.save(
    #     'data10.npy',
    #     [
    #         xaxis,
    #         relative_gap_second_order_1,
    #         relative_gap_third_order_1_oi,
    #         relative_gap_third_order_1
    #     ]
    # )

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

    fig1.plot(xaxis, relative_gap_RS_1, "g:", label="RS procedure")
    fig1.plot(xaxis, relative_gap_HBW_1, "r--", label="BW procedure")
    fig1.plot(xaxis, relative_gap_second_order_1, "b-.", label="RSBW procedure")

    fig1.set_xlabel("K", fontsize=23)
    fig1.set_ylabel(
        "Relative gap between approximate\nand exact energy of ground state",
        fontsize=23,
    )
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, relative_gap_RS_2, "g:", label="RS procedure")
    fig2.plot(xaxis, relative_gap_HBW_2, "r--", label="BW procedure")
    fig2.plot(xaxis, relative_gap_second_order_2, "b-.", label="RSBW procedure")

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel(
        "Relative gap between approximate\nand exact energy of excited state",
        fontsize=23,
    )
    fig2.legend(loc="best")

    plt.tight_layout()

    """ Comparison of correction orders - First graph """

    fig, (fig1) = plt.subplots(figsize=(8, 6))

    fig1.plot(xaxis, relative_gap_second_order_1, "b:", label="2nd order")
    fig1.plot(xaxis, relative_gap_third_order_1_oi, "c--", label="Truncated 3rd order")
    fig1.plot(xaxis, relative_gap_third_order_1, "m-.", label="Full 3rd order")

    fig1.set_xlabel("K", fontsize=23)
    fig1.set_ylabel("Relative error on the\niter-RSBW approached energy", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()
    # plt.savefig('fig4.pdf')

    """ Comparison of correction orders - Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, relative_gap_second_order_2, "b:", label="2nd order")
    fig2.plot(xaxis, relative_gap_third_order_2_oi, "c--", label="Truncated 3rd order")
    fig2.plot(xaxis, relative_gap_third_order_2, "m-.", label="Full 3rd order")

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel("Relative error on the\niter-RSBW approached energy", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()
    # plt.savefig('Amr_RSBW_Vincent_Urgent/relative_excited_3rd_order_Vbb0.25.pdf')

    plt.show()

    """ Energy gap """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, exact_energy_gap, "k-", label="exact energy gap")
    fig2.plot(xaxis, energy_gap_third_order_oi, "r.-", label="RSBW energy gap")

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel("Energy gap", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()

    plt.show()

    """ Energy gap (relative error) """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, error_energy_gap_truncated_third_order, "k-")

    fig2.set_xlabel("K", fontsize=23)
    fig2.set_ylabel("Relative error on the\niter-RSBW energy gap", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()

    plt.show()


def PlotH12Rho(
    E1, E2, V11, V22, t11, t12, t21, t22, U1, U2, Vbb, h12_min, h12_max, nb_iterations
):

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
            RSBW_perturbed_energy_order3,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(
            E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
        )
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
    V11 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V11")
    V22 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V22")
    t11 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t11")
    t12 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t12")
    t21 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t21")
    t22 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t22")
    U1 = widgets.FloatSlider(min=0, max=50, value=2, step=0.1, description="U1")
    U2 = widgets.FloatSlider(min=0, max=1000, value=3, step=0.1, description="U2")
    Vbb = widgets.FloatSlider(min=-3, max=3, value=-0.3, step=0.1, description="Vbb")
    # Number of iterations for BW self-consistent procedure
    nb_iterations = widgets.IntSlider(
        min=1, max=10, value=1, step=1, description="iterations"
    )
    # Changes the information displayed
    h12_min = widgets.FloatSlider(
        min=-5, max=-0.5, value=-1.5, step=0.1, description="h12_min"
    )
    h12_max = widgets.FloatSlider(
        min=-3, max=3, value=-0.5, step=0.1, description="h12_max"
    )

    energy = widgets.HBox([E1, E2, U1, U2])
    vinterne = widgets.HBox([V11, V22, Vbb])
    couplage = widgets.HBox([t11, t12, t21, t22])
    optimisation = widgets.HBox([nb_iterations, h12_min, h12_max])

    if displayed_info == 0:
        out = widgets.interactive_output(
            PlotH12Energy,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "Vbb": Vbb,
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
                "V11": V11,
                "V22": V22,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "Vbb": Vbb,
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
                "V11": V11,
                "V22": V22,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "Vbb": Vbb,
                "h12_min": h12_min,
                "h12_max": h12_max,
                "nb_iterations": nb_iterations,
            },
        )

    display(energy, vinterne, couplage, optimisation, out)


""" Plotting functions of Vbb """


def PlotVbbEnergy(
    E1, E2, V11, V22, t11, t12, t21, t22, U1, U2, h12, Vbb_min, Vbb_max, nb_iterations
):

    xaxis = []
    # list containing the RSBW energy corrected to the BW second order for state 1 as a function of
    # Vbb
    second_order_yaxis1 = []

    # list containing the RSBW energy corrected to the BW second order for state 2 as a function of
    # Vbb
    second_order_yaxis2 = []

    # list containing the RSBW energy corrected to the BW third order for state 1 as a function of
    # Vbb
    third_order_yaxis1 = []

    # list containing the RSBW energy corrected to the BW third order for state 2 as a function of
    # Vbb
    third_order_yaxis2 = []

    # list containing the exact energy of state 1 as a function of Vbb
    exact_yaxis1 = []

    # list containing the exact energy of state 2 as a function of Vbb
    exact_yaxis2 = []

    step = (Vbb_max - Vbb_min) / 20

    for index in range(21):

        Vbb = step * index + Vbb_min
        (
            exact_energy,
            RSBW_perturbed_energy_order2,
            RSBW_perturbed_energy_order3,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(
            E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
        )
        xaxis.append(Vbb)
        second_order_yaxis1.append(RSBW_perturbed_energy_order2[0])
        second_order_yaxis2.append(RSBW_perturbed_energy_order2[1])
        third_order_yaxis1.append(RSBW_perturbed_energy_order3[0])
        third_order_yaxis2.append(RSBW_perturbed_energy_order3[1])
        exact_yaxis1.append(exact_energy[0])
        exact_yaxis2.append(exact_energy[1])

        # print(f"Vbb = {Vbb}, {huckelBW_perturbed_energy[1]}")
        # print(f"Vbb = {Vbb}, {exact_energy[0]}")
        # print(f"Vbb = {Vbb}, {exact_energy[1]}")

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"Vbb = {Vbb}")
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
    fig1.plot(xaxis, second_order_yaxis1, "g:", label="2nd order corrected energy")
    fig1.plot(xaxis, third_order_yaxis1, "r--", label="3rd order corrected energy")

    fig1.set_xlabel("$V_{\\beta \\beta'}$", fontsize=23)
    fig1.set_ylabel("Energy of ground state", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, exact_yaxis2, "k-", label="Exact energy")
    fig2.plot(xaxis, second_order_yaxis2, "g:", label="2nd order corrected energy")
    fig2.plot(xaxis, third_order_yaxis2, "r--", label="3rd order corrected energy")

    fig2.set_xlabel("$V_{\\beta \\beta'}$", fontsize=23)
    fig2.set_ylabel("Energy of excited state", fontsize=23)
    fig2.legend(loc="best")

    plt.tight_layout()

    plt.show()


def PlotVbbRelGap(
    E1, E2, V11, V22, t11, t12, t21, t22, U1, U2, h12, Vbb_min, Vbb_max, nb_iterations
):

    xaxis = []
    # list containing the relative gap between RSBW and exact energy of state 1, as a function of
    # Vbb
    relative_gap_second_order_1 = []

    # list containing the relative gap between RSBW and exact energy of state 2, as a function of
    # Vbb
    relative_gap_second_order_2 = []

    # list containing the relative gap between HBW and exact energy of state 1, as a function of Vbb
    relative_gap_third_order_1 = []

    # list containing the relative gap between HBW and exact energy of state 2, as a function of Vbb
    relative_gap_third_order_2 = []

    step = (Vbb_max - Vbb_min) / 20

    for index in range(21):

        Vbb = step * index + Vbb_min
        (
            exact_energy,
            RSBW_perturbed_energy_order2,
            RSBW_perturbed_energy_order3,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(
            E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
        )
        xaxis.append(Vbb)
        relative_gap_second_order_1.append(
            abs((RSBW_perturbed_energy_order2[0] - exact_energy[0]) / exact_energy[0])
        )
        relative_gap_second_order_2.append(
            abs((RSBW_perturbed_energy_order2[1] - exact_energy[1]) / exact_energy[1])
        )
        relative_gap_third_order_1.append(
            abs((RSBW_perturbed_energy_order3[0] - exact_energy[0]) / exact_energy[0])
        )
        relative_gap_third_order_2.append(
            abs((RSBW_perturbed_energy_order3[1] - exact_energy[1]) / exact_energy[1])
        )

        if abs(exact_energy[0]) < 0.07:
            print(
                f"Vbb = {Vbb} ERROR state 1: exact energy too low for the relative gap to be "
                f"significant\n"
            )
        if abs(exact_energy[1]) < 0.07:
            print(
                f"Vbb = {Vbb} ERROR state 2: exact energy too low for the relative gap to be "
                f"significant\n"
            )

        print(
            f"Vbb = {Vbb} perturbation: {perturbation_contributions[2][0]}"
        )  # [0] for HBW, [1] for RSBW, [n][0] for ground state, [n][1] for excited state

        if warning_list != ["HBW procedure", "RSBW procedure"]:
            print(f"Vbb = {Vbb}")
            for warning in warning_list:
                print(f" {warning}")
            print("\n")

    print("Relative gap RSBW state 1, Relative gap RSBW state 2")
    for index in range(21):
        print(
            f"Vbb = {step*index + Vbb_min} {relative_gap_second_order_1[index]}, "
            f"{relative_gap_third_order_1[index]}"
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

    fig1.plot(xaxis, relative_gap_second_order_1, "g:", label="2nd order RSBW")
    fig1.plot(xaxis, relative_gap_third_order_1, "r--", label="3rd order RSBW")

    fig1.set_xlabel("$V_{\\beta \\beta'}$", fontsize=23)
    fig1.set_ylabel(
        "Relative gap between approximate\nand exact energy of ground state",
        fontsize=23,
    )
    fig1.legend(loc="best")

    plt.tight_layout()

    """ Second graph """

    fig, (fig2) = plt.subplots(figsize=(8, 6))

    fig2.plot(xaxis, relative_gap_second_order_2, "g:", label="2nd order RSBW")
    fig2.plot(xaxis, relative_gap_third_order_2, "r--", label="3rd order RSBW")

    fig2.set_xlabel("$V_{\\beta \\beta'}$", fontsize=23)
    fig2.set_ylabel(
        "Relative gap between approximate\nand exact energy of excited state",
        fontsize=23,
    )
    fig2.legend(loc="best")

    plt.tight_layout()
    plt.show()


def PlotVbbRho(
    E1, E2, V11, V22, h12, t11, t12, t21, t22, U1, U2, Vbb_min, Vbb_max, nb_iterations
):

    xaxis = []
    rho_RS = []  # list containing the RS rho parameter, as a function of Vbb
    rho_HBW = []  # list containing the HBW rho parameter, as a function of Vbb
    rho_RSBW = []  # list containing the RSBW rho parameter, as a function of Vbb

    step = (Vbb_max - Vbb_min) / 20

    for index in range(21):

        Vbb = step * index + Vbb_min
        (
            exact_energy,
            RSBW_perturbed_energy,
            RSBW_perturbed_energy_order3,
            HBW_perturbed_energy,
            RS_perturbed_energy,
            perturbation_contributions,
            rho_list,
            warning_list,
        ) = Results(
            E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
        )
        xaxis.append(Vbb)
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

    fig1.set_xlabel("$V_{\\beta \\beta'}$", fontsize=23)
    fig1.set_ylabel("$\\rho$ parameter", fontsize=23)
    fig1.legend(loc="best")

    plt.tight_layout()
    plt.show()


def DisplayFunctionVbb(displayed_info):

    E1 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="E1")
    E2 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="E2")
    V11 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V11")
    V22 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V22")
    t11 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t11")
    t12 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t12")
    t21 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t21")
    t22 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t22")
    U1 = widgets.FloatSlider(min=0, max=10, value=3, step=0.1, description="U1")
    U2 = widgets.FloatSlider(min=0, max=10, value=3, step=0.1, description="U2")
    h12 = widgets.FloatSlider(min=-3, max=0, value=-1, step=0.1, description="h12")
    # Number of iterations for BW self-consistent procedure
    nb_iterations = widgets.IntSlider(
        min=1, max=10, value=1, step=1, description="iterations"
    )
    # Changes the information displayed
    Vbb_min = widgets.FloatSlider(
        min=-5, max=-0.5, value=-2, step=0.1, description="Vbb_min"
    )
    Vbb_max = widgets.FloatSlider(
        min=-3, max=0, value=-0.05, step=0.1, description="Vbb_max"
    )

    energy = widgets.HBox([E1, E2, U1, U2])
    vinterne = widgets.HBox([V11, V22, h12])
    couplage = widgets.HBox([t11, t12, t21, t22])
    optimisation = widgets.HBox([nb_iterations, Vbb_min, Vbb_max])

    if displayed_info == 0:
        out = widgets.interactive_output(
            PlotVbbEnergy,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "h12": h12,
                "Vbb_min": Vbb_min,
                "Vbb_max": Vbb_max,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 1:
        out = widgets.interactive_output(
            PlotVbbRelGap,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "h12": h12,
                "Vbb_min": Vbb_min,
                "Vbb_max": Vbb_max,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 2:
        out = widgets.interactive_output(
            PlotVbbRho,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "h12": h12,
                "Vbb_min": Vbb_min,
                "Vbb_max": Vbb_max,
                "nb_iterations": nb_iterations,
            },
        )

    display(energy, vinterne, couplage, optimisation, out)


""" Display of miscellaneous values """


def PrintValuesEnergy(
    E1, E2, V11, V22, h12, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
):
    (
        exact_energy,
        RSBW_perturbed_energy_order2,
        RSBW_perturbed_energy_order3,
        HBW_perturbed_energy,
        RS_perturbed_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    ) = Results(E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations)
    print(f"Exact energy: {exact_energy}")
    print(f"RS energy: {RS_perturbed_energy}")
    print(f"HBW energy: {HBW_perturbed_energy}")
    print(f"RSBW second-order corrected energy: {RSBW_perturbed_energy_order2}")
    print(f"RSBW third-order corrected energy: {RSBW_perturbed_energy_order3}")


def PrintValuesRelGap(
    E1, E2, V11, V22, h12, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
):
    (
        exact_energy,
        RSBW_perturbed_energy,
        RSBW_perturbed_energy_order3,
        HBW_perturbed_energy,
        RS_perturbed_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    ) = Results(E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations)

    to_print = [
        abs((RS_perturbed_energy[0] - exact_energy[0]) / exact_energy[0]),
        abs((RS_perturbed_energy[1] - exact_energy[1]) / exact_energy[1]),
    ]
    print(f"RS relative gap: {to_print}")

    to_print = [
        abs((HBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0]),
        abs((HBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1]),
    ]
    print(f"HBW relative gap: {to_print}")

    to_print = [
        abs((RSBW_perturbed_energy[0] - exact_energy[0]) / exact_energy[0]),
        abs((RSBW_perturbed_energy[1] - exact_energy[1]) / exact_energy[1]),
    ]
    print(f"RSBW relative gap: {to_print}")


def PrintValuesRho(
    E1, E2, V11, V22, h12, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations
):
    (
        exact_energy,
        RSBW_perturbed_energy,
        RSBW_perturbed_energy_order3,
        HBW_perturbed_energy,
        RS_perturbed_energy,
        perturbation_contributions,
        rho_list,
        warning_list,
    ) = Results(E1, E2, V11, h12, V22, t11, t12, t21, t22, U1, U2, Vbb, nb_iterations)
    print(f"rho RS: {rho_list[0]}")
    print(f"rho HBW: {rho_list[1]}")
    print(f"rho RSBW: {rho_list[2]}")
    print(f"rho RSBW / rho RS: {rho_list[2]/rho_list[0]}")
    print(f"rho RSBW / rho HBW: {rho_list[2]/rho_list[1]}")


def DisplayValues(displayed_info):

    E1 = widgets.FloatSlider(min=0, max=5, value=0, step=0.1, description="E1")
    E2 = widgets.FloatSlider(min=0, max=5, value=0, step=0.1, description="E2")
    V11 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V11")
    V22 = widgets.FloatSlider(min=-3, max=3, value=0, step=0.1, description="V22")
    h12 = widgets.FloatSlider(min=-4, max=0, value=-1, step=0.05, description="h12")
    t11 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t11")
    t12 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t12")
    t21 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t21")
    t22 = widgets.FloatSlider(min=-3, max=3, value=-1, step=0.1, description="t22")
    U1 = widgets.FloatSlider(min=0, max=50, value=3, step=0.1, description="U1")
    U2 = widgets.FloatSlider(min=0, max=50, value=3, step=0.1, description="U2")
    Vbb = widgets.FloatSlider(min=-3, max=3, value=-2, step=0.1, description="Vbb")
    # Number of iterations for BW self-consistent procedure
    nb_iterations = widgets.IntSlider(
        min=1, max=10, value=1, step=1, description="iterations"
    )

    energy = widgets.HBox([E1, E2, U1, U2])
    vinterne = widgets.HBox([V11, V22, Vbb])
    couplage = widgets.HBox([h12, t11, t12, t21, t22])
    optimisation = widgets.HBox([nb_iterations])

    if displayed_info == 0:
        out = widgets.interactive_output(
            PrintValuesEnergy,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "Vbb": Vbb,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 1:
        out = widgets.interactive_output(
            PrintValuesRelGap,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "Vbb": Vbb,
                "nb_iterations": nb_iterations,
            },
        )
    elif displayed_info == 2:
        out = widgets.interactive_output(
            PrintValuesRho,
            {
                "E1": E1,
                "E2": E2,
                "V11": V11,
                "V22": V22,
                "h12": h12,
                "t11": t11,
                "t12": t12,
                "t21": t21,
                "t22": t22,
                "U1": U1,
                "U2": U2,
                "Vbb": Vbb,
                "nb_iterations": nb_iterations,
            },
        )

    display(energy, vinterne, couplage, optimisation, out)


""" Displays the results """


def DisplayChoice(display_mode, displayed_info):
    if display_mode == 0:
        DisplayValues(displayed_info)
    elif display_mode == 1:
        DisplayFunctionH12(displayed_info)
    elif display_mode == 2:
        DisplayFunctionVbb(displayed_info)


print("display mode: Values/Functions of h12/Functions of Vbb")
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
