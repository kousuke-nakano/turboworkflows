#!/usr/bin/env python
# coding: utf-8

# # pySCF -> TREX-IO (Water molecule)

#Logger
from logging import config, getLogger, StreamHandler, Formatter
logger = getLogger('Turbo-Genius').getChild(__name__)

# load python packages
import numpy as np
import os, sys
import pandas as pd
import scipy
import numpy
import itertools

# load ASE modules
from ase.io import read
from ase.units import Bohr
# load pyscf packages
from pyscf import gto, scf, mp, tools
from pyscf.pbc import gto as gto_pbc
from pyscf.pbc import dft as pbcdft
from pyscf.pbc import scf as pbcscf
from pyscf.scf.chkfile import dump_scf

def pyscf_to_trexio(pyscf_checkfile="pyscf.chk", trexio_filename="trexio.hdf5", twist_average_in=False):
    # ## pySCF -> TREX-IO
    # - how to install trexio
    # - pip install trexio

    # import trexio
    import trexio

    logger.info(f"pyscf_checkfile = {pyscf_checkfile}")
    logger.info(f"trexio_filename = {trexio_filename}")
    logger.info(f"Conversion starts...")

    # pyscf instances
    mol = scf.chkfile.load_mol(pyscf_checkfile)
    mf = scf.chkfile.load(pyscf_checkfile, "scf")

    # PBC info
    try:
        mol.a
        pbc_flag = True
    except AttributeError:
        pbc_flag = False
    logger.info(f"PBC flag = {pbc_flag}")

    # twist_average info
    if pbc_flag:
        try:
            k = mf['kpt']
            twist_average = False
            logger.info("Single-k calculation")
            k_list = [k]
            logger.info(k_list)
        except KeyError:
            twist_average = True
            # logger.error("Twisted-average = True is not implemented")
            # raise NotImplementedError
            logger.info("Twisted-average calculation")
            logger.info("Separated TREXIO files are generated")
            logger.info("The Correspondence between the index and k is written in kp_info.dat")
            with open("kp_info.dat", "w") as f:
                f.write("# k_index, kx, ky, kz\n")
            k_list = mf['kpts']
        finally:
            mol = pbcscf.chkfile.load_cell(pyscf_checkfile)
            k_list = mol.get_scaled_kpts(k_list)
            logger.info(k_list)

    else:
        twist_average = False
        k_list = [[0.0 ,0.0, 0.0]]

    assert twist_average_in == twist_average

    # if pbc_flag == true, check if ecp or pseudo
    if pbc_flag:
        if len(mol._pseudo) > 0:
            logger.error("TREXIO does not support 'pseudo' format for PBC. Plz. use 'ecp'")
            raise NotImplementedError

    if twist_average:
        logger.warning(f"WF at each k point is saved as a separated file, kXXXX_{trexio_filename}")
        logger.warning(f"k points info. is stored in kp_info.dat.")

    # each k WF is stored as a separate file!!
    # for an open-boundary calculation, and a single-k one,
    # k_index is a dummy variable
    for k_index, k_vec in enumerate(k_list):
        assert len(k_vec) == 3 # 3d variable
        # set a filename
        if twist_average:
            logger.info(f"kpt={k_vec}")
            filename = os.path.join(os.path.dirname(trexio_filename), f"k{k_index}_" + os.path.basename(trexio_filename))
            logger.info(f"filename={filename}")
            with open("kp_info.dat", "a") as f:
                f.write(f"{k_index} {k_vec[0]} {k_vec[1]} {k_vec[2]}\n")
        else:
            filename = trexio_filename

        if os.path.exists(filename): os.remove(filename)

        # trexio file
        trexio_file = trexio.File(filename, mode='w', back_end=trexio.TREXIO_HDF5)

        ##########################################
        # PBC info
        ##########################################
        if pbc_flag:
            a = np.array(mol.a[0]) / Bohr # angstrom -> bohr
            b = np.array(mol.a[1]) / Bohr # angstrom -> bohr
            c = np.array(mol.a[2]) / Bohr # angstrom -> bohr
            k_point = k_vec
            periodic = True
        else:
            periodic = False

        # pbc and cell info
        trexio.write_pbc_periodic(trexio_file, periodic)
        if pbc_flag:
            trexio.write_cell_a(trexio_file, a)
            trexio.write_cell_b(trexio_file, b)
            trexio.write_cell_c(trexio_file, c)
            trexio.write_pbc_k_point(trexio_file, k_point)

        # structure info.
        electron_up_num, electron_dn_num=mol.nelec
        nucleus_num=mol.natm
        atom_charges_list=[mol.atom_charge(i) for i in range(mol.natm)]
        atom_nelec_core_list=[mol.atom_nelec_core(i) for i in range(mol.natm)]
        atomic_number_list=[mol.atom_charge(i) + mol.atom_nelec_core(i) for i in range(mol.natm)]
        chemical_symbol_list=[mol.atom_pure_symbol(i) for i in range(mol.natm)]
        coords_np=mol.atom_coords(unit='Bohr')

        ##########################################
        # Structure info
        ##########################################
        trexio.write_electron_up_num(trexio_file, electron_up_num)
        trexio.write_electron_dn_num(trexio_file, electron_dn_num)
        trexio.write_nucleus_num(trexio_file, nucleus_num)
        trexio.write_nucleus_charge(trexio_file, atom_charges_list)
        trexio.write_nucleus_label(trexio_file, chemical_symbol_list)
        trexio.write_nucleus_coord(trexio_file, coords_np)

        ##########################################
        # basis set info
        ##########################################
        # check the orders of the spherical atomic basis in pyscf!!
        # gto.spheric_labels(mol, fmt="%d, %s, %s, %s")
        # for s -> s
        # for p -> px, py, pz
        # for l >= d -> m=(-l ... 0 ... +l)

        basis_type="G" # thanks anthony!
        basis_shell_num=int(np.sum([mol.atom_nshells(i) for i in range(nucleus_num)]))
        nucleus_index=[]
        for i in range(nucleus_num):
            for _ in range(len(mol.atom_shell_ids(i))):
                nucleus_index.append(i)
        shell_ang_mom=[mol.bas_angular(i) for i in range(basis_shell_num)]
        basis_prim_num=int(np.sum([mol.bas_nprim(i) for i in range(basis_shell_num)]))

        basis_exponent=[]
        basis_coefficient=[]
        for i in range(basis_shell_num):
            for bas_exp in mol.bas_exp(i):
                basis_exponent.append(float(bas_exp))
            for bas_ctr_coeff in mol.bas_ctr_coeff(i):
                basis_coefficient.append(float(bas_ctr_coeff))

        basis_shell_index=[]
        for i in range(basis_shell_num):
            for _ in range(len(mol.bas_exp(i))):
                basis_shell_index.append(i)

        # normalization factors
        basis_shell_factor = [1.0 for _ in range(basis_shell_num)] # 1.0 in pySCF

        # gto_norm(l, expnt) => l is angmom, expnt is exponent
        # Note!! Here, the normalization factor of the spherical part are not included.
        # The normalization factor is computed according to Eq.8 of the following paper
        # H. B. Schlegel and M. J. Frisch, Int. J. Quant.  Chem., 54(1995), 83-87.
        basis_prim_factor=[]
        for prim_i in range(basis_prim_num):
            coeff=basis_coefficient[prim_i]
            expnt=basis_exponent[prim_i]
            l=shell_ang_mom[basis_shell_index[prim_i]]
            basis_prim_factor.append(mol.gto_norm(l, expnt)/np.sqrt(4*np.pi)*np.sqrt(2*l+1))

        ##########################################
        # ao info
        ##########################################
        ao_cartesian = 0 # spherical basis representation
        ao_shell=[]
        for i, ang_mom in enumerate(shell_ang_mom):
            for _ in range(2*ang_mom + 1):
                ao_shell.append(i)
        ao_num=len(ao_shell)

        # 1.0 in pyscf (because spherical)
        ao_normalization = [1.0 for _ in range(ao_num)]

        ##########################################
        # mo info
        ##########################################
        mo_type="MO"

        if twist_average:
            mo_num = len(mf['mo_coeff'][k_index])
            mo_occupation = mf['mo_occ'][k_index]
            mo_energy = mf['mo_energy'][k_index]
            mo_coeff = mf['mo_coeff'][k_index]
        else:
            mo_num = len(mf['mo_coeff'])
            mo_occupation = mf['mo_occ']
            mo_energy = mf['mo_energy']
            mo_coeff = mf['mo_coeff']

        # mo reordering because mo_coeff[:,mo_i]!!
        mo_coeff=[mo_coeff[:, mo_i] for mo_i in range(mo_num)]

        logger.debug(mo_num)
        logger.debug(len(mo_coeff))
        logger.debug(mo_occupation)
        logger.debug(mo_energy)
        #logger.info(mo_coeff)

        # check if MOs are descending order with respect to "mo occ"
        # this is usually true, but not always true for RO (restricted open-shell) calculations.
        order_bool=all([True if mo_occupation[i] >= mo_occupation[i+1] else False for i in range(len(mo_occupation)-1)])
        logger.info(f"MO occupations are in the descending order ? -> {order_bool}")
        if not order_bool:
            logger.warning("MO occupations are not in the descending order!!")
            logger.warning("RO (restricted open-shell) calculations?")
            logger.warning("Reordering MOs...")
            # reordering MOs.
            # descending order (mo occ)
            reo_moocc_index = np.argsort(mo_occupation)[::-1]
            mo_occupation_o = [mo_occupation[l] for l in reo_moocc_index]
            mo_energy_o = [mo_energy[l] for l in reo_moocc_index]
            mo_coeff_o = [mo_coeff[l] for l in reo_moocc_index]
            # descending order (mo energy)
            mo_coeff=[]; mo_occupation=[]; mo_energy=[]
            set_mo_occupation=sorted(list(set(mo_occupation_o)), reverse=True)
            for mo_occ in set_mo_occupation:
                mo_re_index=[i for i,mo in enumerate(mo_occupation_o) if mo == mo_occ]
                mo_occupation_t = [mo_occupation_o[l] for l in mo_re_index]
                mo_energy_t = [mo_energy_o[l] for l in mo_re_index]
                mo_coeff_t = [mo_coeff_o[l] for l in mo_re_index]
                reo_ene_index = np.argsort(mo_energy_t)
                mo_occupation += [mo_occupation_t[l] for l in reo_ene_index]
                mo_energy += [mo_energy_t[l] for l in reo_ene_index]
                mo_coeff += [mo_coeff_t[l] for l in reo_ene_index]

        logger.debug(mo_num)
        logger.debug(len(mo_coeff))
        logger.debug(mo_occupation)
        logger.debug(mo_energy)
        #logger.debug(mo_coeff)

        permutation_matrix=[] # for ao and mo swaps, used later

        # molecular orbital reordering
        # TREX-IO employs (m=-l,..., 0, ..., +l) for spherical basis
        mo_coefficient=[]

        for mo_i in range(mo_num):
            mo=mo_coeff[mo_i]
            mo_coeff_buffer=[]

            perm_list=[]
            perm_n=0
            for ao_i, ao_c in enumerate(mo):

                # initialization
                if ao_i==0:
                    mo_coeff_for_reord=[]
                    current_ang_mom=-1

                # read ang_mom (i.e., angular momentum of the shell)
                bas_i=ao_shell[ao_i]
                ang_mom=shell_ang_mom[bas_i]

                previous_ang_mom=current_ang_mom
                current_ang_mom=ang_mom

                # set multiplicity
                multiplicity = 2 * ang_mom + 1
                #print(f"multiplicity = {multiplicity}")

                # check if the buffer is null, when ang_mom changes
                if previous_ang_mom != current_ang_mom:
                    assert len(mo_coeff_for_reord) == 0

                if current_ang_mom==0: # s shell
                    #print("s shell/no permutation is needed.")
                    #print("(pyscf notation): s(l=0)")
                    #print("(trexio notation): s(l=0)")
                    reorder_index=[0]

                elif current_ang_mom==1: # p shell

                    #print("p shell/permutation is needed.")
                    #print("(pyscf notation): px(l=+1), py(l=-1), pz(l=0)")
                    #print("(trexio notation): pz(l=0), px(l=+1), py(l=-1)")
                    reorder_index=[2,0,1]


                elif current_ang_mom>=2: # > d shell

                    #print("> d shell/permutation is needed.")
                    #print("(pyscf notation): e.g., f3,-3(l=-3), f3,-2(l=-2), f3,-1(l=-1), f3,0(l=0), f3,+1(l=+1), f3,+2(l=+2), f3,+3(l=+3)")
                    #print("(trexio  notation): e.g, f3,0(l=0), f3,+1(l=+1), f3,-1(l=-1), f3,+2(l=+2), f3,-2(l=-2), f3,+3(l=+3), f3,-3(l=-3)")
                    l0_index=int((multiplicity-1)/2)
                    reorder_index=[l0_index]
                    for i in range(1, int((multiplicity-1)/2)+1):
                        reorder_index.append(l0_index+i)
                        reorder_index.append(l0_index-i)

                else:
                    raise ValueError("A wrong value was set to current_ang_mom.")

                mo_coeff_for_reord.append(ao_c)

                # write MOs!!
                if len(mo_coeff_for_reord) == multiplicity:
                    #print("--write MOs!!--")
                    mo_coeff_buffer+=[mo_coeff_for_reord[i] for i in reorder_index]

                    # reset buffer
                    mo_coeff_for_reord=[]

                    #print("--write perm_list")
                    perm_list+=list(np.array(reorder_index)+perm_n)
                    perm_n=perm_n+len(reorder_index)

            mo_coefficient.append(mo_coeff_buffer)
            permutation_matrix.append(perm_list)

        """
        #Phases are attached!!!??
        #this is also needed for a real WF (e.g., pi, pi, pi)
        phase_factor=complex(np.cos(np.sum(np.array(k_vec)) * 2*np.pi), -1 * np.sin(np.sum(np.array(k_vec))* 2*np.pi) )
        logger.info(f"phase factor = {phase_factor}")
        mo_coefficient = [[coeff * +1 * phase_factor for coeff in mo] for mo in mo_coefficient]
        """

        # here, we should think about complex cases
        #logger.info(mo_coefficient[0])
        # check if the MOs are imag.!
        imag_flags=[]
        for mo in mo_coefficient:
            imag_flags+=list(np.isreal(list(np.real_if_close(mo,tol=100))))
        #print(imag_flags)
        if all(imag_flags):
            complex_flag = False
        else:
            complex_flag = True
        #however, due to numerical error, we need more loose criteria
        imag_flags = []
        imag_thr=1.0e-3
        if complex_flag:
            for mo in mo_coefficient:
                imag_flags += list([True if np.abs(a.imag) < imag_thr else False for a in mo])
                if not all(list([True if np.abs(a.imag) < imag_thr else False for a in mo])):
                    logger.debug([np.abs(a.imag) for a in mo])
            if all(imag_flags):
                complex_flag = False
            else:
                complex_flag = True

        if complex_flag:
            logger.info("The WF is complex")
            mo_coefficient_real = []
            mo_coefficient_imag = []

            for mo__ in mo_coefficient:
                mo_real_b = []
                mo_imag_b = []
                for coeff in mo__:
                    mo_real_b.append(coeff.real)
                    mo_imag_b.append(coeff.imag)
                mo_coefficient_real.append(mo_real_b)
                mo_coefficient_imag.append(mo_imag_b)

        else:
            logger.info("The WF is real")
            mo_coefficient=[list(np.array(mo).real) for mo in mo_coefficient]

        logger.debug("--MOs Done--")

        ##########################################
        # atomic orbital integrals
        ##########################################

        def row_column_swap(inp_matrix, perm_list):
            mat_org=inp_matrix
            mat_row_swap=np.array([mat_org[i] for i in perm_list])
            mat_row_swap_T=mat_row_swap.T
            mat_row_swap_col_swap=np.array([mat_row_swap_T[i] for i in perm_list])
            mat_inv=mat_row_swap_col_swap.T

            """
            for i in range(len(mat_org)):
                for j in range(len(mat_org)):
                    assert np.round(mat_inv[i][j],10) == np.round(mat_inv[j][i],10)
                    #print("-------------------------")
            """

            return mat_inv

        perm_list=permutation_matrix[0]
        if pbc_flag:
            # logger.warning("1b integral for pbc is at gamma! Generic k points will be implemented.")
            intor_int1e_ovlp = row_column_swap(mol.pbc_intor("int1e_ovlp"), perm_list)
            intor_int1e_nuc = row_column_swap(mol.pbc_intor("int1e_nuc"), perm_list)
            intor_int1e_kin = row_column_swap(mol.pbc_intor("int1e_kin"), perm_list)
        else:
            intor_int1e_ovlp = row_column_swap(mol.intor("int1e_ovlp"), perm_list)
            intor_int1e_nuc = row_column_swap(mol.intor("int1e_nuc"), perm_list)
            intor_int1e_kin = row_column_swap(mol.intor("int1e_kin"), perm_list)

        ##########################################
        # basis set info
        ##########################################
        trexio.write_basis_type(trexio_file, basis_type) #
        trexio.write_basis_shell_num(trexio_file, basis_shell_num) #
        trexio.write_basis_prim_num(trexio_file, basis_prim_num) #
        trexio.write_basis_nucleus_index(trexio_file, nucleus_index) #
        trexio.write_basis_shell_ang_mom(trexio_file, shell_ang_mom) #
        trexio.write_basis_shell_factor(trexio_file, basis_shell_factor) #
        trexio.write_basis_shell_index(trexio_file, basis_shell_index) #
        trexio.write_basis_exponent(trexio_file, basis_exponent) #
        trexio.write_basis_coefficient(trexio_file, basis_coefficient) #
        trexio.write_basis_prim_factor(trexio_file, basis_prim_factor) #

        ##########################################
        # ao info
        ##########################################
        trexio.write_ao_cartesian(trexio_file, ao_cartesian) #
        trexio.write_ao_num(trexio_file, ao_num) #
        trexio.write_ao_shell(trexio_file, ao_shell) #
        trexio.write_ao_normalization(trexio_file, ao_normalization) #

        ##########################################
        # mo info
        ##########################################
        trexio.write_mo_type(trexio_file, mo_type) #
        trexio.write_mo_num(trexio_file, mo_num) #
        trexio.write_mo_occupation(trexio_file, mo_occupation) #

        if complex_flag:
            trexio.write_mo_coefficient(trexio_file, mo_coefficient_real)  #
            trexio.write_mo_coefficient_im(trexio_file, mo_coefficient_imag)  #
            logger.info(mo_coefficient_imag)
        else:
            trexio.write_mo_coefficient(trexio_file, mo_coefficient)  #

        ##########################################
        # ao integrals
        ##########################################
        trexio.write_ao_1e_int_overlap(trexio_file,intor_int1e_ovlp)
        trexio.write_ao_1e_int_kinetic(trexio_file,intor_int1e_kin)
        trexio.write_ao_1e_int_potential_n_e(trexio_file,intor_int1e_nuc)

        ##########################################
        # ECP
        ##########################################
        # internal format of pyscf
        # https://pyscf.org/pyscf_api_docs/pyscf.gto.html?highlight=ecp#module-pyscf.gto.ecp
        """
        { atom: (nelec,  # core electrons
        ((l, # l=-1 for UL, l>=0 for Ul to indicate |l><l|
        (((exp_1, c_1), # for r^0
        (exp_2, c_2), …),
        
        ((exp_1, c_1), # for r^1
        (exp_2, c_2), …),
        
        ((exp_1, c_1), # for r^2
        …))))),
        
        …}
        """

        # Note! here, the smallest l for the local part is l=1(i.e., p).
        # As a default, nwchem does not have a redundant non-local term (i.e., coeff=0) for H and He.

        if len(mol._ecp) > 0:

            ecp_num=0
            ecp_max_ang_mom_plus_1=[]
            ecp_z_core=[]
            ecp_nucleus_index=[]
            ecp_ang_mom=[]
            ecp_coefficient=[]
            ecp_exponent=[]
            ecp_power=[]

            for nuc_index, chemical_symbol in enumerate(chemical_symbol_list):
                #print(f"Chemical symbol is {chemical_symbol}")
                z_core, ecp_list = mol._ecp[chemical_symbol]

                #ecp zcore
                ecp_z_core.append(z_core)

                #max_ang_mom+1
                """ wrong!!
                max_ang_mom_minus_1 = max([ecp[0] for ecp in ecp_list])
                if max_ang_mom_minus_1 == -1: # special case!! H and He. PySCF database does not define the ul-s part for them.
                    max_ang_mom = 1
                    max_ang_mom_plus_1 = 2
                else:
                    max_ang_mom = max_ang_mom_minus_1 + 1
                    max_ang_mom_plus_1 = max_ang_mom_minus_1 + 2
                ecp_max_ang_mom_plus_1.append(max_ang_mom_plus_1)
                """

                max_ang_mom = max([ecp[0] for ecp in ecp_list]) # this is lmax, right?
                if max_ang_mom == -1: # special case!! H and He. PySCF database does not define the ul-s part for them.
                    max_ang_mom = 0
                    max_ang_mom_plus_1 = 1
                else:
                    max_ang_mom_plus_1 = max_ang_mom + 1
                ecp_max_ang_mom_plus_1.append(max_ang_mom_plus_1)

                for ecp in ecp_list:
                    ang_mom=ecp[0]
                    if ang_mom==-1:
                        ang_mom=max_ang_mom_plus_1
                    for r, exp_coeff_list in enumerate(ecp[1]):
                        for exp_coeff in exp_coeff_list:
                            exp,coeff = exp_coeff

                            #store variables!!
                            ecp_num+=1
                            ecp_nucleus_index.append(nuc_index)
                            ecp_ang_mom.append(ang_mom)
                            ecp_coefficient.append(coeff)
                            ecp_exponent.append(exp)
                            ecp_power.append(r-2)

                # special case!! H and He.
                # For the sake of clarity, here I put a dummy coefficient (0.0) for the ul-s part here.
                ecp_num+=1
                ecp_nucleus_index.append(nuc_index)
                ecp_ang_mom.append(0)
                ecp_coefficient.append(0.0)
                ecp_exponent.append(1.0)
                ecp_power.append(0)

            # write to the trex file
            trexio.write_ecp_num(trexio_file, ecp_num) #
            trexio.write_ecp_max_ang_mom_plus_1(trexio_file, ecp_max_ang_mom_plus_1) #
            trexio.write_ecp_z_core(trexio_file, ecp_z_core) #
            trexio.write_ecp_nucleus_index(trexio_file, ecp_nucleus_index) #
            trexio.write_ecp_ang_mom(trexio_file, ecp_ang_mom) #
            trexio.write_ecp_coefficient(trexio_file, ecp_coefficient) #
            trexio.write_ecp_exponent(trexio_file, ecp_exponent) #
            trexio.write_ecp_power(trexio_file, ecp_power) #

        # close the TREX-IO file
        trexio_file.close()

    logger.info("Conversion to TREXIO is done.")

def cli():
    import argparse
    from logging import config, getLogger, StreamHandler, Formatter

    log_level = "INFO"
    logger = getLogger("pyscf-trexio")
    logger.setLevel(log_level)
    stream_handler = StreamHandler()
    stream_handler.setLevel(log_level)
    # handler_format = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
    handler_format = Formatter('%(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    # define the parser
    parser = argparse.ArgumentParser(epilog='From pyscf chk file to TREXIO file',
                                     usage='python pyscf_to_trexio.py -c pyscf_checkfile -o trexio_filename',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--pyscf_checkfile', help=f'pyscf checkfile', type=str, required=True)
    parser.add_argument('-o', '--trexio_filename', help=f'trexio filename', type=str, default="trexio.hdf5")

    # parse the input values
    args = parser.parse_args()
    parsed_parameter_dict = vars(args)

    pyscf_to_trexio(pyscf_checkfile=args.pyscf_checkfile, trexio_filename=args.trexio_filename)

if __name__ == "__main__":
    logger = getLogger("Turbo-Genius")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("DEBUG")
    handler_format = Formatter('%(name)s - %(levelname)s - %(lineno)d - %(message)s')
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)

    cli()

    """
    # moved to examples
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
    from utils_workflows.env import turbo_genius_root
    from pyscf_wrapper import Pyscf_wrapper

    pyscf_to_trexio_test_dir = os.path.join(turbo_genius_root, "tests", "trexio_to_turborvb")

    os.chdir(pyscf_to_trexio_test_dir)
    pyscf_to_trexio(pyscf_checkfile="diamond_q.chk", trexio_filename="diamond_trexio_q.hdf5")
    """