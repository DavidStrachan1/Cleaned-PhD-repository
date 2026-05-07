
leftlim(m::AbstractMPS) = m.llim
rightlim(m::AbstractMPS) = m.rlim




"""
Initialising state
"""

export initial_state_gates
export initial_state_gates_separated
export initialise_psi
export initialise_psi_old


"""
MPO Choi Isomorphism functions
"""

export extract_map_as_MPO
export partial_tr
export partial_tr_v2


"""
Non MPO Choi Isomorphism functions.
"""

export particle_hole_transform
export NESS_fn
export QN_matching
export QNumbers
export rdm_para
export vectorise_ρ
export choi_isomorphism


"""
MPO Map contraction and system extraction, keeping an interleaved ordering
for efficiency
"""

export switch_linkind_direction
export efficient_map_contraction
export efficient_system_state_extraction
export swap_gate_new


"""
Map contraction and system extraction, using a separated system ancilla
ordering. This is a poor implementation as the ancilla modes 
are separated from their entangled system modes, causing 
a rainbow state and a large bond dimension.
"""

export contract_ρ_with_ρΛ
export extract_system_state_as_MPO_deprecated
export comparison_between_non_MPO_vs_MPO_map_extraction


"""
Helper functions for map extraction
"""

export expect_op
export change_inds
export particle_hole_transform_MPO
export particle_hole_transform_on_system


"""
Corrections due to fermi properties.
"""

export ancilla_phase_gate_PH
export ancilla_phase_gate_swap
export system_swaps
export swap_gate


"""
Helper functions, mostly for fermions
"""

export MPO_to_Tensor
export removeqns_mod
export map_check
export map_check2 
export unprime_ind
export unprime_string
export JW_string
export Id_string
export create_fermi_ann_op
export create_fermi_cre_op
export apply_gates_to_ρ
export apply_gates_to_ψ
export spin_operators
export JW_string_mat
export matrix_operators


"""
Enrichment functions
"""

export enrich_generic3
export Krylov_linkdims
export Krylov_states
export bipart_maxdim
export expand_
export growbond!
export growMPS!

"""
TDVP measure functions
"""

export perform_GC
export measure_mem
export current_time 
export measure_den
export measure_SIAM_double_occ
export measure_SIAM_diag_elements
export measure_SIAM_diag_elements_spinful
export measure_SvN
export measure_correlation_matrix
export measure_spin_dn_electron_correlation_matrix
export measure_spin_up_electron_correlation_matrix
export measure_spin_correlation_matrix
export measure_pauli_x
export measure_pauli_y
export measure_pauli_z
export measure_den_SF
export measure_norm
export entanglement_entropy
export boundary_test
export leftvacuum


function initial_state_gates_separated(ψ_input,P,DP;kwargs...)
    
    """
    This function is a way of getting round the weird behaviour of the Jordan Wigner strings
    inside ITensor (https://itensor.discourse.group/t/are-jordan-wigner-strings-handled-in-apply/2266).
    This initialises the Choi state in separated form with no Jordan Wigner operators considered, by initialising in 
    interleaved format which doesn't implement any JW factors, then applying qubit swap gates.
    """

    """
    NOTE:interleaved_inds to reverse(interleaved_inds) was changed in January 2026.
    """

    (;sys_mode_type) = P
    (;cdag,Id,q,s) = DP

    if sys_mode_type == "Electron"
        cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagdn","Adagup"]]) for n in 1:length(s)]
    end

    ψ = deepcopy(ψ_input)
    interleaved_inds = q[1:2:end]
    system_gate = [(cdag[n][i]*Id[n+1] + Id[n]*cdag[n+1][i])/sqrt(2) for i in 1:length(cdag[1]) for n in reverse(interleaved_inds)]
    ψ = apply(system_gate,ψ;cutoff=1e-15)
    ψ = system_swaps(ψ,q[1],DP,P;use_spin_operators = true)
    return ψ 
end

function initial_state_gates(system_inds,P,DP;kwargs...)
    ##Creates the gates for the maximally entangled pairs. Note that if the interleaved ordering is chosen,
    ##the JW string strings other than wrt to each other can be ignored as the pairs are created sequentially, 
    ## so orienting the JW string in direction of the order the pairs are being created, Z always acts on the vac.
    ##This is not the case for the separated ordering.

    (;Ns,sys_mode_type,compute_maps_bool,method_type,init_occ,using_system_ancilla,ordering_choice) = P
    (;cdag,Id,F,s,q) = DP


    if sys_mode_type == "Electron"
        cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagdn","Adagup"]]) for n in 1:length(s)]
    end

    init_occ_vec = get(kwargs,:init_occ_vec,init_occ*ones(length(system_inds)))
    if method_type == "SFMC"
        system_gate = [init_occ_vec[j]*cdag[n][i]*Id[n+1]+(1-init_occ_vec[j])*F[n]*cdag[n+1][i] for i in length(cdag[1]) for (j,n) in enumerate(system_inds)] 
    else
        if compute_maps_bool == true
            if ordering_choice == "interleaved"
                system_gate = [(cdag[n][i]*Id[n+1] + Id[n]*cdag[n+1][i])/sqrt(2) for i in 1:length(cdag[1]) for n in system_inds]

                # if sys_mode_type == "S=1/2"
                #     system_gate = [(cdag[n][i]*Id[n+1] + Id[n]*cdag[n+1][i])/sqrt(2) for i in 1:length(cdag[1]) for n in system_inds]
                # else

                #     system_gate = [(cdag[n][i]*Id[n+1] + F[n][i]*cdag[n+1][i])/sqrt(2) for i in 1:length(cdag[1]) for n in system_inds]
                # end
            else
                if sys_mode_type == "S=1/2"
                    system_gate = [(cdag[n][i]*Id[n+Ns] + Id[n]*cdag[n+Ns][i])/sqrt(2) for i in 1:length(cdag[1]) for n in system_inds]
                else
                    system_gate = [(cdag[n][i]*Id[n+Ns] + Id[n]*cdag[n+Ns][i])/sqrt(2) for i in 1:length(cdag[1]) for n in system_inds]
                    #system_gate = [(create_fermi_cre_op(n,q,DP;spin=i,cdag=cdag)+create_fermi_cre_op(n+Ns,q,DP;spin=i,cdag=cdag))/sqrt(2) for i in 1:length(cdag[1]) for n in system_inds]
                end
            end
        else
            if using_system_ancilla 
                if ordering_choice == "interleaved"
                    if sys_mode_type == "S=1/2"
                        system_gate = [(√(init_occ_vec[j])*cdag[n][i]*Id[n+1] + √(1-init_occ_vec[j])*Id[n]*cdag[n+1][i])/sqrt(2) for i in 1:length(cdag[1]) for (j,n) in enumerate(system_inds)]
                    else
                        system_gate = [(√(init_occ_vec[j])*cdag[n][i]*Id[n+1] + √(1-init_occ_vec[j])*F[n]*cdag[n+1][i])/sqrt(2) for i in 1:length(cdag[1]) for (j,n) in enumerate(system_inds)]
                    end
                else
                    if sys_mode_type == "S=1/2"
                        system_gate = [(√(init_occ_vec[j])*cdag[n][i]*Id[n+Ns] +√(1-init_occ_vec[j])*Id[n]*cdag[n+Ns][i])/sqrt(2) for i in 1:length(cdag[1]) for (j,n) in enumerate(system_inds)]
                    else
                        system_gate = [(√(init_occ_vec[j])*create_fermi_cre_op(n,q,DP;spin=i,cdag=cdag)+√(1-init_occ_vec[j])*create_fermi_cre_op(n+Ns,q,DP;spin=i,cdag=cdag))/sqrt(2) for i in 1:length(cdag[1]) for (j,n) in enumerate(system_inds)]
                    end
                end
            elseif init_occ_vec == zeros(length(system_inds))
                system_gate = [Id[n]  for n in system_inds]
            elseif init_occ_vec == ones(length(system_inds))
                system_gate = [cdag[n][i] for i in 1:length(cdag[1]) for n in system_inds]
            else
                system_gate = [(√(init_occ_vec[j])*cdag[n][i] + √(1-init_occ_vec[j])*Id[n]) for i in 1:length(cdag[1])  for (j,n) in enumerate(system_inds)]
            end
        end
    end
    return system_gate
end

function initialise_psi(P,DP;kwargs...)
    (;Ns,N_L,N_R,ordering_choice,sys_mode_type,bath_mode_type,compute_maps_bool,
    init_occ,method_type,Nc,using_system_ancilla,init_occ) = P
    (;s,qtot,cdag,Id,F,q,qS,qB_L,qB_R,N) = DP    
    
    """
    NOTE: This function assumes an interleaved ordering.
    """

    if bath_mode_type == "Electron"
        Empty = "Emp"
        Full = "UpDn"
    else
        Empty = "0"
        Full = "1"
    end
    occs = [Empty for n in qtot]
    if bath_mode_type != "Boson" 
        if method_type == "SFMC"
            """
            This represents a filled bath mode and an empty bath mode interleaved
            with their respective superfermion, which has opposite filling for QN conservation.
            """
            bath_occs = [Full,Empty,Empty,Full] 
            N_L>0 && (left_bath_occs = repeat(bath_occs,Int(length(qB_L)/4)))
            N_R>0 && (right_bath_occs = repeat(bath_occs,Int(length(qB_R)/4)))
        else
            bath_occs = [Full,Empty]
            N_L>0 && (left_bath_occs = repeat(bath_occs,Int(length(qB_L)/2)))
            N_R>0 && (right_bath_occs = repeat(bath_occs,Int(length(qB_R)/2)))
        end
        N_L>0 && (occs[qB_L] = left_bath_occs)
        N_R>0 && (occs[qB_R] = right_bath_occs)
    end
    therm = MPS(ComplexF64,s,occs)

    if ordering_choice == "separated" && compute_maps_bool
        if P.U ==0
            println("Note if separated ordering is used and you want to use correlation matrix map extraction method, initial_state_gates should be
            used along with the spin transform in the map calculation function. Alternatively, initial_state_gates_separated can be used
            with the fermionic particle hole transform, but I'm not sure why this works. 
            ")
        end

        ψ_init = initial_state_gates_separated(therm,P,DP;kwargs...)
    else
        system_gate = initial_state_gates(qS,P,DP;kwargs...)
        ψ_init = apply(system_gate,therm;cutoff=1e-15)
    end

    # Normalize state
    if P.method_type == "SFMC"
        left_vac = leftvacuum(s)
        nrm = inner(left_vac, ψ_init)
        ψ_init .= ψ_init / nrm
        @show(inner(left_vac, ψ_init))
    else
        orthogonalize!(ψ_init,N)
        ψ_init[DP.N]= ψ_init[N]/norm(ψ_init)
        @show(norm(ψ_init))
    end

    #Not sure if the code below is needed

    # ##Create total JW string created from all the gates put together. The ith system ancilla 
    # ## has a JW string across all pairs before it (1->i-1), so the ith pair has a JW factor of
    # ##(F[qS[i]]*F[qA[i]])^(Ns-i). This translates to the following conditions.
    # if compute_maps_bool
    #     JW_string = Vector{String}(undef,2*Ns)
    #     for (i,n) in enumerate(1:2:2*Ns)
    #         if iseven(Ns) && isodd(i)
    #             JW_string[n:n+1] .= "F"
    #         elseif isodd(Ns) && iseven(i)
    #             JW_string[n:n+1] .= "F"
    #         else
    #             JW_string[n:n+1] .= "Id"
    #         end
    #     end    

    #     for (i,JW) in enumerate(JW_string)
    #         op1 = op(JW,s,q[i])
    #         therm = apply(op1,ψ_init)
    #     end
    # end


    if method_type == "SFMC"
        ###Patchy way to get Ci_physical
        @assert(using_system_ancilla == false)
        N_physical = 2*N_L+2*N_R+Ns
        sys_modes = 2*N_L+1:2*N_L+Ns
        Ci_physical = complex(zeros(N_physical,N_physical))
        N_L>0 && [Ci_physical[i,i] = isodd(i) ? 1 : 0 for i in (1:2*N_L)]
        [Ci_physical[i,i] = init_occ for i in sys_modes]
        N_R>0 && [Ci_physical[i,i] = isodd(n) ? 1 : 0 for (n,i) in enumerate(sys_modes[end]+1:N_physical)]

        return ψ_init,Ci_physical
    else
        return ψ_init
    end
end


function initialise_psi_old(P,DP)
    (;Ns,N_L,ordering_choice,sys_mode_type,bath_mode_type,compute_maps_bool,
    init_occ) = P
    (;s,qtot,cdag,Id,q,qS,qB_L) = DP    
    """
    Thermofield vacuum
    """
    occs = ["0" for n in qtot]
    if bath_mode_type == "Fermion"
        occs = [isodd(n) ? "1" : "0" for n in qtot]
        occs = [i in q ? "0" : occs[i] for i in eachindex(occs)]
    end
    therm = MPS(ComplexF64,s,occs)
    if compute_maps_bool == true
        if ordering_choice == "interleaved"
            if sys_mode_type == "Fermion"
                system_gate = [(create_fermi_cre_op(n,q,DP)+ create_fermi_cre_op(n+1,q,DP))/sqrt(2) for n in qS]
            else
                system_gate = [(cdag[n]*Id[n+1] + Id[n]*cdag[n+1])/sqrt(2) for n in qS]
            end
        elseif ordering_choice == "separated"
            if sys_mode_type == "Fermion"
                system_gate = [(create_fermi_cre_op(n,q,DP)+ create_fermi_cre_op(n+Ns,q,DP))/sqrt(2) for n in qS]
            else
                system_gate = [(Id[n+Ns]*cdag[n] + Id[n]*cdag[n+Ns])/sqrt(2) for n in qS]
            end
        end
    else
        system_gate =  [(√(init_occ)*create_fermi_cre_op(n,q,DP)+√(1-init_occ)*Id_string(qB_L[end],q[end],DP))/sqrt(2) for n in qS]
    end
    return apply(system_gate,therm;cutoff=1e-15)
end









"MPO Choi Isomorphism functions
---------------------------------------------------------------------------------------------------------------------------------------"

function extract_map_as_MPO(ψ,PH_bool,P,DP; kwargs...)
    """
    the input argument interleaved_or separated determines whether
    the system ancilla modes are left in their interleaved ordering,
    or are rearranged into a separated ordering. The separated ordering
    simplifies the contraction of the map and the input state, but greatly
    increases the maximum bond dimension so shouldn't be used for large systems.
    """

    interleaved_or_separated = get(kwargs, :interleaved_or_separated, "interleaved")

    (;Ns,sys_mode_type,ordering_choice) = P
    (;q,qS,qA,N) = DP

    #Applying corrective phase gates
    if sys_mode_type == "Fermion"
        if ordering_choice == "interleaved" && interleaved_or_separated=="separated"
            println("Perform swap phase gates")
            ##swap phases.
            gates =  ancilla_phase_gate_swap(qA,true,DP,P)
            @time ψ = apply_gates_to_ψ(ψ,gates)
        end
    
        if PH_bool
            ##Particle hole phases.
            if isodd(Ns)

                gates =  ancilla_phase_gate_PH(qA,true,DP,P)
                ψ = apply_gates_to_ψ(ψ,gates)
            end
        end
    end

    #Putting MPS into separated ordering between ancillas and system modes.
    if ordering_choice == "interleaved" && interleaved_or_separated== "separated"
        ###Applies fermionic swap gates to change the order from interleaved to separated.
        println("Perform swap gates to move from interleaved to separated ordering")
        @time ψ = system_swaps(ψ,q[1],DP,P)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end

    
    println("Time for partial_tr_v2")
    rm_inds = q
    @time ρΛ_MPO = partial_tr_v2(ψ,rm_inds,DP,P)
    #Apply particle hole transform

    # println("Time for partial_tr")
    # @time begin
    #     M = outer(ψ',ψ)
    #     partial_tr(M,rm_inds,P)
    # end
    
    if PH_bool
        ρΛ_MPO =  removeqns_mod(ρΛ_MPO)

        gates_MPO =  particle_hole_transform_MPO(qS,qA,DP,P)
        ρΛ_MPO = apply_gates_to_ρ(ρΛ_MPO,gates_MPO,false)
        #orthogonalize!(ρΛ_MPO,1)
    end
    ρΛ_MPO = 2^(Ns)*ρΛ_MPO
    return ρΛ_MPO
end


function partial_tr(M::MPO,rm_inds,P; plev::Pair{Int,Int}=0 => 1, tags::Pair=ts"" => ts"")
    (;N_L,N_R,Ns) = P
    N = length(M)
    M_reduced = MPO(length(rm_inds))
    
    """
    Tracing out the left bath
    """
    if N_L >0
        L = tr(M[1]; plev=plev, tags=tags)
        for j in 2:rm_inds[1]-1
            L *= M[j]
            L = tr(L; plev=plev, tags=tags)
        end
        L *= M[rm_inds[1]]
        M_reduced[1] = L
    else
        M_reduced[1] = M[rm_inds[1]]
    end
    
    """
    Tracing out the right bath
    """    
    if N_R >0
        R = tr(M[end]; plev=plev, tags=tags)
        for j in reverse(rm_inds[end]+1:N-1)
            R *= M[j]
            R = tr(R; plev=plev, tags=tags)
        end
        if length(rm_inds) == 1 && N_L >0
            M_reduced[end] *= R 
        else
            R *= M[rm_inds[end]]
            M_reduced[end] = R
        end
    else
        M_reduced[end] = M[rm_inds[end]]
    end

    """
    Making new MPO
    """
    for (i,ind) in enumerate(rm_inds[2:end-1])
        M_reduced[i+1] = M[ind]
    end
    orthogonalize!(M_reduced,1)
    return M_reduced
end



function partial_tr_v2(ψ::MPS,rm_inds,DP,P; plev::Pair{Int,Int}=0 => 1, tags::Pair=ts"" => ts"")
    
    (;N_L,N_R,Ns) = P
    (;N) = DP

    ψ_copy = deepcopy(ψ)
    ψdag = dag(ψ)
    ITensors.prime!(linkinds, ψ_copy)
    
    

    ρl =  1
    ρr =  1

    """
    Tracing out the left bath
    """
    

    if N_L >0
        ρl =  ψ_copy[1]*ψdag[1]
        for j in 2:rm_inds[1]-1

            ρl = ρl* ψdag[j]
            ρl = ρl* ψ_copy[j]
        end
    end

    
    """
    Tracing out the right bath
    """ 

    
    if N_R >0    
        for j in reverse(rm_inds[end]+1:N)

            ρr = ρr* ψdag[j]
            ρr = ρr* ψ_copy[j]
        end
    end


    ψ_reduced = MPS(length(rm_inds))

    

    for (i,ind) in enumerate(rm_inds)
        ψ_reduced[i] = ψ[ind]
    end
    orthogonalize!(ψ_reduced,1)
    ρ = outer(ψ_reduced',ψ_reduced; cutoff= 1e-10)
    ρ[1] *= ρl
    ρ[end] *= ρr
    return ρ
end
















"Non MPO Choi Isomorphism functions
------------------------------------------------------------------------------------"


function particle_hole_transform(ind,ITensor_bool,DP,P;kwargs...)
    (;Ns,sys_mode_type) = P
    (;q,c,cdag,s) = DP

    use_spin_operators = get(kwargs,:use_spin_operators,false)
    
    if sys_mode_type == "S=1/2"
        use_spin_operators= true
    end
    if sys_mode_type == "Electron" && !ITensor_bool
        println("Map extraction using matrices rather than Tensors is not implemented for
        Electron site type")
    end

    if sys_mode_type == "Electron" && !use_spin_operators    
        println("Map extraction using fermionic operators with fermionic corrections 
        rather than qubit operators with no corrections is not implemented for
        Electron site type")
    end

    if sys_mode_type == "Electron"
        cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagdn","Adagup"]]) for n in 1:length(s)]
        c =  [ops(s, [(cre_op, n) for cre_op in ["Adn","Aup"]]) for n in 1:length(s)]
    end

    if ITensor_bool
        (;c,cdag) = DP
        gates = [cdag[n][spin] + c[n][spin] for n in ind for spin in 1:length(c[1])]
    else

        cdag_mat,c_mat = matrix_operators(2*Ns,P)

        # Use spin_operators if initial_state_gate is used, not
        # initial_state_gate_separated
        if use_spin_operators
            _,cdag_mat,c_mat,_ = spin_operators(2*Ns)
        end
        gates = [cdag_mat[n] + c_mat[n] for n in ind]
    end

    # ### For fermions, subtract them if it's even, add them if they're odd
    # if (sys_mode_type == "Fermion" || sys_mode_type == "Electron") && !use_spin_operators
    #     if ITensor_bool
    #         (;q,c,cdag) = DP
    #         gates = [(create_fermi_cre_op(n,q,DP;spin)
    #         -((-1)^(Ns))*create_fermi_ann_op(n,q,DP;spin)) for n in ind for spin in 1:length(c[1])]
    #     else
    #         cdag_mat,c_mat = matrix_operators(2*Ns,P)
    #         gates = [cdag_mat[n] - ((-1)^(Ns))*c_mat[n] for n in ind]
    #     end
    # else
    #     if ITensor_bool
    #         gates = [cdag[n][spin] + c[n][spin] for n in ind for spin in 1:length(c[1])]
    #     else
    #         cdag_mat,c_mat = matrix_operators(2*Ns,P)
    #         gates = [cdag_mat[n] + c_mat[n] for n in ind]
    #     end
    # end

    return gates    
end

function NESS_fn(ψ_input,P,DP;kwargs...)
    (;Ns,sys_mode_type,ordering_choice) = P
    (;s,q,qS,qA) = DP

    """
    NOTE:The problems with this function and the phase corrections are now fixed 

    """
    use_spin_operators = get(kwargs,:use_spin_operators,false)
    
    # if sys_mode_type == "S=1/2"
    #     use_spin_operators=true
    # elseif ordering_choice == "interleaved" && P.tc != 0 && use_spin_operators == true
    #     println("map extraction using spin operators is invalid for this setup,
    #     switching to using fermionic operators w phases.")
    #     use_spin_operators = false
    # elseif ordering_choice == "separated" && use_spin_operators == false
    #     println("map extraction using fermionic operators is invalid for this setup,
    #     switching to using spin operators .")
    #     use_spin_operators = true
    # end

    ψ = deepcopy(ψ_input)
    if !use_spin_operators
        if sys_mode_type == "Fermion" || sys_mode_type == "Electron"
            if sys_mode_type == "Electron"
                println("Map extraction using fermionic operators with fermionic corrections 
                rather than qubit operators with no corrections is not implemented for
                Electron site type")
            end
            if ordering_choice == "interleaved"
                ##swap phases.
                gates =  ancilla_phase_gate_swap(qA,true,DP,P)
                ψ = apply_gates_to_ψ(ψ,gates)
            end
            
            # ##Particle hole phases.
            # if isodd(Ns)
            #     gates =  ancilla_phase_gate_PH(qA,true,DP,P)
            #     ψ = apply_gates_to_ψ(ψ,gates)
            # end
        end
    end

    if ordering_choice == "interleaved"
        ###Applies fermionic swap gates to change the order from interleaved to separated.
        ψ = system_swaps(ψ,q[1],DP,P;kwargs...)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end
    ##calculate reduced density matrix.
    rm_inds = [qS ; qA]
    ρf = rdm_para(ψ,rm_inds,DP,P)

    ###applies a particle hole transformation to the ancilla states.
    gates =  particle_hole_transform(qA,true,DP,P;kwargs...)
    ρf = apply_gates_to_ρ(ρf,gates,true)

    ###Convert Tensor to matrix.
    cutoff = 1e-5
    ρmat,Λmat = choi_isomorphism(ρf,qA,qS,cutoff,P,DP)
    return ρmat,Λmat
end

function QN_matching(x,y,d,symmetry_subspace)
    if symmetry_subspace == "Full"
        return true
    else
        QN_x = QNumbers(x,d,symmetry_subspace)
        QN_y = QNumbers(y,d,symmetry_subspace)
        if QN_x == QN_y
            return true
        else
            return false
        end
    end
end

function QNumbers(index_string,d,symmetry_subspace)
    if d == 4 
        ##if d is 4, we have both particle number conservation and Sz conversation
        ## 0 = |0>, N = 0, Sz = 0
        ## 1 = |↑>, N = 1, Sz = 1
        ## 2 = |⤓>, N = 1, Sz = -1
        ## 3 = |⤓↑>,N = 2, Sz = 0
        
        # Count occurrences of each value
        counts = Dict(i => count(==(i), index_string) for i in [1,2,3,4])
        QN_N = counts[2]+ counts[3] + 2*counts[4]
        QN_Sz = counts[2]-counts[3]
        if symmetry_subspace == "Number and Sz conserving"
            return [QN_N,QN_Sz]
        else
            return [QN_N]
        end
    else
        QN = sum(index_string .- 1)
        return [QN]
    end
end

function rdm_para(ψ_input,rm_inds,DP,P)
    (;Ns,N_L,N_R,symmetry_subspace) = P
    (;q,s,N,qtot) = DP
    N_inds = length(rm_inds)
    left_bath_bool,right_bath_bool = N_L>0,N_R>0
    d = NDTensors.dim(s[1])
    lk = ReentrantLock()

    ψ = deepcopy(ψ_input)

    ψdag = dag(ψ)
    ITensors.prime!(linkinds, ψdag)
    rdm_ = ITensor(dag(s[rm_inds]),s[rm_inds]')
    ρl =  1
    ρr =  ψdag[N]*ψ[N]

    ##Trace out left bath
    if left_bath_bool
        ρl =  ψdag[1]*ψ[1]
        left_inds = qtot[2:rm_inds[1]-1]
        for k in left_inds
            ρl = ρl* ψdag[k]
            ρl = ρl* ψ[k]
        end
    end

    ##Trace out right bath
    if right_bath_bool
        right_inds = qtot[rm_inds[end]+1:end-1]
        for k in reverse(right_inds)   
            ρr = ρr* ψdag[k]
            ρr = ρr* ψ[k]
        end
    end

    @threads for i=0:(d^(2*N_inds) - 1)  
        #The first 2Ns (N_inds) indices are taken as s[rm_inds] and the last 2Ns (N_inds) indices are taken as s[rm_inds]'.
        ##I'm deliberately contracting ψdag[j] and ψ[j] with ρ separately to prevent creating a tensor of size
        ##χ^4 with χ being the local bond dimension. 
        ##The largest tensor created is of size \chi^2*d where d is the site dimension (2).
        v = Vector{Any}(undef,N_inds)
        w = Vector{Any}(undef,N_inds)
        
        #creating a ditstring (bitstring of base d), representing the indices of the density matrix.
        #x gives the indices of s[q], y gives the indices of s[q]'.
        ditstring = zeros(Int,2*N_inds)
        dit = reverse(digits(i,base=d))
        ditstring[(end-(length(dit))+1):end] = dit
        x = ditstring[1:N_inds] .+1
        y = ditstring[N_inds+1:end] .+1

        # inds_tot = split(bitstring(Int32(i)),"")[(end-2*N_inds+1):end]
        # @show(inds_tot)
        # inds = inds_tot[1:N_inds]
        # prime_inds = inds_tot[(N_inds+1):end]
        # x =  ([parse(Int8,ind) for ind in inds] .+1)
        # y =  ([parse(Int8,ind) for ind in prime_inds] .+1)

        if QN_matching(x,y,d,symmetry_subspace)
            local ρ = copy(ρl)
            lock(lk) do 
                b = 0
                for k in rm_inds
                    b += 1
                    C1 = ψ[k]*onehot(dag(s[k])=>x[b])
                    C2 = ψdag[k]*onehot(s[k]=>y[b])
                    ρ = ρ*C1
                    ρ = ρ*C2
                    v[b] = dag(s[k]) => x[b]
                    w[b] = s[k]' => y[b]
                end
                ρ = ρ*ρr
                rdm_[v...,w...] = ρ[1]
            end
        end
    end
    return rdm_
end

function rdm_para(ψ_input,ϕ_input,rm_inds,DP,P;kwargs...)
    """
    Rather than just calculating the reduced density matrix of |ψ><ψ|, this
    allows the reduced density matrix of |ψ><ϕ|.
    """

    (;Ns,N_L,N_R,symmetry_subspace) = P
    (;q,s,N,qtot) = DP
    rdm_block_sparse_bool = get(kwargs,:rdm_block_sparse_bool,true)
    para_bool = get(kwargs,:para_bool,true)
    if para_bool
        lk = ReentrantLock()
    end
    
    ψ = deepcopy(ψ_input)
    ϕ = deepcopy(ϕ_input)

    ###This is needed for doing the greens function calculation where ψ and ϕ
    ##have different QNs
    if !rdm_block_sparse_bool
        s = removeqns_mod(s)
    end


    N_inds = length(rm_inds)
    left_bath_bool,right_bath_bool = N_L>0,N_R>0
    d = NDTensors.dim(s[1])
    ϕdag = dag(ϕ)
    ITensors.prime!(linkinds, ϕdag)
    rdm_ = ITensor(dag(s[rm_inds]),s[rm_inds]')
    ρl =  1
    ρr =  ϕdag[N]*ψ[N]

    ##Trace out left bath
    if left_bath_bool
        ρl =  ϕdag[1]*ψ[1]
        left_inds = qtot[2:rm_inds[1]-1]
        for k in left_inds
            ρl = ρl* ϕdag[k]
            ρl = ρl* ψ[k]
        end
    end

    ##Trace out right bath
    if right_bath_bool
        right_inds = qtot[rm_inds[end]+1:end-1]
        for k in reverse(right_inds)   
            ρr = ρr* ϕdag[k]
            ρr = ρr* ψ[k]
        end
    end

    if para_bool 
        @threads for i=0:(d^(2*N_inds) - 1)  
            #The first 2Ns (N_inds) indices are taken as s[q] and the last 2Ns (N_inds) indices are taken as s[q]'.
            ##I'm deliberately contracting ψdag[j] and ψ[j] with ρ separately to prevent creating a tensor of size
            ##χ^4 with χ being the local bond dimension. 
            ##The largest tensor created is of size \chi^2*d where d is the site dimension (2).
            v = Vector{Any}(undef,N_inds)
            w = Vector{Any}(undef,N_inds)
            
            #creating a ditstring (bitstring of base d), representing the indices of the density matrix.
            #x gives the indices of s[q], y gives the indices of s[q]'.
            ditstring = zeros(Int,2*N_inds)
            dit = reverse(digits(i,base=d))
            ditstring[(end-(length(dit))+1):end] = dit
            x = ditstring[1:N_inds] .+1
            y = ditstring[N_inds+1:end] .+1

            if rdm_block_sparse_bool 
                matrix_element_bool = QN_matching(x,y,d,symmetry_subspace)
            else
                matrix_element_bool = true 
            end

            if matrix_element_bool
                local ρ = copy(ρl)
                lock(lk) do 
                    b = 0
                    for k in rm_inds
                        b += 1
                        C1 = ψ[k]*onehot(dag(s[k])=>x[b])
                        C2 = ϕdag[k]*onehot(s[k]=>y[b]) 
                        ρ = ρ*C1
                        ρ = ρ*C2
                        v[b] = dag(s[k]) => x[b]
                        w[b] = s[k]' => y[b] 
                    end
                    ρ = ρ*ρr
                    rdm_[v...,w...] = ρ[1]
                end
            end
        end
    else
        for i=0:(d^(2*N_inds) - 1)  
            #The first 2Ns (N_inds) indices are taken as s[q] and the last 2Ns (N_inds) indices are taken as s[q]'.
            ##I'm deliberately contracting ψdag[j] and ψ[j] with ρ separately to prevent creating a tensor of size
            ##χ^4 with χ being the local bond dimension. 
            ##The largest tensor created is of size \chi^2*d where d is the site dimension (2).
            v = Vector{Any}(undef,N_inds)
            w = Vector{Any}(undef,N_inds)
            
            #creating a ditstring (bitstring of base d), representing the indices of the density matrix.
            #x gives the indices of s[q], y gives the indices of s[q]'.
            ditstring = zeros(Int,2*N_inds)
            dit = reverse(digits(i,base=d))
            ditstring[(end-(length(dit))+1):end] = dit
            x = ditstring[1:N_inds] .+1
            y = ditstring[N_inds+1:end] .+1

            if rdm_block_sparse_bool 
                matrix_element_bool = QN_matching(x,y,d,symmetry_subspace)
            else
                matrix_element_bool = true 
            end

            if matrix_element_bool
                local ρ = copy(ρl)
                b = 0
                for k in rm_inds
                    b += 1
                    C1 = ψ[k]*onehot(dag(s[k])=>x[b])
                    C2 = ϕdag[k]*onehot(s[k]=>y[b]) 
                    ρ = ρ*C1
                    ρ = ρ*C2
                    v[b] = dag(s[k]) => x[b]
                    w[b] = s[k]' => y[b] 
                end
                ρ = ρ*ρr

                rdm_[v...,w...] = ρ[1]
            end
        end
    end
    return rdm_
end


function vectorise_ρ(ψ_input,P,DP;kwargs...)
    """
    In order for the rdm to be valid here, the ordering of qS and qA must be separated. 
    """
    (;sys_mode_type,Ns,ordering_choice,using_system_ancilla) = P
    (;s,q,qS,qA,c,cdag) = DP
    ψ = deepcopy(ψ_input)

    use_spin_operators = get(kwargs,:use_spin_operators,false)
    if sys_mode_type == "S=1/2"
        use_spin_operators = true
    end

    if !use_spin_operators
        if sys_mode_type == "Fermion" && ordering_choice == "interleaved"
            ##swap phases.
            anc_phase_gates =  ancilla_phase_gate_swap(qA,true,DP,P)
            ψ = apply_gates_to_ψ(ψ,anc_phase_gates)
        elseif sys_mode_type == "Electron"
            println("system rdm extraction using fermionic swaps with fermionic corrections 
            rather than qubit operators with no corrections is not implemented for
            Electron site type")
        end
    end

    if ordering_choice == "interleaved" && using_system_ancilla
        ###Applies fermionic swap gates to change the order from interleaved to separated
        ψ = system_swaps(ψ,q[1],DP,P;kwargs...)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end


    #Creating the evolved system density matrix 
    rm_inds = qS
    ρ = rdm_para(ψ,rm_inds,DP,P)

    s = removeqns(s)
    ρ = removeqns(ρ)
    #combined all system legs together
    Cs = combiner(reverse(s[qS]))
    ρ = ρ*dag(Cs)
    ρ = ρ*Cs'
    ρ = Array(ρ,inds(ρ))
    ρvec = Vector{ComplexF64}(undef,length(ρ))
    d = size(ρ)[1]
    for i =1:d
        for j = 1:d
            ρvec[(i-1)*d + j] = ρ[j,i]
        end
    end  
    return ρvec
end

function vectorise_ρ(ket_input,bra_input,P,DP;kwargs...)
    """
    In order for the rdm to be valid here, the ordering of qS and qA must be separated. 
    Rather than just calculating the reduced density matrix of |ψ><ψ|, this allows the reduced density matrix of |ψ><ϕ|.

    """
    (;sys_mode_type,Ns,ordering_choice,using_system_ancilla) = P
    (;s,q,qS,qA,c,cdag) = DP
    rdm_block_sparse_bool = get(kwargs,:rdm_block_sparse_bool,true)
    ψ_ket = deepcopy(ket_input)
    ϕ_bra = deepcopy(bra_input)

    use_spin_operators = get(kwargs,:use_spin_operators,false)

    if sys_mode_type == "S=1/2"
        use_spin_operators = true
    end

    if !use_spin_operators
        if sys_mode_type == "Fermion" && ordering_choice == "interleaved"
            ##swap phases.
            anc_phase_gates =  ancilla_phase_gate_swap(qA,true,DP,P)
            ψ_ket = apply_gates_to_ψ(ψ_ket,anc_phase_gates)
            ϕ_bra = apply_gates_to_ψ(ϕ_bra,anc_phase_gates)
        elseif sys_mode_type == "Electron"
            println("system rdm extraction using fermionic swaps with fermionic corrections 
            rather than qubit operators with no corrections is not implemented for
            Electron site type")
        end
    end

    if ordering_choice == "interleaved" && using_system_ancilla
        ###Applies fermionic swap gates to change the order from interleaved to separated
        ψ_ket = system_swaps(ψ_ket,q[1],DP,P;kwargs...)
        ϕ_bra = system_swaps(ϕ_bra,q[1],DP,P;kwargs...)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end


    #Creating the evolved system density matrix 
    rm_inds = qS
    if !rdm_block_sparse_bool
        ψ_ket = removeqns_mod(ψ_ket)
        ϕ_bra = removeqns_mod(ϕ_bra)
    end
    ρ = rdm_para(ψ_ket,ϕ_bra,rm_inds,DP,P;kwargs...)

    s = removeqns(s)
    ρ = removeqns(ρ)
    #combined all system legs together
    Cs = combiner(reverse(s[qS]))

    ρ = ρ*dag(Cs)
    ρ = ρ*Cs'
    ρ = Array(ρ,inds(ρ))
    ρvec = Vector{ComplexF64}(undef,length(ρ))
    d = size(ρ)[1]
    for i =1:d
        for j = 1:d
            ρvec[(i-1)*d + j] = ρ[j,i]
        end
    end  
    return ρvec
end

function choi_isomorphism(ρ,qA,qS,cutoff,P,DP)
    (;Ns) = P
    (;s) = DP
    d = NDTensors.dim(s[1])^Ns
    s = removeqns_mod(s)
    ρ = removeqns_mod(ρ)
    Cs = combiner(reverse(s[qS])) # Combiner tensor for merging system legs into a fat index
    Ca = combiner(reverse(s[qA])) # Combiner tensor for merging ancilla legs into a fat index
    ρΛ = ρ*dag(Cs)*Cs'*dag(Ca)*Ca'# Merge physical legs to form a density matrix

    Css = combiner([inds(Cs)[1]',dag(inds(Cs)[1])])
    Caa = combiner([inds(Ca)[1]',dag(inds(Ca)[1])])
    Csa = combiner([inds(Cs)[1],inds(Ca)[1]])
    ρmat = ρΛ*dag(Csa)*Csa'
    ρmat = Matrix(ρmat,inds(ρmat));
    
    # message,bool = ρ_test(ρmat,cutoff)
    # if bool
    #     error(message)
    # end
    Λmat = d*ρΛ*Css*Caa
    Λmat = conj(Matrix(Λmat,inds(Λmat)));
    return ρmat,Λmat
end

























"""
MPO Map contraction and system extraction, keeping an interleaved ordering
for efficiency
-----------------------------------------------------------------------------------
"""


function switch_linkind_direction(ρ)
    N = length(ρ)
    ρ_copy = MPO(N)

    δs = [delta(dag(lind),dag(lind)') for lind in linkinds(ρ)]
    for i =1:N
        if i == 1
            ρ_copy[i] = ρ[i]*δs[i]
        elseif i == N
            ρ_copy[i] = ρ[i]*dag(δs[i-1])
        else
            ρ_copy[i] = ρ[i]*dag(δs[i-1])*δs[i]
        end
    end
    return ρ_copy
end
    
    
function apply_PH_to_MPO_with_QN(ρ,P,DP)

    """
    NOTE: This function is dodgy, it gives a density operator 
    with trace -1, which is removed by dividing by the trace.
    """


    (;Ns) = P
    (;q,c,cdag,s,sys_cre_op,sys_ann_op) = DP

    @assert(typeof(ρ)==MPO)
    N = length(ρ)
    inds = q[1:Ns]
    ρ_PH = MPO(N)

    for (i,n) in enumerate(inds)
        PH_gate = cdag[n] +c[n]#-((-1)^(Ns))*c[n]
        fac = N-i
        if isodd(fac)
            Z = op("F",s,n)
            PH_gate = apply(Z,PH_gate)
        end  

        
        ρi = ρ[i]
        ρi = apply(PH_gate,ρi)
        ρi = apply(ρi,PH_gate)
        ρ_PH[i] = ρi

    end
    ρ_PH = ρ_PH/tr(ρ_PH)
    ρ_PH = orthogonalize(ρ_PH,1)
    return ρ_PH
end

function efficient_map_contraction(swapping_cutoff,ρ_input,Λ,PH_bool,P,DP)

"""
    To perform the contraction of ρ with Λ, any
    tracing must be contiguous. The naive method is to use swap gates
    to go from an interleaved system-ancilla ordering to a separated one, such 
    that qS=q[1:Ns],qA[Ns+1:2*Ns]. However, this leads to a bond dimension χ=>2^Ns
    between sites qS[end],qA[1] which limits this method to small systems. This method
    instead performs the contraction of each ancilla mode with the input modes sequentially, 
    performing swaps to move each pair to the end of the chain. 
    
    This function assumes an interleaved ordering for the input Λ.
    
    
    
    Step 1: Forming combined MPO of map and input state. To do
    the contraction with minimal swaps, we input ρ in reverse ordering
    such that Λ_ext[2*Ns+1] = ρ[Ns]. The siteinds of ρ are also 
    changed to a new set of dummy indices so there are no unintentional contractions
    with the Λ modes.
    
    
    Step 2: Perform swapping for indth entangled pair
    We first move the indth entangled system-ancilla pair through
    the sites formed from previous contractions, such that 
    the indth ancilla and indth mode of the input are contiguous. The 
    indth entangled pair are at sites 2*ind-1:2*ind.
    For the indth mode there are Ns-ind of these 
    (for ind=Ns, there are no previous contractions,
    for ind=Ns-1 there is one previous contraction etc, 
    and Ns-ind = length(ancilla_site: ancilla_site + Ns+ind-1)).
    
        
    Step 3: Perform contraction with indth input mode.
        
    After swapping procedure:
    Entangled pairs for sites
    1:2(ind-1)

    Previously contracted modes for sites 
    2*ind-1:Ns+ind-2 =(length Ns-ind),

    (ind)th entangled system pair at sites 
    Ns+ind-1:Ns+ind

    Remaining uncontracted
    input modes at sites: 
    Ns+ind+1:Ns+2ind (length ind). 

    We now contract the (ind)th entangled system
    pair with the adjacent system input mode (site Ns+ind+1).
    To do this, we must change the site inds of the input mode from 
    the dummy index to the ancilla index its contracting with.

    Step 4: Form new, reduced MPO.
    
    Step 5: iterate step 2:4 for ind in reverse(1:Ns)
    Step 6: After all contractions, the resulting MPO will have siteinds s[q[1:Ns]],
    but the true sites are s[q[Ns:1]]. This is because we flip the input to do the contractions,
    and after the ith contraction of the indth site of the input,  we swap the T tensor to the left
    such that it becomes the ith site, but it represents the indth site. To account for this, we create
    a new MPO that's the reversal of the output, then re label the site inds to give the correct ordering.
    This means the output is in the same ordering as the input.
    """



    (;Ns,sys_mode_type,tdvp_cutoff,symmetry_subspace,N_L) = P
    (;q,qA,qS,qB_L,s) = DP

    @assert(qA == q[2:2:2*Ns])
    @assert(qS == q[1:2:2*Ns])
    @assert(length(Λ)==2*Ns)
    @assert(length(ρ_input)==Ns)

    ρ = deepcopy(ρ_input)
    if PH_bool == true
        #PH_gates = particle_hole_transform_on_system(DP,P)
        #ρ = apply_gates_to_ρ(ρ,PH_gates,true)
        ρ = apply_PH_to_MPO_with_QN(ρ,P,DP)
    end

    ##creating a set of dummy indices
    if symmetry_subspace == "Number conserving"
        s_new =siteinds(sys_mode_type,Ns,conserve_qns=true)
    else
        s_new = siteinds(sys_mode_type,Ns)
    end

    s_new = [settags(mode,sys_mode_type*",Site_new,n="*string(q[i])) for (i,mode) in enumerate(s_new)]
    s_old = s[q[1:Ns]]
    ρ = conj(dag(change_inds(ρ,s_old,s_new)))
    
    ρ_flipped = MPO(Ns)
    [ρ_flipped[i] = ρ[ind] for (i,ind) in enumerate(reverse(1:Ns))]
    ρ = ρ_flipped
    
    for i =1:Int(floor(Ns/2))
        ind = Ns-i+1
        δi = delta(s_new[ind],dag(s_new[i])'')

        ρ[i] = ρ[i]*dag(δi)*δi'
        δind = delta(dag(s_new[ind])'',s_new[i])
        ρ[ind] = ρ[ind]*dag(δind)*δind'

    end

    ρ = replaceprime(ρ,2=>0)
    ρ = replaceprime(ρ,3=>1)
    ρ = ρ/tr(ρ)
    
    
    
    Λ_ext = MPO(3*Ns)
    Λ_ext[1:2*Ns] = Λ
    for (i,index) in enumerate(reverse(1:Ns))
        Λ_ext[2*Ns+ i] = ρ[i]
    end

    for (i,ind) in enumerate(reverse(1:Ns))
        L = length(Λ_ext)
        ancilla_site = 2*ind
        
        for j in ancilla_site: ancilla_site + Ns-ind-1

            orthogonalize!(Λ_ext,j)
            swap1 = swap_gate_new(Λ_ext[j],Λ_ext[j+1])   
            wf = Λ_ext[j]*Λ_ext[j+1]
            wf = apply(swap1,wf)
            wf = apply(wf,swap1)


            left_inds = uniqueinds(Λ_ext[j],Λ_ext[j+1])
            right_inds = uniqueinds(Λ_ext[j+1],Λ_ext[j])
            U,S,V = svd(wf,left_inds,cutoff=swapping_cutoff)
            @assert hasinds(U,left_inds)
            @assert hasinds(V,right_inds)
            Λ_ext[j] = U
            Λ_ext[j+1] = S*V
        

            orthogonalize!(Λ_ext,j-1)
            swap2 =  swap_gate_new(Λ_ext[j-1],Λ_ext[j])  #swap_gate(2*N_L+j-1,2*N_L+j,DP,P)
            wf = Λ_ext[j-1]*Λ_ext[j]
            wf = apply(swap2,wf)
            wf = apply(wf,swap2)

            left_inds = uniqueinds(Λ_ext[j-1],Λ_ext[j])
            right_inds = uniqueinds(Λ_ext[j],Λ_ext[j-1])
            U,S,V = svd(wf,left_inds,cutoff=swapping_cutoff)
        
            @assert hasinds(U,left_inds)
            @assert hasinds(V,right_inds)
            Λ_ext[j-1] = U
            Λ_ext[j] = S*V

        end
        
        ##After swapping, modes are in the following order
        uncontracted_pair_sites = 1:2(ind-1)
        previously_contracted_output_sites = 2*ind-1:Ns+ind-2
        indth_entangled_pair_sites = Ns+ind-1:Ns+ind
        uncontracted_input_sites = Ns+ind+1:Ns+2*ind
        
        ##Switch dummy site inds to ancilla site inds to perform contraction.
        sys_site = Ns+ind-1
        input_siteind = s_new[i] 
        anc_siteind = s[qA[end]-i+1] ## Starts at the qA[end] and decreases by 1 each iteration 
        
        δ1 = delta(dag(anc_siteind)',input_siteind')
        δ2 = delta(anc_siteind,dag(input_siteind))
        input_site = Λ_ext[sys_site+2]*δ1*δ2
        
        
        ##Perform contraction of entangled pair with input site
        T = Λ_ext[sys_site]*Λ_ext[sys_site+1]*input_site
        
    
        ##Form new MPO, with a length smaller by 2
        reduced_MPO = MPO(L-2)
        [reduced_MPO[i] = Λ_ext[i] for i in uncontracted_pair_sites]
        [reduced_MPO[i] = Λ_ext[i] for i in previously_contracted_output_sites]
        reduced_MPO[sys_site] = T
        [reduced_MPO[i-2] = Λ_ext[i] for i in uncontracted_input_sites[2:end]]

        ###iterate procedure
        Λ_ext = reduced_MPO

    end
    
    ρ = MPO(Ns)
    ##These will be ordered from q[1:Ns], but should be ordered q[Ns:1] 
    Λ_ext_inds = [noprime(siteinds(Λ_ext)[i][1]) for i in 1:length(siteinds(Λ_ext))] 
    [ρ[i] = Λ_ext[ind] for (i,ind) in enumerate(reverse(1:Ns))]

    for i =1:Int(floor(Ns/2))
        ind = Ns-i+1
        δi = delta(Λ_ext_inds[ind],dag(Λ_ext_inds[i])'')

        ρ[i] = ρ[i]*δi*dag(δi')
        δind = delta(dag(Λ_ext_inds[ind])'',Λ_ext_inds[i])
        ρ[ind] = ρ[ind]*δind*dag(δind')

    end

    ρ = replaceprime(ρ,2=>0)
    ρ = replaceprime(ρ,3=>1)
    ρ = ρ/tr(ρ)

    return ρ
end     
    
    





function efficient_system_state_extraction(ψ,P,DP)
    """
    The system state can be extracting by tracing out the baths and ancillas.
    As the ancillas are not contiguous we trace them out sequentially by 
    contracting them with normalized identity matrices.
    """
    
    
    (;Ns) = P
    (;s,qS,q) = DP


    ρ = op_mpo(s[q[1:Ns]], "Id", 1)
    ρ = ρ/tr(ρ)

    Λ_MPO = extract_map_as_MPO(ψ,false,P,DP)
    ρ_sys = efficient_map_contraction(1e-12,ρ,Λ_MPO,false,P,DP)

    ###Change indices into a contiguous form
    s_old = [noprime(siteinds(ρ_sys)[i][1]) for i in 1:length(siteinds(ρ_sys))]
    s_new = s[q[1:Ns]]
    ρ_sys = change_inds(ρ_sys,s_old,s_new)
    return ρ_sys
end

function swap_gate_new(ρ1,ρ2)

    """
    Applies a swap gate to MPO/MPS elements
    ρ1 and ρ2
    """
    s1 = noprime(inds(ρ1)[1])
    s2 = noprime(inds(ρ2)[1])
    
    sitetype1 = string(tags(s1)[1])
    sitetype2 = string(tags(s2)[1])
    
    if sitetype1 != sitetype2
        error("sitetypes for the swap gate aren't the same. The functionality for this
        hasn't been made.")
    end
    if sitetype1 == "Fermion"
        ann_op = "C"
        cre_op = "Cdag"
    end

    cdag1 = op(cre_op,s1)
    cdag2 = op(cre_op,s2)
    c1 = op(ann_op,s1)
    c2 = op(ann_op,s2)
    
    swap =  cdag1*c2+cdag2*c1
    N1 = prime(cdag1)*c1
    N2 = prime(cdag2)*c2
    N1_dag = prime(c1)*cdag1
    N2_dag = prime(c2)*cdag2

    unprime_ind(inds(N1)[1],N1)
    unprime_ind(inds(N2)[1],N2)
    unprime_ind(inds(N1_dag)[1],N1_dag)
    unprime_ind(inds(N2_dag)[1],N2_dag)  

    N = N1*N2  
    N_dag = N1_dag*N2_dag
    swap = swap + N + N_dag

    if sitetype1 == "Fermion"
        Rinds = inds(N,plev=0)
        Linds = Rinds'
        fermi_fac=  exp(-im*π*N,Linds,Rinds)
        swap = swap*prime(fermi_fac)
        unprime_ind(inds(swap)[3],swap)
        unprime_ind(inds(swap)[4],swap)
    end

    return swap
end














"""
Map contraction and system extraction, using a separated system ancilla
ordering. This is a poor implementation as the ancilla modes 
are separated from their entangled system modes, causing 
a rainbow state and a large bond dimension.
---------------------------------------------------------------------------------------
"""

function contract_ρ_with_ρΛ(ρ,Λ,DP,P)
    (;Ns,ordering_choice) = P
    (;q,qA,qS,s) = DP

    """
    ρΛ must have a separated mode ordering between the 
    system and ancilla modes. This function assumes 
    the Ns+1:2*Ns modes of ρΛ are the ancilla modes,
    and modes of ρ are system modes.We want the output 
    to have system legs like the input, but
    to do the contraction ρ must contract with the ancilla legs of Λ,
    so we switch the the labels to do this.
    """
    
    ρ_ev = deepcopy(ρ)
    if qS != q[1:Ns]
        qS = q[1:Ns]
        qA = q[Ns+1:2*Ns]
    end
    
    ρ = dag(change_inds(ρ_ev,s[qS],s[qA]))
    
    R = Λ[end]*ρ[end]
    for i in reverse(1:Ns-1)
        R *= Λ[i+Ns]
        R *= ρ[i]
    end
    R *= Λ[Ns]
    ρ_ev[Ns] = R
    for i=1:Ns-1
        ρ_ev[i] = Λ[i]
    end
    ρ_ev = orthogonalize(ρ_ev,1)
    ρ_ev = ρ_ev/tr(ρ_ev)
    return ρ_ev
end




function extract_system_state_as_MPO_deprecated(ψ,P,DP)



    (;Ns,sys_mode_type,ordering_choice) = P
    (;q,qS,qA,N) = DP


    if sys_mode_type == "Fermion" && ordering_choice == "interleaved"

        anc_phase_gates =  ancilla_phase_gate_swap(qA,true,DP,P)
        for i = 1:length(anc_phase_gates)
            ψ = apply(anc_phase_gates[i],ψ)
        end


        ###Applies fermionic swap gates to change the order from interleaved to separated
        ψ = system_swaps(ψ,q[1],DP,P)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end

    #Form MPO from MPS and trace out baths
    rm_inds = qS
    @time ρ_sys = partial_tr_v2(ψ,rm_inds,DP,P)

    orthogonalize!(ρ_sys,1)
    if Ns > 1
        ρ_sys = ρ_sys/tr(ρ_sys)
    else
        ρ_sys = ρ_sys/(tr(ρ_sys)[1])
    end
    return ρ_sys
end



function comparison_between_non_MPO_vs_MPO_map_extraction(P,DP)

    (;Ns,sys_mode_type,ordering_choice) = P
    (;q,qS,qA,N) = DP
    
    ψ = initialise_psi(P,DP)
    ψ_correct= initialise_psi_old(P,DP)

    println("initialise_psi works up to error"*string(norm(ψ-ψ_correct)))
    orthogonalize!(ψ,DP.N)
    ψ[DP.N]= ψ[DP.N]/norm(ψ)
    orthogonalize!(ψ_correct,DP.N)
    ψ_correct[DP.N]= ψ_correct[DP.N]/norm(ψ_correct)
    
    """
    Applying corrective phase gates
    """


    if sys_mode_type == "Fermion"
        if ordering_choice == "interleaved"
            ##swap phases.
            gates =  ancilla_phase_gate_swap(qA,true,DP,P)
            ψ = apply_gates_to_ψ(ψ,gates)
            ψ_correct = apply_gates_to_ψ(ψ_correct,gates)
        end

        ##Particle hole phases.
        if isodd(Ns)
            gates =  ancilla_phase_gate_PH(qA,true,DP,P)
            ψ = apply_gates_to_ψ(ψ,gates)
            ψ_correct = apply_gates_to_ψ(ψ_correct,gates)
        end
    end


    """
    Putting MPS into separated ordering between ancillas and system modes.
    """

    if ordering_choice == "interleaved"
        ###Applies fermionic swap gates to change the order from interleaved to separated.
        ψ = system_swaps(ψ,q[1],DP,P)
        ψ_correct = system_swaps(ψ_correct,q[1],DP,P)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end



    """
    Form MPO from MPS.
    """

    M = outer(ψ',ψ)

    """
    Trace out baths.
    """

    rm_inds = [qS ; qA]
    @time ρΛ_MPO = partial_tr(M,rm_inds,P)
    @time ρΛ_correct = rdm_para(ψ_correct,rm_inds,DP,P)
    ρ_sys = partial_tr(M,DP.qS,P)

    println("partial_tr works up to error "*string(norm(ρΛ_correct-MPO_to_Tensor(ρΛ_MPO))))

    """
    Apply particle hole transform
    """

    gates_MPO =  particle_hole_transform_MPO(qS,qA,DP,P)
    gates_correct = particle_hole_transform(qA,true,DP,P)
    @show(norm(gates_correct[1]-MPO_to_Tensor(gates_MPO[1])))

    for i =1:length(gates_MPO)
        println("particle_hole_transform_MPO for mode "*string(i)*" works up to error "*string(norm(gates_correct[i]-MPO_to_Tensor(gates_MPO[i]))))
    end



    ρΛ_MPO = apply_gates_to_ρ(ρΛ_MPO,gates_MPO,false)
    ρΛ_correct = apply_gates_to_ρ(ρΛ_correct,gates_correct,false)
    @show(norm(ρΛ_correct-MPO_to_Tensor(ρΛ_MPO)))

    println("particle_hole_transform_MPO works up to error "*string(norm(ρΛ_correct-MPO_to_Tensor(ρΛ_MPO))))
    return ρΛ_MPO
end























"""
Helper functions for map extraction
--------------------------------------------------------------------------------------------
"""


function expect_op(ρ,op,P)
    """
    ρ is the input density operator, op is the 
    operator we want to take the expectation value of.
    """
    (;Ns) = P

    ρ = deepcopy(ρ)
    ###Checking its the correct size and normalized
    #@assert(length(ρ) == Ns)
    @assert(abs(tr(ρ)-1)<1e-5)
    ρ = apply(op,ρ)
    return tr(ρ)
end

function change_inds(ρ,s_old,s_new)

    @assert(length(s_old) == length(s_new))
    ρ_new = MPO(s_new)
    for i =1:length(s_old)
        if s_old[i] != s_new[i]
            δ1 = delta(s_old[i],dag(s_new[i]))

            MPO_term_new = ρ[i]*δ1*dag(δ1)'
            ρ_new[i] = MPO_term_new

        else
            ρ_new[i] = ρ[i]
        end
    end

    return ρ_new
end



function particle_hole_transform_MPO(qS,qA,DP,P)
    (;Ns,N_L,sys_mode_type) = P
    (;q,c,cdag,s,sys_cre_op,sys_ann_op) = DP
    
    MPO_list = []
    
    s = removeqns(s)

    """
    Create JW string for PH on ith ancilla mode
    """
    JW_string = Vector{String}(undef,2*Ns)
    for (i,n) in enumerate(qA)
        
        JW_string[1:(n-2*N_L-1)] .= "F"
        JW_string[(n-2*N_L):end] .= "Id"
        gate_MPO = MPO(s[q],JW_string)
    
        gate = op(sys_cre_op,s,n)
        if sys_mode_type == "Fermion"
            gate +=  -((-1)^(Ns))*op(sys_ann_op,s,n)
        else
            gate += op(sys_ann_op,s,n)
        end 
        #@show(JW_string)
        #@show(gate)
        gate_MPO[n-2*N_L] = apply(gate,gate_MPO[n-2*N_L])
        orthogonalize!(gate_MPO,1)
        push!(MPO_list,gate_MPO)
    end

    return MPO_list
end

function particle_hole_transform_on_system(DP,P)
    (;Ns,N_L,sys_mode_type) = P
    (;q,c,cdag,s,sys_cre_op,sys_ann_op) = DP

    MPO_list = []
    
    inds = q[1:Ns]
    
    
    
    #  s = removeqns(s)

    """
    Create JW string for PH on ith ancilla mode
    """
    JW_string = Vector{String}(undef,Ns)
    for (i,n) in enumerate(inds)

        JW_string[1:(n-2*N_L-1)] .= "F"
        JW_string[(n-2*N_L):end] .= "Id"

        gate_MPO = MPO(s[inds],JW_string)

        gate = op(sys_cre_op,s,n)
        if sys_mode_type == "Fermion"
            gate +=  -((-1)^(Ns))*op(sys_ann_op,s,n)
        else
            gate += op(sys_ann_op,s,n)
        end 

        gate_MPO[n-2*N_L] = apply(gate,gate_MPO[n-2*N_L])
        orthogonalize!(gate_MPO,1)
        push!(MPO_list,gate_MPO)
    end
    return MPO_list
end




































"""
Corrections due to fermi properties.
-------------------------------------------------------------------------------
"""



function ancilla_phase_gate_swap(ind,ITensor_bool,DP,P)
    (;Ns,sys_mode_type) = P
    if ITensor_bool == false
        cdag,c = matrix_operators(2*Ns,P)
    else
        (;c,cdag) = DP
    end

    if sys_mode_type == "Electron" && !ITensor_bool
        println("Map extraction using matrices rather than Tensors is not implemented for
        Electron site type")
    end

    Uph_list = []
    for spin in 1:length(c[1])
        for i =1:Ns
            for j =1:(i-1)
            if ITensor_bool
                ##Old version used
                #x = prime(c[ind[i]][spin])*cdag[ind[i]][spin]*prime(c[ind[j]][spin])*cdag[ind[j]][spin]
                x = prime(cdag[ind[j]][spin])*c[ind[j]][spin]*prime(c[ind[i]][spin])*cdag[ind[i]][spin]
                unprime_ind(inds(x)[1],x)
                unprime_ind(inds(x)[3],x)  

                Rinds = inds(x,plev=0)
                Linds = Rinds'
                Uph = exp(-im*π*x,Linds,Rinds)
            else
                x = c[ind[i]][spin]*cdag[ind[i]][spin]*c[ind[j]][spin]*cdag[ind[j]][spin]
                Uph = exp(-im*π*x)
            end
            push!(Uph_list,Uph)
            end
        end
    end
    return Uph_list
end

function ancilla_phase_gate_PH(ind,ITensor_bool,DP,P)
    (;Ns,sys_mode_type) = P
    if ITensor_bool == false
        cdag,c = matrix_operators(2*Ns,P)
    else
        (;c,cdag,s) = DP
    end
    """
    Only apply when Ns is odd
    """
    if sys_mode_type == "Electron" && !ITensor_bool
        println("Map extraction using matrices rather than Tensors is not implemented for
        Electron site type")
    end

    # if sys_mode_type == "Electron"
    #     cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagup","Adagdn"]]) for n in 1:length(s)]
    #     c =  [ops(s, [(cre_op, n) for cre_op in ["Aup","Adn"]]) for n in 1:length(s)]
    # end

    Uph_list = []
    for spin in 1:length(c[1])
        for i =1:Ns
            if ITensor_bool
                x = prime(c[ind[i]][spin])*cdag[ind[i]][spin]
                unprime_ind(inds(x)[1],x) 
                Rinds = inds(x,plev=0)
                Linds = Rinds'
                Uph = exp(-im*π*x,Linds,Rinds)
            else
                x = c[ind[i]]*cdag[ind[i]]
                Uph = exp(-im*π*x)
            end
            push!(Uph_list,Uph)
        end
    end
    return Uph_list
end


function system_swaps(ψ_input,start,DP,P;kwargs...)
    (;Ns,sys_mode_type) = P
    (;s) = DP
    """
    This function takes a state where the system and ancilla modes are interleaved
    (with the first system mode at the start site) and swaps the ordering such that 
    indices start:start+Ns-1 are all the system modes and start+Ns:start+2Ns-1 are
    the ancilla modes.
    The index start+2(i-1) gives the site index of the ith system mode.

    """
    ψ = deepcopy(ψ_input)
    use_spin_operators = get(kwargs,:use_spin_operators,false)

    if sys_mode_type == "S=1/2"
        use_spin_operators = true
    end

    if sys_mode_type == "Electron" && !use_spin_operators    
        println("Fermionic swaps are not implemented for
        Electron site type")
    end

    for i=2:Ns
        for j = 1:(i-1)
            ind = start+2(i-1)-j
            ind1 = ind
            ind2 = ind+1
            swap = swap_gate(ind1,ind2,DP,P;use_spin_operators)  
            orthogonalize!(ψ,ind)
            wf = (ψ[ind] * ψ[ind+1]) * swap
            noprime!(wf)
            inds3 = uniqueinds(ψ[ind],ψ[ind+1])
            U,S,V = svd(wf,inds3,cutoff=0)
            ψ[ind] = U
            ψ[ind+1] = S*V
        end
    end
    return ψ
end

function swap_gate(i,j,DP,P;kwargs...)
    (;sys_mode_type) = P
    (;c,cdag,Id,s) = DP

    if sys_mode_type == "Electron"
        cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagup","Adagdn"]]) for n in 1:length(s)]
        c =  [ops(s, [(cre_op, n) for cre_op in ["Aup","Adn"]]) for n in 1:length(s)]
    end

    use_spin_operators = get(kwargs,:use_spin_operators,false)
    swap_gate_vec = Vector{ITensor}(undef,length(c[1]))

    for spin in 1:length(c[1])
        swap =  cdag[i][spin]*c[j][spin]+cdag[j][spin]*c[i][spin]
        N1 = prime(cdag[i][spin])*c[i][spin]
        N2 = prime(cdag[j][spin])*c[j][spin]
        N1_dag = prime(c[i][spin])*cdag[i][spin]
        N2_dag = prime(c[j][spin])*cdag[j][spin]

        unprime_ind(inds(N1)[1],N1)
        unprime_ind(inds(N2)[1],N2)
        unprime_ind(inds(N1_dag)[1],N1_dag)
        unprime_ind(inds(N2_dag)[1],N2_dag)  

        N = N1*N2  
        N_dag = N1_dag*N2_dag
        swap = swap + N + N_dag

        if (sys_mode_type == "Fermion" || sys_mode_type == "Electron") && !use_spin_operators
            Rinds = inds(N,plev=0)
            Linds = Rinds'
            fermi_fac=  exp(-im*π*N,Linds,Rinds)
            swap = swap*prime(fermi_fac)
            unprime_ind(inds(swap)[3],swap)
            unprime_ind(inds(swap)[4],swap)
        end
        swap_gate_vec[spin] = swap
    end

    ###If there are spin components on the site indices, 
    ##this loop combines all the swap gates for each spin component
    swap = swap_gate_vec[1]
    for i=2:length(swap_gate_vec)
        swap = prime(swap_gate_vec[i])*swap
        swap = replaceprime(swap,2=>1)
    end
    return swap
end












"""
Helper functions, mostly for fermions
-----------------------------------------------------------------------
"""


function MPO_to_Tensor(M)
    N = length(M)
    T = M[1]
    for i=2:N
        T *= M[i]
    end
    return T
end


function removeqns_mod(x)
    try 
        x = removeqns(x)
    catch
        if typeof(x) == MPO
            x = MPO([removeqns(x[i]) for i =1:length(x)])
        elseif typeof(x) == MPS
            x = MPS([removeqns(x[i]) for i =1:length(x)])
        end
    end
    return x
end

function map_check(ψf,ψi,Λmat,P,DP;kwargs...)
    ρf = vectorise_ρ(ψf,P,DP;kwargs...)
    ρi = vectorise_ρ(ψi,P,DP;kwargs...)
    return norm(ρf - Λmat*ρi)
end

function map_check2(ρ_list,Λ_list,ρi)
    n = length(ρ_list)
    diff = zeros(n)
    for i=1:n
        diff[i] = norm(ρ_list[i] - Λ_list[i]*ρi)
    end
    return diff
end



function unprime_ind(ind,x)
    int = 1
    j = setprime(ind,int)
    replaceind!(x,ind,j)
    return x
end

function unprime_string(T)

    int = 1
    indices = inds(x)
    l = Int(length(indices)/2)
    for i =1:l
        ind = indices[length(indices)-2*i+1]
        j = setprime(ind,int)
        replaceind!(x,ind,j)
    end
    return T
end


function JW_string(n,start,DP)
    (;F) = DP
    
    if n>start
        x = F[n-1]
        if n-1>start
            for i in reverse(start:n-2)
                x = F[i]*x
            end
        end
    else
        x=1
    end
    return x
end 

function Id_string(n1,n2,DP)
    """
    Gives an Id string from n1+1 to n2. Note that
    these operators don't need to be applied in a specific order as they
    all commute with everything. 
    """
    (;Id) = DP 
    x=1
    for i =n1+1:n2
        x = Id[i]*x
    end
    return x
end




function create_fermi_ann_op(site,subsys_inds,DP;kwargs...) 
    c = get(kwargs,:c,DP.c)
    cdag = get(kwargs,:cdag,DP.cdag)
    spin = get(kwargs,:spin,1)
    return JW_string(site, subsys_inds[1],DP) * c[site][spin] * Id_string(site, subsys_inds[end], DP) 
end

function create_fermi_cre_op(site,subsys_inds,DP;kwargs...)
    c = get(kwargs,:c,DP.c)
    cdag = get(kwargs,:cdag,DP.cdag)
    spin = get(kwargs,:spin,1)
    return JW_string(site, subsys_inds[1],DP) * cdag[site][spin] * Id_string(site, subsys_inds[end], DP) 
end




function apply_gates_to_ρ(ρ,gates,truncate_bool)

    vector_gates_bool = isa(gates, Vector)
    vector_ρ_bool =  isa(ρ, Vector)

    if vector_gates_bool+vector_ρ_bool == 2
        println("functionality of apply_gates_to_ρ not implemented")
    end
    if vector_ρ_bool == true
        ρ_copy = gates
        gates_copy = ρ
        ρ = ρ_copy
        gates = gates_copy
    end
        
    if isa(gates, Vector)
        for i=1:length(gates)    
            if typeof(gates[i]) == MPO
                if truncate_bool
                    ρ = apply(gates[i],ρ; cutoff = 1e-15)
                    ρ = apply(ρ,gates[i]; cutoff = 1e-15)
                else
                    ρ = apply(gates[i],ρ;alg="naive",truncate=false)
                    ρ = apply(ρ,gates[i];alg="naive",truncate=false)
                end
            else
                ρ = apply(gates[i],ρ)
                ρ = apply(ρ,gates[i])
            end
        end
    
    
    elseif typeof(gates) == MPO
        if truncate_bool
            ρ = apply(gates,ρ; cutoff = 1e-15)
            ρ = apply(ρ,gates; cutoff = 1e-15)
        else
            ρ = apply(gates,ρ;alg="naive",truncate=false)
            ρ = apply(ρ,gates;alg="naive",truncate=false)
        end
    else
        ρ = apply(gates,ρ)
        ρ = apply(ρ,gates)
    end
    return ρ
end


function apply_gates_to_ψ(ψ,gates)
    for i = 1:length(gates)
        ψ = apply(gates[i],ψ)
    end
    return ψ
end

function spin_operators(M)

    # Build sparse matrix version of basic spin (Pauli) operators :
    sp = spdiagm(2,2,1=>ones(1))
    sm = spdiagm(2,2,-1=>ones(1))
    sz = spdiagm(2,2,0=>[1;-1]);
    num = spdiagm(2,2,0=>[0;1])
    # Notice there are NO factors of (1/2) for spin-1/2 included here.

    # Construct spin operators for each spin in the full Hilbert space :
    Sz = Vector{Any}(undef, M)
    Sp = Vector{Any}(undef, M)
    Sm = Vector{Any}(undef, M)
    Num = Vector{Any}(undef,M)
    for m=1:M
        Sz[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sz),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Sp[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sp),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Sm[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),sm),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
        Num[m] = kronecker(kronecker(spdiagm(2^(m-1),2^(m-1),0=>ones(2^(m-1))),num),spdiagm(2^(M-m),2^(M-m),0=>ones(2^(M-m))));
    end
    return Sz,Sp,Sm,Num
end

function JW_string_mat(Sz,site,M;kwargs...)
    inds = get(kwargs,:inds,1:(site-1))

    Z = 1.0*Matrix(I, 2^M, 2^M)
    for i in inds
        Z = Z*Sz[i];
    end
    return Z
end

function matrix_operators(number_of_modes,P;kwargs...)
    (;sys_mode_type) = P
    Sz,Sp,Sm,_ = spin_operators(number_of_modes)
    cdag_mat = Vector{Any}(undef,number_of_modes)
    c_mat = Vector{Any}(undef,number_of_modes)

    if sys_mode_type == "Fermion" 
        for n=1:number_of_modes
            #Build JW_string
            Z = JW_string_mat(Sz,n,number_of_modes)
            cdag_mat[n] = Z*Sm[n]
            c_mat[n]  = Z*Sp[n]
        end
        return cdag_mat,c_mat
    elseif sys_mode_type == "Electron"
        ##For the electron sitetype, we need to follow the convention in ITensor,
        ##which orients the JW string to the left with the exception of inside each site, 
        ##where the basis is dn ⊗ up, but dn operators have a Z on the up space. 

        for n =1:number_of_modes
            if isodd(n)
                ###down spin, has Z's on all sites to the left of it,
                ##and also a Z to the site on its right as this represents
                ##the up spin of the same site
                Z = JW_string_mat(Sz,n,number_of_modes)
                Z = Z*Sz[n+1]
                cdag_mat[n] = Z*Sm[n]
                c_mat[n]  = Z*Sp[n]
            else
                ###up spin, has Z's on all sites to the left of it 
                ###except the site directly to its left, as this is the 
                ###down spin of the same site
                Z = JW_string_mat(Sz,n-1,number_of_modes)
                cdag_mat[n] = Z*Sm[n]
                c_mat[n]  = Z*Sp[n]
            end
        end
        return cdag_mat,c_mat
    else
        return Sm,Sp
    end
end  

















"""
Enrichment functions, based on the algorithm in https://arxiv.org/abs/2005.06104
"""



function enrich_generic3(ϕ, ψ⃗; P, kwargs...)
    (;Kr_cutoff) = P
    """
        Given spec from the eigen function, to extract its information use the 
        following functions:

        eigs(spec) returns the spectrum
        truncerror(spec) returns the truncation error
    """  

    normalise = get(kwargs,:normalise,true)

    Nₘₚₛ = length(ψ⃗) ##number of MPS

    @assert all(ψᵢ -> length(ψ⃗[1]) == length(ψᵢ), ψ⃗) ##check that all MPS inputs are of the same length

    N = length(ψ⃗[1]) 
    ψ⃗ = copy.(ψ⃗)

    ###Isn't this already a vector of MPS's?  
    ψ⃗ = convert.(MPS, ψ⃗)

    s = siteinds(ψ⃗[1])
    ##makes the orthogonality centre for each MPS to be at site N  
    ψ⃗ = orthogonalize.(ψ⃗, N)
    ϕ = orthogonalize!(ϕ, N)

    ##storage MPS
    phi = deepcopy(ϕ)

    ρϕ = prime(ϕ[N], s[N]) * dag(ϕ[N])
    ρ⃗ₙ = [prime(ψᵢ[N], s[N]) * dag(ψᵢ[N]) for ψᵢ in ψ⃗]
    ρₙ = sum(ρ⃗ₙ)

    """
    Is this needed?
    """
    ρₙ /=tr(ρₙ)

#   # Maximum theoretical link dimensions

    Cϕprev = ϕ[N]
    C⃗ₙ = last.(ψ⃗)


    for n in reverse(2:N)
        """
    In the paper they propose to do this step with no truncation. At the very
    least this cutoff should be a function parameter.
    """    

    left_inds = linkind(ϕ,n-1)

            #Diagonalize primary state ψ's density matrix    
    U,S,Vϕ,spec = svd(Cϕprev,left_inds; 
        lefttags = tags(linkind(ϕ, n - 1)),
        righttags = tags(linkind(ϕ, n - 1)))   
        
    x = ITensors.dim(inds(S)[1])

    @assert(x == ITensors.dim(linkind(ϕ, n - 1)))
    
    r = uniqueinds(Vϕ, S) # Indices of density matrix
    lϕ = commonind(S, Vϕ) # Inner link index from density matrix diagonalization


    # Compute the theoretical maximum bond dimension that the enriched state cannot exceed:
    abs_maxdim = bipart_maxdim(s,n - 1) - ITensors.dim(lϕ)
    # Compute the number of eigenvectors of ɸ's projected density matrix to retain:
    Kry_linkdim_vec = [ITensors.dim(linkind(ψᵢ, n - 1)) for ψᵢ in ψ⃗]


    ω_maxdim = min(sum(Kry_linkdim_vec),abs_maxdim)

    if ω_maxdim !== 0


        # Construct identity matrix
        ID = 1
        rdim = 1
        for iv in r
            IDv = ITensor(iv', dag(iv));
            rdim *= ITensors.dim(iv)
            for i in 1:ITensors.dim(iv)
            IDv[iv' => i, iv => i] = 1.0
            end      
            ID = ID*IDv
        end   


        P = ID - prime(Vϕ, r)*dag(Vϕ) # Projector on to null-space of ρψ   

        C = combiner(r) # Combiner for indices
        # Check that P is non-zero   
        if abs(tr(matrix(C'*P*dag(C)))) > 1e-10    


            Dp, Vp, spec_P = eigen(
                    P, r', r,
                    ishermitian=true,
                    tags="P space",
                    cutoff=1e-1,
                    maxdim=rdim-ITensors.dim(lϕ),             ###potentially wrong
                    kwargs...,
                )

            lp = commonind(Dp,Vp)

            ##constructing VpρₙVp
            VpρₙVp = Vp*ρₙ        
            VpρₙVp = VpρₙVp*dag(Vp')
            chkP = abs(tr(matrix(VpρₙVp))) ##chkP

        else
            chkP = 0    
        end
    else
        chkP = 0
    end

    if chkP >1e-15
        Dₙ, Vₙ, spec =eigen(VpρₙVp, lp', lp;
            ishermitian=true,
            tags=tags(linkind(ψ⃗[1], n - 1)),
            cutoff=Kr_cutoff,
            maxdim=ω_maxdim,            
            kwargs...,
        )

        Vₙ = Vp*Vₙ

        lₙ₋₁ = commonind(Dₙ, Vₙ)

        # Construct the direct sum isometry 
        V, lnew = directsum(Vϕ => lϕ, Vₙ => lₙ₋₁; tags = tags(linkind(ϕ, n - 1)))
    else
            V = Vϕ
            lnew = lϕ

    end
    @assert ITensors.dim(linkind(ϕ, n - 1)) - ITensors.dim(lϕ) <=0
    # Update the enriched state
    phi[n] = V


    # Compute the new density matrix for the ancillary states
    C⃗ₙ₋₁ = [ψ⃗[i][n - 1] * C⃗ₙ[i] * dag(V) for i in 1:Nₘₚₛ]   
    C⃗ₙ₋₁′ = [prime(Cₙ₋₁, (s[n - 1], lnew)) for Cₙ₋₁ in C⃗ₙ₋₁]    
    ρ⃗ₙ₋₁ = C⃗ₙ₋₁′ .* dag.(C⃗ₙ₋₁)
    ρₙ₋₁ = sum(ρ⃗ₙ₋₁)

    # compute the density matrix for the real state    
    Cϕ = ϕ[n - 1] * Cϕprev * dag(V)
    Cϕd = prime(Cϕ, (s[n - 1], lnew))
    ρϕ = Cϕd * dag(Cϕ) 


    Cϕprev = Cϕ
    C⃗ₙ = C⃗ₙ₋₁
    ρₙ = ρₙ₋₁

    end


    phi[1] = Cϕprev

    if normalise
        phi[1] = phi[1]/norm(phi)
    end
    
    return phi
end


function Krylov_states(ψ,P,DP;kwargs...)
    (;k1,τ_Krylov) = P
    (;s,H_MPO) = DP
    ishermitian = get(kwargs, :ishermitian, true)


    ##Create the first k Krylov states
    Id = MPO(s,"Id")
    Kry_op = 1
    if ishermitian
        try
            Kry_op = Id-im*τ_Krylov*H_MPO
        catch 
            Kry_op = H_MPO
        end
    else
        Kry_op = H_MPO
    end

    list = []
    term = copy(ψ)

    for i =1:k1-1
        term = noprime(Kry_op*term)
        term = term/norm(term)
        push!(list,term)
    end

    return list
end


function Krylov_linkdims(Krylov)
    """
Determining whether a Krylov state has the dimensions of lower Krylov states within it.
We have a list of vectors, and we want to see if the 1st vector has the highest entries for every entry.
I want the output to be a vector of length of linkdims, where each entry denotes which Krylov vector has the maximum dimension.
"""
    x = linkdims_(Krylov[1])
    dim1 = length(x)
    dim2 = length(Krylov)
    output = zeros(length(x))
    stuff = zeros(dim1,dim2)
    for i =1:dim1
        for j =1:dim2
            stuff[i,j] = linkdims_(Krylov[j])[i]
        end
        vec = stuff[i,:]
        term = Int(argmax(vec)) 
        output[i] = term
        if term != dim2
            test = vec[term] -vec[dim2]
            if test<=0
                output[i] = dim2
            end
        end
    end
    return output
end


function bipart_maxdim(s,n)

    left_maxdim = 2^n
    right_maxdim = 2^(length(s)-n)
    if left_maxdim==0
        left_maxdim = 2^63
    end
    if right_maxdim==0
        right_maxdim = 2^63
    end
    return min(left_maxdim,right_maxdim)
end;

    
function expand_(
    state::MPS,
    references;
    cutoff,
)

    n = length(state)
    state = orthogonalize(state, n)
    references = map(reference -> orthogonalize(reference, n), references)
    s = siteinds(state)
    for j in reverse(2:n)
    # SVD state[j] to compute basisⱼ
    linds = [s[j - 1]; linkinds(state, j - 1)]
    _, λⱼ, basisⱼ = svd(state[j], linds; righttags="bψ_$j,Link")
    rinds = uniqueinds(basisⱼ, λⱼ)
    # Make projectorⱼ
    idⱼ = prod(rinds) do r
        return adapt(unwrap_array_type(basisⱼ), denseblocks(δ(scalartype(state), r', dag(r))))
    end
    projectorⱼ = idⱼ - prime(basisⱼ, rinds) * dag(basisⱼ)
    # Sum reference density matrices
    ρⱼ = sum(reference -> prime(reference[j], rinds) * dag(reference[j]), references)
    ρⱼ /= tr(ρⱼ)
    # Apply projectorⱼ
    ρⱼ_projected = apply(apply(projectorⱼ, ρⱼ), projectorⱼ)
    expanded_basisⱼ = basisⱼ
    if norm(ρⱼ_projected) > 10^3 * eps(real(scalartype(state)))
        # Diagonalize projected density matrix ρⱼ_projected
        # to compute reference_basisⱼ, which spans part of right basis
        # of references which is orthogonal to right basis of state
        dⱼ, reference_basisⱼ = eigen(
        ρⱼ_projected; cutoff, ishermitian=true, righttags="bϕ_$j,Link"
        )
        state_indⱼ = only(commoninds(basisⱼ, λⱼ))
        reference_indⱼ = only(commoninds(reference_basisⱼ, dⱼ))
        expanded_basisⱼ, expanded_indⱼ = directsum(
        basisⱼ => state_indⱼ, reference_basisⱼ => reference_indⱼ
        )
    end
    # Shift ortho center one site left using dag(expanded_basisⱼ)
    # and replace tensor at site j with expanded_basisⱼ
    state[j - 1] = state[j - 1] * (state[j] * dag(expanded_basisⱼ))
    state[j] = expanded_basisⱼ
    for reference in references
        reference[j - 1] = reference[j - 1] * (reference[j] * dag(expanded_basisⱼ))
        reference[j] = expanded_basisⱼ
    end
    end
    return state
end
function growbond!(v::MPS, bond::Integer; increment::Integer=1)::Integer
    bond_index = commonind(v[bond], v[bond + 1])
    current_bonddim = ITensors.dim(bond_index)
    aux = Index(current_bonddim + increment; tags=tags(bond_index))
    v[bond] = v[bond] * delta(bond_index, aux)
    v[bond + 1] = v[bond + 1] * delta(bond_index, aux)
    return current_bonddim + increment
end
function growMPS!(v::MPS, dims::Vector{<:Integer})
    @assert length(dims) == length(v) - 1
    v_prev = copy(v)
    currentdims = linkdims(v)
    for (n, new_d, d) in zip(1:(length(v) - 1), dims, currentdims)
        growbond!(v, n; increment=new_d - d)
    end
    v_overlap = dot(v, v_prev)
    @debug "Overlap ⟨original|extended⟩: $v_overlap"
    return v_overlap
end
growMPS!(v::MPS, d::Integer) = growMPS!(v, fill(d, length(v) - 1))



















    ###Observer functions

    """
    Performs GC.gc() after every sweep
    """
    
    function perform_GC(; bond, half_sweep, kwargs...)    
    if bond==1 && half_sweep==2
        GC.gc()
        println("Doing garbage collection")
    end
end



"""
    Memory observer function from https://itensor.discourse.group/t/memory-usage-in-dmrg-julia/562/4
    """
    function measure_mem(; bond, half_sweep, state, kwargs...)    
        if bond==1 && half_sweep==2
            psi_size =  Base.format_bytes(Base.summarysize(state))
            println("|psi| = $psi_size")
        end
    end
    
    function current_time(; current_time, bond, half_sweep)
        if bond == 1 && half_sweep == 2
        return real(im*current_time)
        end
        return nothing
    end

    function measure_den(; state, bond, half_sweep,N_L)
        if bond == 1 && half_sweep == 2
        return expect(state, "n"; sites=2*N_L+1)
        end
        return nothing
    end;

    function measure_SIAM_double_occ(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
        s = siteinds(state)
        q = []
        for (i,ind) in enumerate(s)
            if hastags(ind,"SystemSite") == true
                push!(q,i)
            end
        end
        
        double_occ_op = op("N",s[q[1]])*op("N",s[q[3]])
        return inner(state,apply(double_occ_op,state))
        end
    return nothing
end;

function measure_SIAM_diag_elements(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
        s = siteinds(state)
        q = []
        for (i,ind) in enumerate(s)
            if hastags(ind,"SystemSite") == true
                push!(q,i)
            end
        end
        spin_up_op = op("N",s[q[1]])
        spin_dn_op = op("N",s[q[3]])
        spin_up_Id = op("Id",s[q[1]])
        spin_dn_Id = op("Id",s[q[3]])
        
        projector_11 = spin_up_op*spin_dn_op
        projector_00 = (spin_up_Id -spin_up_op)*(spin_dn_Id -spin_dn_op)
        projector_10 = op("N",s[q[1]])*(op("Id",s[q[3]]) - op("N",s[q[3]]))
        projector_01 = op("N",s[q[3]])*(op("Id",s[q[1]]) - op("N",s[q[1]]))
        diag_elements = zeros(4)
        diag_elements[1] = real(inner(state,apply(projector_00,state)))
        diag_elements[2] = real(inner(state,apply(projector_01,state)))
        diag_elements[3] = real(inner(state,apply(projector_10,state)))
        diag_elements[4] = real(inner(state,apply(projector_11,state)))

        return diag_elements
        end
    return nothing
end;

function measure_SIAM_diag_elements_spinful(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
        s = siteinds(state)
        q = []
        for (i,ind) in enumerate(s)
            if hastags(ind,"SystemSite") == true
                push!(q,i)
            end
        end
        spin_up_op = op("Nup",s[q[1]])
        spin_dn_op = op("Ndn",s[q[1]])
        Id = op("Id",s[q[1]])

        projector_11 = apply(spin_up_op,spin_dn_op)
        projector_00 = apply((Id -spin_up_op),(Id -spin_dn_op))
        projector_10 = apply(spin_dn_op,(Id  - spin_up_op))
        projector_01 =  apply(spin_up_op,(Id  - spin_dn_op))
        diag_elements = zeros(4)
        diag_elements[1] = real(inner(state,apply(projector_00,state)))
        diag_elements[2] = real(inner(state,apply(projector_01,state)))
        diag_elements[3] = real(inner(state,apply(projector_10,state)))
        diag_elements[4] = real(inner(state,apply(projector_11,state)))


        return diag_elements
        end
    return nothing
end;

function measure_SvN(; state, bond, half_sweep)
        if bond == 1 && half_sweep == 2
            return entanglement_entropy(state)
        end
        return nothing
    end

function measure_correlation_matrix(; state, bond, half_sweep)
    if bond==1 && half_sweep == 2
        return transpose(correlation_matrix(state,"Cdag","C"))
    end
    return nothing
end

function measure_spin_up_electron_correlation_matrix(; state, bond, half_sweep)
    if bond==1 && half_sweep == 2
        return transpose(correlation_matrix(state,"Cdagup","Cup"))
    end
    return nothing
end

function measure_spin_dn_electron_correlation_matrix(; state, bond, half_sweep)
    if bond==1 && half_sweep == 2
        return transpose(correlation_matrix(state,"Cdagdn","Cdn"))
    end
    return nothing
end

function measure_spin_correlation_matrix(; state, bond, half_sweep)
    if bond==1 && half_sweep == 2
        spin_sites = []
        for (i,ind) in enumerate(siteinds(state))
            if hastags(ind,"S=1/2") == true
                push!(spin_sites,i)
            end
        end
        return transpose(correlation_matrix(state,"S+","S-",sites=spin_sites))
    end
    return nothing
end


    function measure_pauli_x(; state, bond, half_sweep)
    if bond==1 && half_sweep == 2
        spin_sites = []
        for (i,ind) in enumerate(siteinds(state))
            if hastags(ind,"S=1/2") == true
                push!(spin_sites,i)
            end
        end
        return expect(state,"Sx",sites=spin_sites)
    end
end

function measure_pauli_y(; state, bond, half_sweep)
    if bond==1 && half_sweep == 2
        spin_sites = []
        for (i,ind) in enumerate(siteinds(state))
            if hastags(ind,"S=1/2") == true
                push!(spin_sites,i)
            end
        end
        return expect(state,"Sy",sites=spin_sites)
    end
end

function measure_pauli_z(; state, bond, half_sweep)
    if bond==1 && half_sweep == 2
        spin_sites = []
        for (i,ind) in enumerate(siteinds(state))
            if hastags(ind,"S=1/2") == true
                push!(spin_sites,i)
            end
        end
        return expect(state,"Sz",sites=spin_sites)
    end
end

function measure_den_SF(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
        s = siteinds(state)
        left_vac = leftvacuum(s)
        q = []
        for (i,ind) in enumerate(siteinds(state))
            if hastags(ind,"SystemSite") == true
                push!(q,i)
            end
        end
        @show(inner(left_vac,state))
        @show(abs(inner(left_vac,state)))
        den_vec = Vector{Any}(undef,length(q))
        #  state = state/inner(left_vac,state)

        [den_vec[i] = inner(left_vac,apply(op("N",s[q[i]]),state)) for i in 1:length(q)]
        return den_vec
    end
    return nothing
end;
function measure_norm(; state, bond, half_sweep)
    if bond == 1 && half_sweep == 2
        s = siteinds(state)
        left_vac = leftvacuum(s)
        return inner(left_vac,state)
    end
    return nothing
end;
    

function entanglement_entropy(ψ)
# Compute the von Neumann entanglement entropy across each bond of the MPS
    N = length(ψ)
    SvN = zeros(N)
    psi = ψ
    for b=1:N
        psi = orthogonalize(psi, b)
        if b==1
            U,S,V = svd(psi[b] , siteind(psi, b))
        else
            U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi, b)))
        end
        for n=1:ITensors.dim(S, 1)
            p = S[n,n]^2
            SvN[b] -= p * log2(p)
        end
    end
    return SvN
end

function boundary_test(ψ,tol,DP,P)
    (;N_L,N_R,bath_mode_type) = P
    (;N,ψ_init,qB_L,qB_R,sys_cre_op,sys_ann_op) = DP
    left_bath_bool = N_L>0
    right_bath_bool = N_R>0
    
    if bath_mode_type == "Electron"
        density_op = "Nup"
    else
        density_op = "n"
    end
    if left_bath_bool
        left_boundary_test = expect(ψ_init,density_op;sites=qB_L)[1] - expect(ψ,density_op,sites=qB_L)[1]
        left_bool = abs(left_boundary_test)>tol
        if left_bool
            @show(left_boundary_test)
        end
    else 
        left_bool = false
    end
    if right_bath_bool
        right_boundary_test = expect(ψ_init,density_op;sites=qB_R)[end]- expect(ψ,density_op;sites=qB_R)[end]
        right_bool = abs(right_boundary_test)>tol
        if right_bool
            @show(right_boundary_test)
        end
    else
        right_bool = false
    end
    return left_bool,right_bool
end

function leftvacuum(indices)
        
    true_vac = MPS(indices, ["0" for _ in indices])
    gates = reverse([
        op("Cdag", indices[i]) * op("Id", indices[i+1]) +
        op("F", indices[i]) * op("Cdag", indices[i+1]) for i = 1:2:length(indices)
    ])
    return apply(gates,true_vac)
end

