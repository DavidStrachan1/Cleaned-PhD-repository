

"""
From ITensor_init_general
"""

function initialise_psi_MPO(P,DP)
    (;Ns,N_L,ordering_choice,sys_mode_type,bath_mode_type,compute_maps_bool,
    init_occ) = P
    (;s,qtot,cdag,Id,q,qS,qA,qB_L,
    sys_cre_op) = DP    
    """
    Thermofield vacuum
    """
    occs = ["0" for n in qtot]
    if bath_mode_type == "Fermion"
        occs = [isodd(n) ? "1" : "0" for n in qtot]
        occs = [i in q ? "0" : occs[i] for i in eachindex(occs)]
    end
    therm = MPS(ComplexF64,s,occs)

    os = OpSum()
    #sys_ops = Vector{Any}(undef,Ns)
    #anc_ops = Vector{Any}(undef,Ns)
    if compute_maps_bool
        for i=1:Ns
            op1 = op(sys_cre_op,s,qS[i])
            op2 =  op(sys_cre_op,s,qA[i])
            ψ_sys = (1/√2)*apply(op1,therm;cutoff=1e-15)
            ψ_anc = (1/√2)*apply(op2,therm;cutoff=1e-15)
            therm = ψ_sys+ψ_anc
            
        end
    else
        for i=1:Ns
            op1 = op(sys_cre_op,s,qS[i])
            op2 =  op(sys_cre_op,s,qA[i])
            ψ_sys = √(init_occ)*apply(op1,therm;cutoff=1e-15)
            ψ_anc = √(1-init_occ)*apply(op2,therm;cutoff=1e-15)
            therm = ψ_sys+ψ_anc
        end
    end
    
    
    return therm#apply(os,therm;cutoff=1e-15)
end

function particle_hole_transform_as_single_MPO(qS,qA,DP,P)
    """
    Don't think this works.
    """
    (;Ns,sys_mode_type) = P
    (;q,c,cdag,s,sys_cre_op,sys_ann_op,qS,qA) = DP

    ### For fermions, subtract them if it's even, add them if they're odd

    """
    This function only works if the modes are separated between 
    system modes and ancilla modes
    """
    @assert(qS == q[1:Ns])
    @assert(qA == q[Ns+1:2*Ns])

    
    gates = Vector{ITensor}(undef,length(ind))
    
  
    
   
    JW_string = Vector{String}(undef,2*Ns)
    if iseven(Ns)
        JW_string[1:Ns] .= "Id"
    else
        JW_string[1:Ns] .= "F"
    end
    for (i,n) in enumerate(qA)
        fac = Ns-i

        if iseven(fac)
            JW_string[n] = "Id"
        else
            JW_string[n] = "F"
        end
    end


    gates_MPO = MPO(s[q],JW_string)

    for (i,n) in enumerate(ind)
        #os += sys_cre_op,n 
        gate = op(sys_cre_op,s,n)
        if sys_mode_type == "Fermion"
            gate +=  -((-1)^(Ns))*op(sys_ann_op,s,n)
        else
            gate += op(sys_ann_op,s,n)
        end 
        gates[i] = gate
        gates_MPO[n] = apply(gate,gates_MPO[n])
    end
    orthogonalize!(gates_MPO,1)
    return gates_MPO
end

function extract_map_as_MPO_old(ρ,DP,P)
    """
    The input is an ITensor so defeats the purpose of reducing size through MPOs.
    """
    (;sys_mode_type,Ns,ordering_choice) = P
    (;q,qB_L,s,qS,qA) = DP
    sys_modes = siteinds(sys_mode_type,2*P.Ns)
    sys_modes = [settags(mode,sys_mode_type*",Site_new,n="*string(i+qB_L[end])) for (i,mode) in enumerate(sys_modes)]
    s_dummy = sys_modes
    
    if ordering_choice == "interleaved"
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end
    qS_dummy = qS .- DP.qB_L[end]
    qA_dummy = qA .- DP.qB_L[end]
    for i =1:Ns
        δ1 = delta(DP.s[qS[i]]',s_dummy[qA_dummy[i]]')
        δ2 = delta(DP.s[qA[i]]',s_dummy[qA_dummy[i]])
        δ3 = delta(DP.s[qA[i]],s_dummy[qS_dummy[i]])#         
        δ4 = delta(DP.s[qS[i]],s_dummy[qS_dummy[i]]')
        ρ = δ1*ρ
        ρ = δ2*ρ
        ρ = δ3*ρ
        ρ = δ4*ρ
    end
    for i =1:Ns
        δ1 = delta(DP.s[qS[i]],s_dummy[qS_dummy[i]])
        δ2 = delta(DP.s[qS[i]]',s_dummy[qS_dummy[i]]')
        δ3 = delta(DP.s[qA[i]],s_dummy[qA_dummy[i]])
        δ4 = delta(DP.s[qA[i]]',s_dummy[qA_dummy[i]]')
        ρ = ρ*δ1*δ2*δ3*δ4
    end
    Λ_MPO = 2^(Ns)*MPO(ρ,s[q])
    return Λ_MPO,ρ
end

function NESS_fn_MPO(map_type,ψ,P,DP)
    (;Ns,sys_mode_type,ordering_choice) = P
    (;s,q,qS,qA) = DP
    d = 2^Ns

    if sys_mode_type == "Fermion"
        if ordering_choice == "interleaved"
            ##swap phases.
            gates =  ancilla_phase_gate_swap(qA,true,DP,P)
            ψ = apply_gates_to_ψ(ψ,gates)
        end

        ##Particle hole phases.
        if isodd(Ns)
            gates =  ancilla_phase_gate_PH(qA,true,DP,P)
            ψ = apply_gates_to_ψ(ψ,gates)
        end
    end

    if ordering_choice == "interleaved"
        ###Applies fermionic swap gates to change the order from interleaved to separated.
        ψ = system_swaps(ψ,q[1],DP,P)
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end

    ##calculate reduced density matrix.
    rm_inds = [qS ; qA]
    ρf = rdm_para(ψ,rm_inds,DP,P)

    ###applies a particle hole transformation to the ancilla states.
    gates =  particle_hole_transform(qA,true,DP,P)
    ρf = apply_gates_to_ρ(ρf,gates)
    if map_type == "Matrix"
        ###Convert Tensor to matrix.
        cutoff = 1e-5
        ρmat,Λmat = choi_isomorphism(ρf,qA,qS,cutoff,P,DP)
        return ρmat,Λmat
    elseif map_type == "MPO"
        ###Convert Tensor to MPO.
        Λ_MPO,ρ = extract_map_as_MPO(ρf,DP,P)
        return ρf,Λ_MPO,ρ
    end
end


function purification(ρ,DP,P)
    """
    Input ρ: ITensor with legs qS,qS'
    Output ρ: MPS, with qS' replaced with qA.
    """
    (;Ns,ordering_choice) = P
    (;qS,qA,s,q) = DP
    if ordering_choice == "interleaved"
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end
    for i =1:Ns
        δ = delta(DP.s[qS[i]]',DP.s[qA[i]])
        ρ = ρ*δ
    end
    ρ = MPS(ρ,s[q])
    return ρ
end



function vectorised_density_op(site,P,DP)
    (;ordering_choice,Ns) = P
    (;q,qS,qA,s) = DP
    Id = [1 0;0 1]
    N = [0 0;0 1]
    if ordering_choice == "interleaved"
        qA =  q[Ns+1:2*Ns]
        qS =  q[1:Ns]
    end
    Tensor = 1
    for (i,sys_site) in enumerate(qS)
        if sys_site == site
            op = ITensor(N,[DP.s[site],DP.s[qA[i]]])
        else
            op = ITensor(Id,[DP.s[sys_site],DP.s[qA[i]]])
        end
        Tensor = Tensor*op
    end
    return Tensor
end

function propagate_with_map_MPO(Λ_MPO,nsteps,site,P,DP)
    (;ordering_choice,Ns) = P
    (;qA,qS,q,s) = DP
    ψ = deepcopy(DP.ψ_init)
    
    if ordering_choice == "interleaved"
        ##swap phases.
        gates =  ancilla_phase_gate_swap(qA,true,DP,P)
        ψ = apply_gates_to_ψ(ψ,gates)
        ψ = system_swaps(ψ,q[1],DP,P)
        qA =  q[P.Ns+1:2*P.Ns]
        qS =  q[1:P.Ns]
    end
    
    ##Extracting the initial density matrix as an MPS
    ρ = rdm_para(ψ,qS,DP,P)
    ρ_MPS = purification(ρ,DP,P)
    
    ##Applying the map 
    ρ_evolved = ρ_MPS
    for i=1:nsteps
        ρ_evolved = apply(Λ_MPO,ρ_evolved)
    end

    ##Creating the density operator as an MPS
    N_op = vectorised_density_op(site,P,DP)
    N_op = MPS(N_op,s[q])
    Num = inner(N_op,ρ_evolved)
    return ρ_evolved,Num
end

function change_inds_old(ρ,qA,DP,P)
    (;Ns) = P
    (;s,q) = DP
    ρ_new = MPO(s[qA])
    qS = siteinds(ρ)
    for i =1:Ns
        δ1 = delta(s[qS[i]],s[qA[i]]')
        δ2 = delta(s[qS[i]]',s[qA[i]])

        MPO_term_new = ρ[i]*δ1*δ2
        ρ_new[i] = MPO_term_new
    end
    orthogonalize!(ρ_new,1)
    return ρ_new
end



