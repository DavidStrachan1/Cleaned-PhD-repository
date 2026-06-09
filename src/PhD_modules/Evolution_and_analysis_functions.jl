

    ##multitime_correlators_using_NMQMpE method, with rdm_to_MPS and related functions
    export multitime_correlators_using_NMQMpE_using_MPS
    export partial_energies
    export my_fourier_transform
    export green_function_using_correlations_matrices
    export LinearPrediction
    export rdm_to_MPS
    export Block_submatrices
    export Fock_states
    export opposite_QN
    export Spin_boson_multitime_correlators_using_NMQMpE_using_MPS
    export rdm_to_MPS_for_qubit
    export ρ_diff_average_fermions
    export ρ_diff_average_SB
    export extrapolation_vs_map_propagation_spin_boson
    export extrapolation_vs_map_propagation_fermions
    export extrapolation_vs_map_propagation_with_incremental_maps
    export incremental_maps


    ##Markovian calculations
    export markovian_green_function_calculation
    export markovian_Lindbladian
    export SB_lindblad_rates

    ###Convergence analysis/various functions for methodology paper
    export calculate_ρ_sys_using_G
    export propagate_with_converged_map
    export propagation_with_converged_L
    export convergence_analysis

    #These two functions compare the map extraction using 
    #matrices to the one using MPOs 
    export propagate_MPS_comparing_matrix_vs_MPO
    export NESS_extraction_through_power_method_comparison

    #These functions propagate the correlation matrix
    export time_evolution
    export propagate_correlations
    export enrich_state!
    export initialise_observer
    export propagate_MPS_2
    export propagate_MPS
    export MPS_extraction
    export map_extraction

    #Λ and L extraction from correlation matrix
    #Only valid for quadratic systems
    export calculate_ρ_using_G
    export corr_to_Λ
    export N_superoperator
    export apply_gates_to_ρmat
    export ρ_to_Λ



    #Λ and L extraction from the MPS as a full Tensor. 
    #This can only be applied to small systems
    export NESS_calculations
    export kth_mode_extraction
    export NESS_extraction
    export unvectorise_ρ
    
    

    #Λ and L extraction from the MPS as an MPO. 
    #This is a poor implementation as the ancilla modes 
    #are separated from their entangled system modes, causing 
    #a rainbow state and a large bond dimension
    export NESS_extraction_through_power_method
    export NESS_MPO_calculations



    #Various helper functions
    export calculate_spin_values
    export Λs_to_Louv
    export compute_Louv
    export reduced_L_extraction
    export ρ_system_corr
    export Λs_to_dΛdt
    export extract_physical_modes
    export expand_Λ
    export plot_currents
    export plot_spectra



    ##Current calculations
    export calculate_currents
    export current_operator
    export LB_current_general
    export LB_current
    export left_boundary_current
    export right_boundary_current

    function multitime_correlators_using_NMQMpE_using_MPS(end_time,spin,site,P;kwargs...)

        """
        Step 1: Obtain converged objects
        -(a): Obtain L(τmL) and Λ(τmL) and L(τmL)'s zero eigenvalue, ρ∞
        -(b): Obtain ρf, even if unphysical
    
        Step 2: Simulations using whole system and bath
        -(a): Evolve full system and bath density matrix, starting from the state
        ρ_tot(0) = ρf⊗ρ_B(0) up to time τmL. By construction,
        Tr_{B}(ρ_tot(τmL)) = ρ∞, where ρ_tot(τmL)= e^(-iHτmL)*ρ_tot(0)*e^(iHτmL).
        -(b): Create 2 new MPSs,  ϕ_L_greater=cdag|ρf(τmL)> and ϕ_L_greater=c|ρf(τmL)>.
        -(c): Evolve cdag|ρf(τmL)>,c|ρf(τmL)>,|ρf(τmL)> again by the full Hamiltonian, 
        up to time τmL.
    
        # Step 3: Only requires matrix multiplications in system Hilbert space
        # -(a): Trace out the bath to obtain \tilde{ρ}(τmL) = Tr_B(\tilde{ρ}_tot(τmL))
        # -(b): Obtain \tilde{ρ}(τ) = e^((τ-τmL)L)*\tilde{ρ}(τmL).
        
        # We now have Tr(O_4*O_3*\tilde{ρ}(τ)) = lim_(t_1→∞)⟨O_2(t_1)O_4(t_1+τ)O_3(t_1+τ)O_1(t_1)⟩.
    
        Possible kwarg inputs:
        -τmL: the (guessed) memory time of the propagator. If not specified it will default to P.T.
        -use_spin_operators: boolean that decides whether to do the map calculation and rdms using 
        spin operators or fermionic operators (with appropriate corrections). Defaults to false.
        -take symmetry_subset: decides whether to only include the physically allowed matrix elements for the maps,
        defaults to true.
        -TDVP_nsite: defaults to 2.
        -enrich_bool: a boolean that decides whether to enrich every n*δt steps
        -cutoff: used for removal of noise for rdm_to_MPS, defaults to 1e-15
        """
    
        (;minbonddim,maxbonddim,δt,n,sys_mode_type,tdvp_cutoff,Ns,method_type) = P
        τmL1 = get(kwargs,:τmL1,P.T)
        τmL2 = get(kwargs,:τmL2,τmL1)
        τmL3 = get(kwargs,:τmL3,τmL2)
        start_with_factorised_state = get(kwargs,:start_with_factorised_state,false)
        equilibriate_from_fast_state = get(kwargs,:equilibriate_from_fast_state,true)
        times1 = range(δt,stop=τmL1,step = δt)
        times2 = range(δt,stop=τmL2,step = δt) 
        TDVP_nsite = get(kwargs, :TDVP_nsite, 2)
        enrich_bool = get(kwargs,:enrich_bool,true)
        plotting_bool = get(kwargs,:plotting_bool,true)
        take_symmetry_subset = get(kwargs,:take_symmetry_subset,false)
        calculate_NESS_observables = get(kwargs,:calculate_NESS_observables,false)
        use_spin_operators = get(kwargs,:use_spin_operators,false)
        partial_energies_bool = get(kwargs,:partial_energies_bool,true)
        memory_time_ind = get(kwargs,:memory_time_ind,false)
        @assert(P.compute_maps_bool == true)
        @assert(end_time >= τmL2+τmL3)
    
        
        DP = DP_initialisation(P)
        (;cdag,c,H_MPO,qS,s,N) = DP
        site_dimension = NDTensors.dim(s[1])
        Λi = NESS_fn(DP.ψ_init,P,DP;use_spin_operators = use_spin_operators)[2]
        @show(Id_check(Λi))
    
        #Step 1
        println("
        #######################################################################
        Time evolution to calculate maps, propagators and steady state.
        #######################################################################")
        
        ψ,map_calc_obs,L_vec,Λ_vec,ρ_vec,NESS_times =  propagate_MPS_2(P, DP;times=times1,partial_energies_bool=false,kwargs...)
        SvN1 = map_calc_obs.SvN
        if calculate_NESS_observables 
            spectra,NESS_list,JL_NESS_l,JR_NESS_l,den_NESS_l = NESS_calculations(L_vec,site,"L",P,DP;kwargs...)
            if plotting_bool 
                display(Plots.plot(NESS_times,real.(den_NESS_l)))
            end
        else       
            spectra,NESS_list = NESS_calculations(L_vec,site,"L",P,DP;kwargs...)
        end
        if plotting_bool
            display(Plots.plot(NESS_times,real.(spectra)))
        end
        
        Lτm,Λτm,NESS = L_vec[end],Λ_vec[end],vectorise_mat(NESS_list[end])
        if memory_time_ind != false
            Λτm,NESS = Λ_vec[Int(memory_time_ind)],vectorise_mat(NESS_list[Int(memory_time_ind)])
        end

        if start_with_factorised_state == false
            if equilibriate_from_fast_state == true
                if take_symmetry_subset
                    @show(map_check(ψ,DP.ψ_init,expand_Λ(Λτm,P.Ns),P,DP;use_spin_operators=use_spin_operators))
                    ρf= unvectorise_ρ(expand_Λ(pinv(Λτm),P.Ns)*NESS,true)
                else
                    @show(map_check(ψ,DP.ψ_init,Λτm,P,DP;use_spin_operators=use_spin_operators))
                    ρf= unvectorise_ρ(pinv(Λτm)*NESS,true)
                end
                if sum(real.(eigen(ρf).values) .<0) > 0
                    ρf_bool = false
                    for i =reverse(1:length(L_vec))
                        ρf_test = unvectorise_ρ(pinv(Λ_vec[i])*vectorise_mat(NESS_list[i]),true)
                        if sum(real.(eigen(ρf).values) .<0) == 0
                            ρf = ρf_test   
                            ρf_bool = true
                        end
                    end
                    if ρf_bool == false
                        ρf = unvectorise_ρ(NESS,true)
                    end
                end  
            else
                ρf = unvectorise_ρ(NESS,true)
            end


            ##Switching the ordering such that the rdm is valid
            if P.ordering_choice == "interleaved"
                P.ordering_choice = "separated"
                DP = Reinitialise_Hamiltonian(P,DP)
            end

            ρf_MPS = rdm_to_MPS(ρf,P,DP;kwargs...)
            ρf_test = vectorise_ρ(ρf_MPS,P,DP;use_spin_operators=true)
            @show(norm(ρf_test-vectorise_mat(ρf)))
        
            println("
            #######################################################################
            Time evolution towards stationarity, starting with ρf
            #######################################################################")
            #Step 2
            if partial_energies_bool
                ρf_evolved,SS_prep_obs,partial_energies_vec =  propagate_MPS_2(P, DP;ψ_init=ρf_MPS,times=times2,compute_maps_bool=false,
                partial_energies_bool=partial_energies_bool,kwargs...)
            else
                ρf_evolved,SS_prep_obs=  propagate_MPS_2(P, DP;ψ_init=ρf_MPS,times=times2,compute_maps_bool=false,
                partial_energies_bool=partial_energies_bool,kwargs...)
                partial_energies_vec = [0]
            end
            SvN2 = SS_prep_obs.SvN
            if sys_mode_type == "Electron"
                corrs_up = SS_prep_obs.corr_up
                corrs_dn = SS_prep_obs.corr_dn
                den_up_vec = [corr[site,site] for corr in corrs_up];
                den_dn_vec = [corr[site,site] for corr in corrs_dn];
                
                if plotting_bool
                    Plots.plot(times2,real.(den_dn_vec),label="spin down")
                    display(Plots.plot!(times2,real.(den_up_vec),label="spin up"))
                end
            else
                corrs = SS_prep_obs.corr
                den_vec = [corr[site,site] for corr in corrs]
                if plotting_bool
                    display(Plots.plot(times2,real.(den_vec),label="density"))
                end
            end
                
            ρf_evolved_mat = vectorise_ρ(ρf_evolved,P,DP;use_spin_operators=true)
            if take_symmetry_subset
                @show(norm(ρf_evolved_mat-expand_Λ(Λτm,P.Ns)*ρf_test))
            else
                @show(norm(ρf_evolved_mat-Λτm*ρf_test))
            end
            @show(norm(NESS-ρf_evolved_mat))
        else
            ρf_evolved = DP.ψ_init
        end
        println("
        #######################################################################
        Once stationary the state is perturbed with the appropriate 
        creation/annihilation operators for the greens function calulations.
        #######################################################################")
         ##Creating ϕ_L_greater, ϕ_R_greater, ϕ_L_lesser and ϕ_R_lesser
         ##ϕ_R_greater and ϕ_L_lesser are calculated from ϕ=U(t)*ρf_evolved
         ϕ = deepcopy(ρf_evolved)
    
    
    
        #  ##Overriding the fermionic operators as I'm dealing with the JW strings manually
        # if sys_mode_type == "Electron"
        #     cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagdn","Adagup"]]) for n in 1:length(s)]
        #     c = [ops(s, [(cre_op, n) for cre_op in ["Adn","Aup"]]) for n in 1:length(s)]
        # end
    
        # Sz = spin_operators(Ns)[1]
        # parity_op = Matrix(Sz[1])
        # for i=2:Ns
        #     parity_op *= Matrix(Sz[i])
        # end
        # parity_op_vec = lmult(parity_op)
    
        # ops_ = fill("Id",N)
        # ops_[DP.qS] .= "F"
        # parity_op_MPO = MPO(DP.s,ops_)
    
        ϕ_L_greater = apply(create_fermi_cre_op(site,DP.q,DP;spin=spin),ϕ)
        ϕ_R_lesser = apply(create_fermi_ann_op(site,DP.q,DP;spin=spin),ϕ)
        #ϕ_L_greater = apply(parity_op_MPO,ϕ_L_greater)
        #ϕ_R_lesser = apply(parity_op_MPO,ϕ_R_lesser)
    
        Nsteps = Int(end_time/δt)
        Grn_t = zeros(ComplexF64, Nsteps+1)
        Grn_lesser,Grn_greater = similar(Grn_t),similar(Grn_t)
        Grn_t[1] = -1.0*im ##t=0 value
        Grn_t_test = deepcopy(Grn_t)
    
        println("
        #######################################################################
        The three MPSs required for the greens function are propagated for a time
        τmL, after which their evolution can be described by propagation
        with the fixed propagator.
        #######################################################################")
        # Configure updater parameters
        updater_kwargs = Dict(:ishermitian => P.method_type == "TFCM", :issymmetric => P.method_type == "TFCM", :eager => true)
        normalize = P.method_type == "TFCM"
    
        #  #enrichment
        #  enrich_bool && (ϕ = enrich_state!(ϕ, P, DP))
        #  enrich_bool && (ϕ_L_greater = enrich_state!(ϕ_L_greater, P, DP;normalise=false))
        #  enrich_bool && (ϕ_R_lesser = enrich_state!(ϕ_R_lesser, P, DP;normalise=false))
    
    
        sim_t = 0
        no_of_tdvp_steps = Int(τmL3/δt)
    
        SvN3 = complex(zeros(no_of_tdvp_steps))
        SvN_lesser = complex(zeros(no_of_tdvp_steps))
        SvN_greater = complex(zeros(no_of_tdvp_steps))
    
        number_of_modes = Int((site_dimension/2)*Ns)   
        cdag_mat,c_mat = matrix_operators(number_of_modes,P)
        ρ_tilde_lesser_vec = Vector{Any}(undef,Nsteps+1)
        ρ_tilde_greater_vec = Vector{Any}(undef,Nsteps+1)
        ρ_tilde_lesser_vec[1] = vectorise_ρ(ϕ,ϕ_R_lesser,P,DP;rdm_block_sparse_bool = false,kwargs...)
        ρ_tilde_greater_vec[1] = vectorise_ρ(ϕ_L_greater,ϕ,P,DP;rdm_block_sparse_bool = false,kwargs...)
    
        sys_site = findfirst(==(site),qS)
        if sys_mode_type == "Fermion"
            ##If the sites are spinless, the number of modes are 
            ##equal to the number of sites
            mode_site = Int(sys_site)
        else
            mode_site = Int((site_dimension/2)*sys_site-1 +spin-1)
        end
    
        for i =1:(no_of_tdvp_steps)
            ϕ_L_greater= tdvp(H_MPO,-im * δt,ϕ_L_greater; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=false,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)
        
            ϕ_R_lesser = tdvp(H_MPO,-im * δt,ϕ_R_lesser; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=false,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)
    
            ϕ = tdvp(H_MPO,-im * δt,ϕ; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=normalize,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)
            
            SvN3[i] = maximum(entanglement_entropy(ϕ))
            SvN_lesser[i] = maximum(entanglement_entropy(ϕ_R_lesser))
            SvN_greater[i] = maximum(entanglement_entropy(ϕ_L_greater))
    
    
            ##Calculating ϕ_R_greater and ϕ_L_lesser
            ϕ_R_greater = apply(create_fermi_cre_op(site,DP.q,DP;spin=spin),ϕ)
            ϕ_L_lesser = apply(create_fermi_ann_op(site,DP.q,DP;spin=spin),ϕ)
            #ϕ_R_greater = apply(parity_op_MPO,ϕ_R_greater)
           # ϕ_L_lesser = apply(parity_op_MPO,ϕ_L_lesser)
    
            Grn_lesser[i+1] = im*inner(ϕ_R_lesser,ϕ_L_lesser)
            Grn_greater[i+1] = -im*inner(ϕ_R_greater,ϕ_L_greater)
            Grn_t[i+1] = Grn_greater[i+1]-Grn_lesser[i+1]
    
            ρ_tilde_lesser = vectorise_ρ(ϕ,ϕ_R_lesser,P,DP;rdm_block_sparse_bool = false,kwargs...)#rdm_para(ϕ,ϕ_R_lesser,q) 
            ρ_tilde_greater= vectorise_ρ(ϕ_L_greater,ϕ,P,DP;rdm_block_sparse_bool = false,kwargs...)#rdm_para(ϕ_L_greater,ϕ,) 
            G_greater = -im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_greater,false))
            G_lesser = im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_lesser,false))
            Grn_t_test[i+1] = G_greater-G_lesser
    
            ρ_tilde_lesser_vec[i+1] = ρ_tilde_lesser
            ρ_tilde_greater_vec[i+1] = ρ_tilde_greater
            sim_t += δt
            @show(sim_t)
        end
    
        ρ_tilde_lesser = ρ_tilde_lesser_vec[no_of_tdvp_steps+1]
        ρ_tilde_greater = ρ_tilde_greater_vec[no_of_tdvp_steps+1]
    
        println("
        #######################################################################
        After τmL, ρ_tilde_lesser and ρ_tilde_greater are calculated and are 
        propagated using the converged propagator.
        #######################################################################")
        Sz = spin_operators(number_of_modes)[1]
        parity_op = Matrix(Sz[1])
        for i=2:number_of_modes
            parity_op *= Matrix(Sz[i])
        end
        parity_op_vec = lmult(parity_op)
        Lbar_τm = parity_op_vec*Lτm*parity_op_vec
    
    
        remaining_steps = Int((end_time-τmL3)/δt)
        for i = 1:remaining_steps
            ρ_tilde_lesser = exp(Lbar_τm*δt)*ρ_tilde_lesser
            ρ_tilde_greater = exp(Lbar_τm*δt)*ρ_tilde_greater
    
            Grn_greater[no_of_tdvp_steps+i+1] = -im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_greater,false))
            Grn_lesser[no_of_tdvp_steps+i+1] = im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_lesser,false))
            Grn_t[no_of_tdvp_steps+i+1] = Grn_greater[no_of_tdvp_steps+i+1]-Grn_lesser[no_of_tdvp_steps+i+1]
        end
        
        total_time = range(0,stop=end_time,step=δt)
        output_dict = Dict()
        output_dict["Grn_t"] = Grn_t
        output_dict["Grn_lesser"] = Grn_lesser
        output_dict["Grn_greater"] = Grn_greater
        output_dict["L_vec"] = L_vec
        output_dict["Λ_vec"] = Λ_vec
        output_dict["SS_prep_obs"] = SS_prep_obs
        partial_energies_bool && (output_dict["partial_energies_vec"] = partial_energies_vec)
        output_dict["map_calc_obs"] = map_calc_obs
        output_dict["ρf_MPS"] = ρf_MPS
        output_dict["ρf_evolved"] = ρf_evolved
        output_dict["ρ_tilde_lesser"] = ρ_tilde_lesser
        output_dict["ρ_tilde_greater"] = ρ_tilde_greater
        output_dict["Structs"] = P,DP
        output_dict["SvN1"] = SvN1
        output_dict["SvN2"] = SvN2
        output_dict["SvN3"] = SvN3
        output_dict["SvN_greater"] = SvN_greater
        output_dict["SvN_lesser"] = SvN_lesser
        output_dict["τmLs"] = [τmL1,τmL2,τmL3]
        output_dict["ϕ_R_lesser"] = ϕ_R_lesser
        output_dict["ϕ_L_greater"] = ϕ_L_greater
        output_dict["ϕ"] = ϕ
    
        output_dict["Grn_t_test"] = Grn_t_test
        output_dict["ρ_tilde_greater_vec"] = ρ_tilde_greater_vec
        output_dict["ρ_tilde_lesser_vec"] = ρ_tilde_lesser_vec
        return output_dict
    end

    function multitime_correlators_using_NMQMpE_using_MPS_NEQ(end_time,spin,site,P;kwargs...)

        """
        Step 1: Obtain converged objects
        -(a): Obtain L(τmL) and Λ(τmL) and L(τmL)'s zero eigenvalue, ρ∞
        -(b): Obtain ρf, even if unphysical

        Step 2: Simulations using whole system and bath
        -(a): Evolve full system and bath density matrix, starting from the state
        ρ_tot(0) = ρf⊗ρ_B(0) up to time τmL. By construction,
        Tr_{B}(ρ_tot(τmL)) = ρ∞, where ρ_tot(τmL)= e^(-iHτmL)*ρ_tot(0)*e^(iHτmL).
        -(b): Create 2 new MPSs,  ϕ_L_greater=cdag|ρf(τmL)> and ϕ_L_greater=c|ρf(τmL)>.
        -(c): Evolve cdag|ρf(τmL)>,c|ρf(τmL)>,|ρf(τmL)> again by the full Hamiltonian, 
        up to time τmL.

        # Step 3: Only requires matrix multiplications in system Hilbert space
        # -(a): Trace out the bath to obtain \tilde{ρ}(τmL) = Tr_B(\tilde{ρ}_tot(τmL))
        # -(b): Obtain \tilde{ρ}(τ) = e^((τ-τmL)L)*\tilde{ρ}(τmL).
        
        # We now have Tr(O_4*O_3*\tilde{ρ}(τ)) = lim_(t_1→∞)⟨O_2(t_1)O_4(t_1+τ)O_3(t_1+τ)O_1(t_1)⟩.

        Possible kwarg inputs:
        -τmL: the (guessed) memory time of the propagator. If not specified it will default to P.T.
        -use_spin_operators: boolean that decides whether to do the map calculation and rdms using 
        spin operators or fermionic operators (with appropriate corrections). Defaults to false.
        -take symmetry_subset: decides whether to only include the physically allowed matrix elements for the maps,
        defaults to true.
        -TDVP_nsite: defaults to 2.
        -enrich_bool: a boolean that decides whether to enrich every n*δt steps
        -cutoff: used for removal of noise for rdm_to_MPS, defaults to 1e-15
        """

        (;minbonddim,maxbonddim,δt,n,sys_mode_type,tdvp_cutoff,Ns,method_type) = P
        τmL1 = get(kwargs,:τmL1,P.T)
        τmL2 = get(kwargs,:τmL2,τmL1)
        τmL3 = get(kwargs,:τmL3,τmL2)
        start_with_factorised_state = get(kwargs,:start_with_factorised_state,false)
        equilibriate_from_fast_state = get(kwargs,:equilibriate_from_fast_state,true)
        times1 = range(δt,stop=τmL1,step = δt)
        times2 = range(δt,stop=τmL2,step = δt) 
        TDVP_nsite = get(kwargs, :TDVP_nsite, 2)
        enrich_bool = get(kwargs,:enrich_bool,true)
        plotting_bool = get(kwargs,:plotting_bool,true)
        take_symmetry_subset = get(kwargs,:take_symmetry_subset,false)
        calculate_NESS_observables = get(kwargs,:calculate_NESS_observables,false)
        use_spin_operators = get(kwargs,:use_spin_operators,false)
        partial_energies_bool = get(kwargs,:partial_energies_bool,true)
        memory_time_ind = get(kwargs,:memory_time_ind,false)
        @assert(P.compute_maps_bool == true)
        @assert(end_time >= τmL2+τmL3)

        
        DP = DP_initialisation_NEQ(P)
        (;cdag,c,H_MPO,qS,s,N) = DP
        site_dimension = NDTensors.dim(s[1])
        Λi = NESS_fn(DP.ψ_init,P,DP;use_spin_operators = use_spin_operators)[2]
        @show(Id_check(Λi))

        #Step 1
        println("
        #######################################################################
        Time evolution to calculate maps, propagators and steady state.
        #######################################################################")
        
        ψ,map_calc_obs,L_vec,Λ_vec,ρ_vec,NESS_times =  propagate_MPS_2(P, DP;times=times1,partial_energies_bool=false,kwargs...)
        SvN1 = map_calc_obs.SvN
        if calculate_NESS_observables 
            spectra,NESS_list,JL_NESS_l,JR_NESS_l,den_NESS_l = NESS_calculations(L_vec,site,"L",P,DP;kwargs...)
            if plotting_bool 
                display(Plots.plot(NESS_times,real.(den_NESS_l)))
            end
        else       
            spectra,NESS_list = NESS_calculations(L_vec,site,"L",P,DP;kwargs...)
        end
        if plotting_bool
            display(Plots.plot(NESS_times,real.(spectra)))
        end
        
        Lτm,Λτm,NESS = L_vec[end],Λ_vec[end],vectorise_mat(NESS_list[end])
        if memory_time_ind != false
            Λτm,NESS = Λ_vec[Int(memory_time_ind)],vectorise_mat(NESS_list[Int(memory_time_ind)])
        end

        if start_with_factorised_state == false
            if equilibriate_from_fast_state == true
                if take_symmetry_subset
                    @show(map_check(ψ,DP.ψ_init,expand_Λ(Λτm,P.Ns),P,DP;use_spin_operators=use_spin_operators))
                    ρf= unvectorise_ρ(expand_Λ(pinv(Λτm),P.Ns)*NESS,true)
                else
                    @show(map_check(ψ,DP.ψ_init,Λτm,P,DP;use_spin_operators=use_spin_operators))
                    ρf= unvectorise_ρ(pinv(Λτm)*NESS,true)
                end
                if sum(real.(eigen(ρf).values) .<0) > 0
                    ρf_bool = false
                    for i =reverse(1:length(L_vec))
                        ρf_test = unvectorise_ρ(pinv(Λ_vec[i])*vectorise_mat(NESS_list[i]),true)
                        if sum(real.(eigen(ρf).values) .<0) == 0
                            ρf = ρf_test   
                            ρf_bool = true
                        end
                    end
                    if ρf_bool == false
                        ρf = unvectorise_ρ(NESS,true)
                    end
                end  
            else
                ρf = unvectorise_ρ(NESS,true)
            end


            ##Switching the ordering such that the rdm is valid
            if P.ordering_choice == "interleaved"
                P.ordering_choice = "separated"
                DP = Reinitialise_Hamiltonian(P,DP)
            end

            ρf_MPS = rdm_to_MPS(ρf,P,DP;kwargs...)
            ρf_test = vectorise_ρ(ρf_MPS,P,DP;use_spin_operators=true)
            @show(norm(ρf_test-vectorise_mat(ρf)))
        
            println("
            #######################################################################
            Time evolution towards stationarity, starting with ρf
            #######################################################################")
            #Step 2
            if partial_energies_bool
                ρf_evolved,SS_prep_obs,partial_energies_vec =  propagate_MPS_2(P, DP;ψ_init=ρf_MPS,times=times2,compute_maps_bool=false,
                partial_energies_bool=partial_energies_bool,kwargs...)
            else
                ρf_evolved,SS_prep_obs=  propagate_MPS_2(P, DP;ψ_init=ρf_MPS,times=times2,compute_maps_bool=false,
                partial_energies_bool=partial_energies_bool,kwargs...)
                partial_energies_vec = [0]
            end
            SvN2 = SS_prep_obs.SvN
            if sys_mode_type == "Electron"
                corrs_up = SS_prep_obs.corr_up
                corrs_dn = SS_prep_obs.corr_dn
                den_up_vec = [corr[site,site] for corr in corrs_up];
                den_dn_vec = [corr[site,site] for corr in corrs_dn];
                
                if plotting_bool
                    Plots.plot(times2,real.(den_dn_vec),label="spin down")
                    display(Plots.plot!(times2,real.(den_up_vec),label="spin up"))
                end
            else
                corrs = SS_prep_obs.corr
                den_vec = [corr[site,site] for corr in corrs]
                if plotting_bool
                    display(Plots.plot(times2,real.(den_vec),label="density"))
                end
            end
                
            ρf_evolved_mat = vectorise_ρ(ρf_evolved,P,DP;use_spin_operators=true)
            if take_symmetry_subset
                @show(norm(ρf_evolved_mat-expand_Λ(Λτm,P.Ns)*ρf_test))
            else
                @show(norm(ρf_evolved_mat-Λτm*ρf_test))
            end
            @show(norm(NESS-ρf_evolved_mat))
        else
            ρf_evolved = DP.ψ_init
        end
        println("
        #######################################################################
        Once stationary the state is perturbed with the appropriate 
        creation/annihilation operators for the greens function calulations.
        #######################################################################")
            ##Creating ϕ_L_greater, ϕ_R_greater, ϕ_L_lesser and ϕ_R_lesser
            ##ϕ_R_greater and ϕ_L_lesser are calculated from ϕ=U(t)*ρf_evolved
            ϕ = deepcopy(ρf_evolved)



        #  ##Overriding the fermionic operators as I'm dealing with the JW strings manually
        # if sys_mode_type == "Electron"
        #     cdag = [ops(s, [(cre_op, n) for cre_op in ["Adagdn","Adagup"]]) for n in 1:length(s)]
        #     c = [ops(s, [(cre_op, n) for cre_op in ["Adn","Aup"]]) for n in 1:length(s)]
        # end

        # Sz = spin_operators(Ns)[1]
        # parity_op = Matrix(Sz[1])
        # for i=2:Ns
        #     parity_op *= Matrix(Sz[i])
        # end
        # parity_op_vec = lmult(parity_op)

        # ops_ = fill("Id",N)
        # ops_[DP.qS] .= "F"
        # parity_op_MPO = MPO(DP.s,ops_)

        ϕ_L_greater = apply(create_fermi_cre_op(site,DP.q,DP;spin=spin),ϕ)
        ϕ_R_lesser = apply(create_fermi_ann_op(site,DP.q,DP;spin=spin),ϕ)
        #ϕ_L_greater = apply(parity_op_MPO,ϕ_L_greater)
        #ϕ_R_lesser = apply(parity_op_MPO,ϕ_R_lesser)

        Nsteps = Int(end_time/δt)
        Grn_t = zeros(ComplexF64, Nsteps+1)
        Grn_lesser,Grn_greater = similar(Grn_t),similar(Grn_t)
        Grn_t[1] = -1.0*im ##t=0 value
        Grn_t_test = deepcopy(Grn_t)

        println("
        #######################################################################
        The three MPSs required for the greens function are propagated for a time
        τmL, after which their evolution can be described by propagation
        with the fixed propagator.
        #######################################################################")
        # Configure updater parameters
        updater_kwargs = Dict(:ishermitian => P.method_type == "TFCM", :issymmetric => P.method_type == "TFCM", :eager => true)
        normalize = P.method_type == "TFCM"

        #  #enrichment
        #  enrich_bool && (ϕ = enrich_state!(ϕ, P, DP))
        #  enrich_bool && (ϕ_L_greater = enrich_state!(ϕ_L_greater, P, DP;normalise=false))
        #  enrich_bool && (ϕ_R_lesser = enrich_state!(ϕ_R_lesser, P, DP;normalise=false))


        sim_t = 0
        no_of_tdvp_steps = Int(τmL3/δt)

        SvN3 = complex(zeros(no_of_tdvp_steps))
        SvN_lesser = complex(zeros(no_of_tdvp_steps))
        SvN_greater = complex(zeros(no_of_tdvp_steps))

        number_of_modes = Int((site_dimension/2)*Ns)   
        cdag_mat,c_mat = matrix_operators(number_of_modes,P)
        ρ_tilde_lesser_vec = Vector{Any}(undef,Nsteps+1)
        ρ_tilde_greater_vec = Vector{Any}(undef,Nsteps+1)
        ρ_tilde_lesser_vec[1] = vectorise_ρ(ϕ,ϕ_R_lesser,P,DP;rdm_block_sparse_bool = false,kwargs...)
        ρ_tilde_greater_vec[1] = vectorise_ρ(ϕ_L_greater,ϕ,P,DP;rdm_block_sparse_bool = false,kwargs...)

        sys_site = findfirst(==(site),qS)
        if sys_mode_type == "Fermion"
            ##If the sites are spinless, the number of modes are 
            ##equal to the number of sites
            mode_site = Int(sys_site)
        else
            mode_site = Int((site_dimension/2)*sys_site-1 +spin-1)
        end

        for i =1:(no_of_tdvp_steps)
            ϕ_L_greater= tdvp(H_MPO,-im * δt,ϕ_L_greater; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=false,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)
        
            ϕ_R_lesser = tdvp(H_MPO,-im * δt,ϕ_R_lesser; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=false,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)

            ϕ = tdvp(H_MPO,-im * δt,ϕ; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=normalize,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)
            
            SvN3[i] = maximum(entanglement_entropy(ϕ))
            SvN_lesser[i] = maximum(entanglement_entropy(ϕ_R_lesser))
            SvN_greater[i] = maximum(entanglement_entropy(ϕ_L_greater))


            ##Calculating ϕ_R_greater and ϕ_L_lesser
            ϕ_R_greater = apply(create_fermi_cre_op(site,DP.q,DP;spin=spin),ϕ)
            ϕ_L_lesser = apply(create_fermi_ann_op(site,DP.q,DP;spin=spin),ϕ)
            #ϕ_R_greater = apply(parity_op_MPO,ϕ_R_greater)
            # ϕ_L_lesser = apply(parity_op_MPO,ϕ_L_lesser)

            Grn_lesser[i+1] = im*inner(ϕ_R_lesser,ϕ_L_lesser)
            Grn_greater[i+1] = -im*inner(ϕ_R_greater,ϕ_L_greater)
            Grn_t[i+1] = Grn_greater[i+1]-Grn_lesser[i+1]

            ρ_tilde_lesser = vectorise_ρ(ϕ,ϕ_R_lesser,P,DP;rdm_block_sparse_bool = false,kwargs...)#rdm_para(ϕ,ϕ_R_lesser,q) 
            ρ_tilde_greater= vectorise_ρ(ϕ_L_greater,ϕ,P,DP;rdm_block_sparse_bool = false,kwargs...)#rdm_para(ϕ_L_greater,ϕ,) 
            G_greater = -im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_greater,false))
            G_lesser = im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_lesser,false))
            Grn_t_test[i+1] = G_greater-G_lesser

            ρ_tilde_lesser_vec[i+1] = ρ_tilde_lesser
            ρ_tilde_greater_vec[i+1] = ρ_tilde_greater
            sim_t += δt
            @show(sim_t)
        end

        ρ_tilde_lesser = ρ_tilde_lesser_vec[no_of_tdvp_steps+1]
        ρ_tilde_greater = ρ_tilde_greater_vec[no_of_tdvp_steps+1]

        println("
        #######################################################################
        After τmL, ρ_tilde_lesser and ρ_tilde_greater are calculated and are 
        propagated using the converged propagator.
        #######################################################################")
        Sz = spin_operators(number_of_modes)[1]
        parity_op = Matrix(Sz[1])
        for i=2:number_of_modes
            parity_op *= Matrix(Sz[i])
        end
        parity_op_vec = lmult(parity_op)
        Lbar_τm = parity_op_vec*Lτm*parity_op_vec


        remaining_steps = Int((end_time-τmL3)/δt)
        for i = 1:remaining_steps
            ρ_tilde_lesser = exp(Lbar_τm*δt)*ρ_tilde_lesser
            ρ_tilde_greater = exp(Lbar_τm*δt)*ρ_tilde_greater

            Grn_greater[no_of_tdvp_steps+i+1] = -im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_greater,false))
            Grn_lesser[no_of_tdvp_steps+i+1] = im*tr(c_mat[mode_site]*unvectorise_ρ(ρ_tilde_lesser,false))
            Grn_t[no_of_tdvp_steps+i+1] = Grn_greater[no_of_tdvp_steps+i+1]-Grn_lesser[no_of_tdvp_steps+i+1]
        end
        
        total_time = range(0,stop=end_time,step=δt)
        output_dict = Dict()
        output_dict["Grn_t"] = Grn_t
        output_dict["Grn_lesser"] = Grn_lesser
        output_dict["Grn_greater"] = Grn_greater
        output_dict["L_vec"] = L_vec
        output_dict["Λ_vec"] = Λ_vec
        output_dict["SS_prep_obs"] = SS_prep_obs
        partial_energies_bool && (output_dict["partial_energies_vec"] = partial_energies_vec)
        output_dict["map_calc_obs"] = map_calc_obs
        output_dict["ρf_MPS"] = ρf_MPS
        output_dict["ρf_evolved"] = ρf_evolved
        output_dict["ρ_tilde_lesser"] = ρ_tilde_lesser
        output_dict["ρ_tilde_greater"] = ρ_tilde_greater
        output_dict["Structs"] = P,DP
        output_dict["SvN1"] = SvN1
        output_dict["SvN2"] = SvN2
        output_dict["SvN3"] = SvN3
        output_dict["SvN_greater"] = SvN_greater
        output_dict["SvN_lesser"] = SvN_lesser
        output_dict["τmLs"] = [τmL1,τmL2,τmL3]
        output_dict["ϕ_R_lesser"] = ϕ_R_lesser
        output_dict["ϕ_L_greater"] = ϕ_L_greater
        output_dict["ϕ"] = ϕ

        output_dict["Grn_t_test"] = Grn_t_test
        output_dict["ρ_tilde_greater_vec"] = ρ_tilde_greater_vec
        output_dict["ρ_tilde_lesser_vec"] = ρ_tilde_lesser_vec
        return output_dict
    end

    function partial_energies(state,M,DP)
        (;MPO_terms,qB_R) = DP

        ##This ensures M is counting the bath modes
        end_index = Int(2*M+qB_R[1]-1)
        
        # Create a new OpSum to hold filtered terms
        terms = OpSum()

        for term in MPO_terms#H.terms
            str = string(term)
            # Extract all site indices from the term string using regex
            indices = parse.(Int, collect(m.captures[1] for m in eachmatch(r"\((\d+),\)", str)))
            
            # Keep term only if all site indices ≤ max_site
            if all(i <= end_index for i in indices)
                terms += term
            end
        end

        partial_H = MPO(terms,DP.s)

        partial_E = inner(state,apply(partial_H,state))
        return partial_E
    end

    function my_fourier_transform(f_t,t,dt,ω)
        f_t_ = deepcopy(f_t)
        exponential = exp.(im*ω*t)
        f_ω = sum(exponential.*f_t_)*dt
        return f_ω
    end

    function green_function_using_correlations_matrices(site,times,DP,P)

        """
        The correlation matrix C_ij(t,t') = expect(cdag[j](t)*c[i](t')) propagates according to
        C_ij(t,t') =U(t)*C_ij(0)*U(t').
    
        """
        (;T) = P
        (;Ci, H_single,qS) = DP
        δt = times[2] - times[1]
        U_step = exp(-im*δt*H_single)
        G_retarded = Vector{Any}(undef,length(times))
        G_eq = Vector{Any}(undef,length(times))
        
    
        G_lesser = Ci
        G_greater = I-Ci
        G_retarded[1] = -im*(G_greater+G_lesser)[qS[sys_site],qS[sys_site]]
        G_eq[1] = Ci
        for i in 2:length(times)
            G_greater = U_step*G_greater
            G_lesser = G_lesser*U_step'
            G_retarded[i] = -im*(G_greater+G_lesser)[qS[sys_site],qS[sys_site]]
            G_eq[i] = U_step*G_eq[i-1]*U_step'
        end
        return G_retarded,G_eq
    end


    function LinearPrediction(data,δt,p,tfit,t_target)
        """
        This function implements Linear Prediction as described in 
        https://journals.aps.org/prb/abstract/10.1103/PhysRevB.79.245101 
        on page 2.
    
        p denotes the number of points used for the extrapolation.
        No weighting function is used.
        """
        N = length(data)
        times = range(δt,stop = t_target,step = δt)
        tfit_ind = findfirst(==(tfit),round.(times,digits=1))
        Nfit_vec = tfit_ind:N ##vector of indices for points between t=tfit and N*δt.
     
        R = complex(zeros(p,p))
        r = complex(zeros(p))
        for j=1:p
            for i=1:p
                R[j,i] = sum([conj(data[n-j])*data[n-i] for n in Nfit_vec])
            end
            r[j] = sum([conj(data[n-j])*data[n] for n in Nfit_vec])
        end
        Rinv = pinv(R)
        a = -Rinv*r
    
        number_of_extrapolated_points = Int(t_target/δt - N)
        extrapolated_data = [data;complex(zeros(number_of_extrapolated_points))]
        for n=N+1:N+number_of_extrapolated_points
            extrapolated_data[n] = -sum([a[i]*extrapolated_data[n-i] for i=1:p])
        end
        @show(times)
        return times,extrapolated_data
    end



    function opposite_QN(QN,d,symmetry_subspace,Ns)
        if d == 4 
            ##maximum Num is 2*Ns
            #QN_N = QN[1]
            if symmetry_subspace == "Number and Sz conserving"
                #want total QN to be [2*Ns,0]
                return [Int(2*Ns-QN[1]),Int(-QN[2])]
            else
                return [Int(2*Ns-QN[1])]
            end
        elseif d ==2
            ##maximum Num is Ns
            return [Int(Ns - QN[1])]
        end
    end

    function Block_submatrices(Ns,d,symmetry_subspace)
        if d == 2
            Block_mats = Vector{Matrix{ComplexF64}}(undef,Ns+1) ##Vector of the block matrices
            Block_ind_vecs = Vector{Vector{Any}}(undef,Ns+1) ##Vector giving the basis of each blockmatrix in ditstrings. 
            Block_QN_vec = Vector{Vector{Any}}(undef,Ns+1) ##list of QNs of the blocks
            for Num=0:Ns
                ##For Ns spinless fermionic sites, the only symmetry is number conservation. The size
                ##of the block for Num particles is given by the number of ways you can arrange Num sites in Ns sites.
                block_size = binomial(Ns,Num)
                emp_mat = complex(zeros(block_size,block_size))
                emp_vec = Vector{Any}(undef,block_size)
                Block_mats[Num+1] = emp_mat #List of empty matrices representing the allowed block matrix
                Block_ind_vecs[Num+1] = emp_vec 
                Block_QN_vec[Num+1] = [Num]
    
            end
        elseif d == 4
            if symmetry_subspace == "Number conserving"
                Block_mats = Vector{Matrix{ComplexF64}}(undef,2*Ns+1) ##Vector of the block matrices
                Block_ind_vecs = Vector{Vector{Any}}(undef,2*Ns+1) ##Vector giving the basis of each blockmatrix in ditstrings. 
                Block_QN_vec = Vector{Vector{Any}}(undef,2*Ns+1)
                ##The number of ways to have Num fermions across Ns spinful sites is counted by first choosing how many
                ##spinup fermions there are (say Num_up) and multiply that by the number of ways to have Num-Num_up
                ##spindown fermions. This is then summed over Num_up=0 to Num.
                for Num =0:2*Ns
                    block_size = sum(binomial(Ns, Num_up) * binomial(Ns, Num - Num_up) for Num_up in 0:Num)
                    emp_mat = complex(zeros(block_size,block_size))
                    emp_vec = Vector{Any}(undef,block_size)
                    Block_mats[Num+1] = emp_mat #List of empty matrices representing the allowed block matrix
                    Block_ind_vecs[Num+1] = emp_vec 
                    Block_QN_vec[Num+1] = [Num]
                end
            elseif symmetry_subspace == "Number and Sz conserving"
    
                ##Num can again range from 0 to 2*Ns. If Num<Ns, there can be Num single occupied states such that 
                ##Sz can range from -Num:2:Num. However, if Num>Ns, there must be at least Num-Ns sites double occupied 
                ##So the number of holes/singly occupied sites can be at most 2*Ns-Num. Some reasoning:
                ##Ndouble ≥ Num-Ns. Nsingle+2*Ndouble = Num and Nsingle+Ndouble ≤ Ns. If Ndouble = Num-Ns,
                ##Then Nsingle = 2*Ns-Num. Doubly occupied sites must have Sz =0, 
                ##so don't contribute to the different possible Sz values. Therefore, the number of Sz 
                ##values for a given Num is -(2*Ns-Num):2:(2*Ns-Num). Due to symmetry the number of symmetry blocks
                ##for Ns<Num≤2*Ns is the same as for 0≤Num<Ns.  
                
                ##For 0≤Num<Ns sum_{Num=0}^{Ns-1}(1+Num) = Ns +(Ns-1)*Ns/2 = Ns*(Ns+1)/2.
                ##For Num = Ns, there are Ns+1 possible Sz values. In total this gives
                num_of_blocks = Int((Ns+1)^2)
                Block_mats =  Vector{Matrix{ComplexF64}}(undef,num_of_blocks) ##Vector of the block matrices
                Block_ind_vecs = Vector{Vector{Any}}(undef,num_of_blocks) ##Vector giving the basis of each blockmatrix in ditstrings. 
                Block_QN_vec = Vector{Vector{Any}}(undef,num_of_blocks) ##list of QNs of the blocks
                ##start_ind tracks how many blocks have already been considered 
                start_ind = 0
                for Num = 0:2*Ns
                    if Num <= Ns
                        max_Sz_val = Num
                    else
                        max_Sz_val = 2*Ns-Num
                    end
                    for (Sz_counter,Sz) in enumerate(-max_Sz_val:2:max_Sz_val)
                        Num_up = Int((Num + Sz)/2)
                        Num_dn = Num-Num_up
                        block_size = binomial(Ns, Num_up) * binomial(Ns,Num_dn)
                        emp_mat = complex(zeros(block_size,block_size))
                        emp_vec = Vector{Any}(undef,block_size)
                        Block_mats[start_ind+Sz_counter] = emp_mat #List of empty matrices representing the allowed block matrix
                        Block_ind_vecs[start_ind+Sz_counter] = emp_vec 
                        Block_QN_vec[start_ind+Sz_counter] = [Num,Sz]
                    end
                    ##Every Num sector has max_Sz_val+1 subsectors denoting the different possible magnetizations, so start ind
                    ##goes up by this.
                    start_ind += max_Sz_val+1
                end
            end
        end
    
        return Block_mats,Block_ind_vecs,Block_QN_vec
    end
    function Fock_states(P,DP)

    
        ##given Ns sites, there are 2^Ns fock states.
        (;Ns,symmetry_subspace) = P
        (;s,qS) = DP
        d = NDTensors.dim(s[1])
    
        ##initialises an array with the same dimensions as a tensor with indices s[qS].
        fock_list = []
        eig_state = ITensor(s[qS])
        eig_state_arr = 0*Array{ComplexF64}(eig_state,s[qS])
    
        for j=0:((d^Ns)-1)
            ##This converts the index to the appropriate ditstring for the MPS index.
            inds = ones(Int,Ns)
            inds[end-length(digits(j, base=d))+1:end] = reverse(digits(j, base=d)) .+1
            fockstate = copy(eig_state_arr)
            fockstate[CartesianIndex(Tuple(inds))] = 1
            #Num = sum(x .-1) ##number occupation
            QNs = QNumbers(inds,d,symmetry_subspace) 
            push!(fock_list,(fockstate,QNs))
        end
    
        return fock_list
    end
    function rdm_to_MPS(ρ,P,DP;kwargs...)
        """
        -Assuming ρ is in matrix form at this point.
        -Block_ind_vecs is a vector of vectors, where each element is the set of tensor indices that 
        relate the matrix basis of the Block matrices to the tensor basis.
        """
        (;N_L,N_R,Ns,symmetry_subspace,bath_mode_type,ordering_choice) = P
        (;s,q,qS,qA,N,qB_L,qB_R) = DP
        
        @assert(qS == q[1:Ns])
        @assert(qA == q[Ns+1:2*Ns])
        @assert(ordering_choice == "separated")
 
        Fock_list= Fock_states(P,DP)
    
        d = NDTensors.dim(s[1])
    
    
        ##This removes any numerical noise that can interfere with the diagonalisation. 
        ρ = noise_removal(ρ;kwargs...)
    
        ##Initialising vectors and tensors used later in the function
        sys_ITensor = ITensor(s[qS],s[qA]) ##Tensor that will be converted to the system+ancilla part of the MPS.
        Block_mats,Block_ind_vecs,Block_QN_vec = Block_submatrices(Ns,d,symmetry_subspace)
       
        num_of_sym_blocks = length(Block_ind_vecs)
        ##These are counters that track how many indices of a given symmetry block have been stored
        ind_count_rows = zeros(num_of_sym_blocks) 
        ind_count_cols = zeros(num_of_sym_blocks)
    
    
        ##loop over matrix elements of ρ and assign them to the appropriate block matrix.
        for i=1:d^(Ns)
    
            ##This converts the row index to the appropriate ditstring for the MPS index.
            row_inds = ones(Int,Ns)
            row_inds[end-length(digits(i-1, base=d))+1:end] = reverse(digits(i-1, base=d)) .+1
            
            ##This extracts the QNs associated with the index i, and finds the appropriate 
            ##Place to store the MPS index in Block_ind_vecs. block_index is the index
            ##of the block, but not the specific place within the block. This is given by a.
    
            QNs = QNumbers(row_inds,d,symmetry_subspace)
            block_index = findfirst(==(QNs),Block_QN_vec)
            ind_count_rows[block_index] += 1
            a = Int(ind_count_rows[block_index])
            Block_ind_vecs[block_index][a] =  [s[qS[j]] => row_inds[j] for j in 1:Ns]
    
            ind_count_cols[block_index] = 0 ##resets column iteration  
            for j=1:(d^Ns) 
                ##Repeats the same idea for the column index. If i and 
                ##j exists in the same symmetry block, they are assigned to 
                ##that Block in Block_mats
    
                col_inds = ones(Int,Ns)
                col_inds[end-length(digits(j-1, base=d))+1:end] = reverse(digits(j-1, base=d)) .+1
                if  QN_matching(row_inds,col_inds,d,symmetry_subspace)
                    ind_count_cols[block_index] += 1
                    b = Int(ind_count_cols[block_index])
                    Block_mats[block_index][a,b] = ρ[i,j]
                end
            end
        end
    
    
        for (inds,mat,QNs) in zip(Block_ind_vecs,Block_mats,Block_QN_vec)
            ##inds gives the basis of the block matrix mat.
            ###diagonalising each block matrix.
            spec_Num = noise_removal(eigen(mat).values;kwargs...)
            vecs_Num = noise_removal(eigen(mat).vectors;kwargs...)
            Block_dimension = length(spec_Num)
           
            ##These two loops loop over all the matrix elements of the matrix of eigenvectors of a given symmetry subspace, 
            ##where each eigenvector is converted to a tensor (eig_tensor) with the appropriate
            ##indices
    
            for j=1:Block_dimension
                eig_Tensor = ITensor(s[qS])
                for k =1:Block_dimension
                    v = (inds)[k]
                    eig_Tensor[v...] = vecs_Num[k,j]
                end
    
                opposite_QNs = opposite_QN(QNs,d,symmetry_subspace,Ns)
                index = findfirst(Fock -> Fock[2] == opposite_QNs, Fock_list)
                anc_Tensor= ITensor((Fock_list[index])[1],s[qA])
                splice!(Fock_list,index)
                Full_Tensor = eig_Tensor*anc_Tensor
                sys_ITensor += Full_Tensor*√(spec_Num[j])
            end
        end
        """
        We now turn sys_ITensor into an MPS with buffer sites at each end such that
        sys_MPS[q] has open link indices at each end. We then initialise the thermal state 
        of the baths and use delta functions so they share link indices where they're combined.
        """
        left_buffer = N_L >0 ? 1 : 0
        right_buffer = N_R >0 ? 1 : 0
        buff_q = (q[1]-left_buffer):(q[end]+right_buffer)
    
        if N_L>0
            sys_ITensor = diagITensor(1,s[buff_q[1]])*sys_ITensor
        end
        if N_R>0 
            sys_ITensor = sys_ITensor*diagITensor(1,s[buff_q[end]])
        end
        sys_MPS = MPS(sys_ITensor,s[buff_q])

        """
        Now sys_MPS is created, we need to initialise it within a larger MPS in a QN conserving way.
        To do this, we excite the sites from right to left, initialising the thermofield vacuum for the right bath
        and also for the system which we then overwrite with sys_MPS. Then, we apply creation operators to excite the 
        filled states in the thermofield vacuum for the left bath.
        """

        if bath_mode_type == "Electron"
            Empty = "Emp"
            Full = "UpDn"
        else
            Empty = "0"
            Full = "1"
        end
        
        left_states = ["0" for n in qB_L]
        sys_states =  [isodd(n) ? Full : Empty for n in q]
        right_states =  [isodd(n) ? Full : Empty for n in qB_R]
        if N_L>0 && N_R>0
            states = [left_states;sys_states;right_states]
        elseif N_L>0 
            states = [left_states;sys_states]
        elseif N_R>0
            states = [sys_states;right_states]
        end

        therm = MPS(ComplexF64, s, states)
        sys_lk = linkinds(sys_MPS)
        therm_lk = linkinds(therm)

        if N_L >0 
            delta_L = delta(sys_lk[1],dag(therm_lk[2*N_L]))
            sys_MPS[2] = sys_MPS[2]*delta_L
        end
        if N_R >0 
            delta_R = delta(dag(sys_lk[end]),therm_lk[2*N_L+2*Ns])
            sys_MPS[end-1] = sys_MPS[end-1]*delta_R
        end

        b = N_L >0 ? 1 : 0
        for i in q
            b += 1 
            therm[i] = sys_MPS[b]
        end
        
        for i=1:2:qB_L[end]
            if bath_mode_type == "Electron"
                O = apply(op("Cdagup",DP.s[i]),op("Cdagdn",DP.s[i]))
            elseif bath_mode_type == "Fermion"
                O = op("Cdag",DP.s[i])
            else
                throw("rdm_to_MPS Not implemented for this sitetype")
            end
            therm = apply(O,therm)
        end
        
        return therm 
    end

    function Spin_boson_multitime_correlators_using_NMQMpE_using_MPS(end_time,operator,P;kwargs...)

        """
        Step 1: Obtain converged objects
        -(a): Obtain L(τmL) and Λ(τmL) and L(τmL)'s zero eigenvalue, ρ∞
        -(b): Obtain ρf, even if unphysical

        Step 2: Simulations using whole system and bath
        -(a): Evolve full system and bath density matrix, starting from the state
        ρ_tot(0) = ρf⊗ρ_B(0) up to time τmL. By construction,
        Tr_{B}(ρ_tot(τmL)) = ρ∞, where ρ_tot(τmL)= e^(-iHτmL)*ρ_tot(0)*e^(iHτmL).
        -(b): Create 2 new MPSs,  ϕ_L_greater=cdag|ρf(τmL)> and ϕ_L_greater=c|ρf(τmL)>.
        -(c): Evolve cdag|ρf(τmL)>,c|ρf(τmL)>,|ρf(τmL)> again by the full Hamiltonian, 
        up to time τmL.

        # Step 3: Only requires matrix multiplications in system Hilbert space
        # -(a): Trace out the bath to obtain \tilde{ρ}(τmL) = Tr_B(\tilde{ρ}_tot(τmL))
        # -(b): Obtain \tilde{ρ}(τ) = e^((τ-τmL)L)*\tilde{ρ}(τmL).
        
        # We now have Tr(O_4*O_3*\tilde{ρ}(τ)) = lim_(t_1→∞)⟨O_2(t_1)O_4(t_1+τ)O_3(t_1+τ)O_1(t_1)⟩.

        Possible kwarg inputs:
        -τmL: the (guessed) memory time of the propagator. If not specified it will default to P.T.
        -use_spin_operators: boolean that decides whether to do the map calculation and rdms using 
        spin operators or fermionic operators (with appropriate corrections). Defaults to false.
        -take symmetry_subset: decides whether to only include the physically allowed matrix elements for the maps,
        defaults to true.
        -TDVP_nsite: defaults to 2.
        -enrich_bool: a boolean that decides whether to enrich every n*δt steps
        -cutoff: used for removal of noise for rdm_to_MPS, defaults to 1e-15
        """

        (;minbonddim,maxbonddim,δt,n,sys_mode_type,tdvp_cutoff,Ns,method_type) = P
        τmL1 = get(kwargs,:τmL1,P.T)
        τmL2 = get(kwargs,:τmL2,τmL1)
        τmL3 = get(kwargs,:τmL3,τmL2)
        start_with_factorised_state = get(kwargs,:start_with_factorised_state,false)
        equilibriate_from_fast_state = get(kwargs,:equilibriate_from_fast_state,true)
        TDVP_nsite = get(kwargs, :TDVP_nsite, 2)
        enrich_bool = get(kwargs,:enrich_bool,true)
        plotting_bool = get(kwargs,:plotting_bool,true)
        take_symmetry_subset = get(kwargs,:take_symmetry_subset,false)
        calculate_NESS_observables = get(kwargs,:calculate_NESS_observables,false)
        partial_energies_bool = get(kwargs,:partial_energies_bool,true)
        @assert(P.compute_maps_bool == true)
        @assert(end_time >= τmL2+τmL3)


        DP = DP_initialisation(P)
        (;cdag,c,H_MPO,qS,s,N) = DP
        use_spin_operators = true
        site = qS[1]
        times1 = range(δt,stop=τmL1,step = δt)
        times2 = range(δt,stop=τmL2,step = δt) 

        Λi = NESS_fn(DP.ψ_init,P,DP;use_spin_operators = use_spin_operators)[2]
        @show(Id_check(Λi))

        #Step 1
        println("
        #######################################################################
        Time evolution to calculate maps, propagators and steady state.
        #######################################################################")
        
        ψ,map_calc_obs,L_vec,Λ_vec,ρ_vec,NESS_times =  propagate_MPS_2(P, DP;times=times1,partial_energies_bool=false,kwargs...)
        SvN1 = map_calc_obs.SvN
        if calculate_NESS_observables 
            spectra,NESS_list,σx_NESS,σy_NESS,σz_NESS = NESS_calculations(L_vec,site,"L",P,DP;calculate_NESS_observables,kwargs...)
            if plotting_bool 
                Plots.plot(NESS_times,real.(σx_NESS),label="σx_NESS")
                Plots.plot!(NESS_times,real.(σy_NESS),label="σy_NESS")
                display(Plots.plot!(NESS_times,real.(σz_NESS),label="σz_NESS"))
            end
        else       
            spectra,NESS_list = NESS_calculations(L_vec,site,"L",P,DP;calculate_NESS_observables,kwargs...)
        end
        if plotting_bool
            display(Plots.plot(NESS_times,real.(spectra)))
        end
        Lτm,Λτm,NESS = L_vec[end],Λ_vec[end],vectorise_mat(NESS_list[end])
        
        if start_with_factorised_state == false
            if equilibriate_from_fast_state == true
                if take_symmetry_subset
                    @show(map_check(ψ,DP.ψ_init,expand_Λ(Λτm,P.Ns),P,DP;use_spin_operators=use_spin_operators))
                    ρf= unvectorise_ρ(expand_Λ(pinv(Λτm),P.Ns)*NESS,true)
                else
                    @show(map_check(ψ,DP.ψ_init,Λτm,P,DP;use_spin_operators=use_spin_operators))
                    ρf= unvectorise_ρ(pinv(Λτm)*NESS,true)
                end
                if sum(real.(eigen(ρf).values) .<0) > 0
                    ρf_bool = false
                    for i =reverse(1:length(L_vec))
                        ρf_test = unvectorise_ρ(pinv(Λ_vec[i])*vectorise_mat(NESS_list[i]),true)
                        if sum(real.(eigen(ρf_test).values) .<0) == 0
                            ρf = ρf_test   
                            ρf_bool = true
                            println("fast state index of extraction")
                            @show(i)
                            break
                        end
                    end
                    if ρf_bool == false
                        ρf = unvectorise_ρ(NESS,true)
                    end
                end  
            else
                ρf = unvectorise_ρ(NESS,true)
            end


            ##Switching the ordering such that the rdm is valid
            if P.ordering_choice == "interleaved" && (Ns > 1)
                P.ordering_choice = "separated"
                DP = Reinitialise_Hamiltonian(P,DP)
            end
            @show(tr(ρf))
            ρf_MPS = rdm_to_MPS_for_qubit(ρf,P,DP;kwargs...)
            ρf_test = vectorise_ρ(ρf_MPS,P,DP;use_spin_operators=true)
            @show(norm(ρf_test-vectorise_mat(ρf)))
        
            println("
            #######################################################################
            Time evolution towards stationarity, starting with ρf
            #######################################################################")
            #Step 2
            if partial_energies_bool
                ρf_evolved,SS_prep_obs,partial_energies_vec =  propagate_MPS_2(P, DP;ψ_init=ρf_MPS,times=times2,compute_maps_bool=false,
                partial_energies_bool=partial_energies_bool,kwargs...)
            else
                ρf_evolved,SS_prep_obs=  propagate_MPS_2(P, DP;ψ_init=ρf_MPS,times=times2,compute_maps_bool=false,
                partial_energies_bool=partial_energies_bool,kwargs...)
                partial_energies_vec = [0]
            end
            SvN2 = SS_prep_obs.SvN
                
            ρf_evolved_mat = vectorise_ρ(ρf_evolved,P,DP;use_spin_operators=true)
            if take_symmetry_subset
                @show(norm(ρf_evolved_mat-expand_Λ(Λτm,P.Ns)*ρf_test))
            else
                @show(norm(ρf_evolved_mat-Λτm*ρf_test))
            end
            @show(norm(NESS-ρf_evolved_mat))
        else
            ρf_evolved = DP.ψ_init
        end
        println("
        #######################################################################
        Once stationary the state is perturbed with the appropriate 
        spin operator for the calulation.
        #######################################################################")
        
        op_Tensor = op(operator,s[qS[1]])
        if operator == "Sx"
            operator_mat = 0.5*[0 1;1 0]
        elseif operator == "Sy"
            operator_mat = 0.5*im*[0 -1;1 0]
        elseif operator == "Sz"
            operator_mat = 0.5*[1 0 ;0 -1]
        end 
        ϕ = deepcopy(ρf_evolved)
        ϕ_perturbed = apply(op_Tensor,ϕ)
        
        Nsteps = Int(end_time/δt)
        spin_corr_t = complex(zeros(ComplexF64, Nsteps+1))
        spin_corr_t[1] = 0.25
        spin_corr_t_test = deepcopy(spin_corr_t)

        println("
        #######################################################################
        The two MPSs required for the two time correlation are propagated for a time
        τmL, after which their evolution can be described by propagation
        with the fixed propagator.
        #######################################################################")

        # Configure updater parameters
        updater_kwargs = Dict(:ishermitian => P.method_type == "TFCM", :issymmetric => P.method_type == "TFCM", :eager => true)
        normalize = P.method_type == "TFCM"

        #  #enrichment
        #  enrich_bool && (ϕ = enrich_state!(ϕ, P, DP))
        #  enrich_bool && (ϕ_L_greater = enrich_state!(ϕ_L_greater, P, DP;normalise=false))
        #  enrich_bool && (ϕ_R_lesser = enrich_state!(ϕ_R_lesser, P, DP;normalise=false))

        sim_t = 0
        no_of_tdvp_steps = Int(τmL3/δt)

        SvN3 = complex(zeros(no_of_tdvp_steps))
        SvN_perturbed = complex(zeros(no_of_tdvp_steps))
        
        ρ_perturbed_vec = Vector{Any}(undef,Nsteps+1)
        ρ_perturbed_vec[1] = vectorise_ρ(ϕ_perturbed,ϕ,P,DP;rdm_block_sparse_bool = false,kwargs...)

        for i =1:(no_of_tdvp_steps)
            ϕ_perturbed= tdvp(H_MPO,-im * δt,ϕ_perturbed; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=false,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)

            ϕ = tdvp(H_MPO,-im * δt,ϕ; time_step = -im * δt, cutoff = tdvp_cutoff,
            mindim=minbonddim,maxdim=maxbonddim,outputlevel=1, normalize=normalize,updater_kwargs,
            nsite = TDVP_nsite, reverse_step = true)
            
            SvN3[i] = maximum(entanglement_entropy(ϕ))
            SvN_perturbed[i] = maximum(entanglement_entropy(ϕ_perturbed))

            ##using MPS
            ϕ_perturbed2 = apply(op_Tensor,ϕ)
            spin_corr_t[i+1] = inner(ϕ_perturbed2,ϕ_perturbed)

            ##using perturbed reduced density matrix
            ρ_perturbed = vectorise_ρ(ϕ_perturbed,ϕ,P,DP;rdm_block_sparse_bool = false,kwargs...)
            spin_corr_t_test[i+1] = tr(operator_mat*unvectorise_ρ(ρ_perturbed,false))#G_greater-G_lesser
            ρ_perturbed_vec[i+1] = ρ_perturbed

            sim_t += δt
            @show(sim_t)
        end

        ρ_perturbed = ρ_perturbed_vec[no_of_tdvp_steps+1]

        println("
        #######################################################################
        After τmL, ρ_tilde_lesser and ρ_tilde_greater are calculated and are 
        propagated using the converged propagator.
        #######################################################################")

        remaining_steps = Int((end_time-τmL3)/δt)
        for i = 1:remaining_steps
            ρ_perturbed = exp(Lτm*δt)*ρ_perturbed
            spin_corr_t[no_of_tdvp_steps+i+1] = tr(operator_mat*unvectorise_ρ(ρ_perturbed,false))
        end
        
        total_time = range(0,stop=end_time,step=δt)
        output_dict = Dict()
        output_dict["spin_corr_t"] = spin_corr_t
        output_dict["spin_corr_t_test"] = spin_corr_t_test
        output_dict["L_vec"] = L_vec
        output_dict["Λ_vec"] = Λ_vec
        output_dict["SS_prep_obs"] = SS_prep_obs
        partial_energies_bool && (output_dict["partial_energies_vec"] = partial_energies_vec)
        output_dict["map_calc_obs"] = map_calc_obs
        output_dict["ρf_MPS"] = ρf_MPS
        output_dict["ρf_evolved"] = ρf_evolved
        output_dict["ρ_perturbed"] = ρ_perturbed
        output_dict["Structs"] = P,DP
        output_dict["SvN1"] = SvN1
        output_dict["SvN2"] = SvN2
        output_dict["SvN3"] = SvN3
        output_dict["SvN_perturbed"] = SvN_perturbed
        output_dict["τmLs"] = [τmL1,τmL2,τmL3]
        output_dict["ϕ_perturbed"] = ϕ_perturbed
        output_dict["ϕ"] = ϕ
        output_dict["ρ_perturbed_vec"] = ρ_perturbed_vec
        return output_dict
    end
    function rdm_to_MPS_for_qubit(ρ,P,DP;kwargs...)
        """
        -Assuming ρ is in matrix form at this point.
        -This function implements ρ as a schmidt purification using an ancilla for the system,
        assuming ρ is the density matrix of a single qubit and there are no symmetries.
        """
        (;Ns,sys_mode_type) = P
        (;s,qS,qA,qtot,Id) = DP
        
        @assert(Ns==1)
        @assert(sys_mode_type=="S=1/2")

        ##Creates empty MPS, the bath modes are bosonic thermofield modes so are in the thermofield vacuum.
        occs = ["0" for n in qtot]
        therm = MPS(ComplexF64, s, occs)

        ##This removes any numerical noise that can interfere with the diagonalisation. 
        ρ = noise_removal(ρ;kwargs...)
        spec = noise_removal(eigen(ρ).values;kwargs...)
        vecs = noise_removal(eigen(ρ).vectors;kwargs...)

        #Creating the Schmidt states as operators
        op1 = √(spec[1])*(vecs[1,1]*Id[qS[1]]+vecs[2,1]*op("S-",s[qS[1]]))*Id[qA[1]]
        op2 = √(spec[2])*(vecs[1,2]*Id[qS[1]]+vecs[2,2]*op("S-",s[qS[1]]))*op("S-",s[qA[1]])
        op_ = op1+op2

        occs = ["0" for n in qtot]
        therm = MPS(ComplexF64, s, occs)
        therm = apply(op_,therm)
        return therm 
    end


    function ρ_diff_average_fermions(ρ_tilde_lesser_vec,ρ_tilde_greater_vec,Grn_t,NESS_times,end_time,L_vec,site,P)
        (;δt,n,sys_mode_type) = P
        ρ_lesser_diff_average_vec = similar(NESS_times)
        ρ_greater_diff_average_vec = similar(NESS_times)
        corr_diff_average_vec = similar(NESS_times)
        corr_diff_max_vec = similar(NESS_times)
        @showprogress for (i,extrapolation_time) in enumerate(NESS_times)
            start_ind = Int(round(extrapolation_time/P.δt))
            start_L_ind = Int(round(extrapolation_time/(P.n*P.δt)))
            times2 = range(start_ind/10,stop=end_time,step=0.1)
            number_of_modes = Int((NDTensors.dim(DP.s[1])/2)*P.Ns)
            Sz = spin_operators(number_of_modes)[1]
            parity_op = Matrix(Sz[1])
            for i=2:number_of_modes
                parity_op *= Matrix(Sz[i])
            end
            parity_op_vec = lmult(parity_op)
            Lbar_τm = parity_op_vec*L_vec[start_L_ind]*parity_op_vec

        
            ρ_tilde_lesser = ρ_tilde_lesser_vec[start_ind]
            ρ_tilde_greater = ρ_tilde_greater_vec[start_ind]

            ρ_tilde_lesser_vec_test = Vector{Any}(undef,length(total_times))
            ρ_tilde_greater_vec_test = Vector{Any}(undef,length(total_times))
            Grn_t_test = 0*similar(Grn_t)
            Grn_t_test[1:start_ind] = Grn_t[1:start_ind]
            ρ_tilde_lesser_vec_test[1:start_ind] = ρ_tilde_lesser_vec[1:start_ind]
            ρ_tilde_greater_vec_test[1:start_ind] = ρ_tilde_greater_vec[1:start_ind]
            for i=1:length(times2)
                ρ_tilde_lesser = exp(Lbar_τm*P.δt)*ρ_tilde_lesser
                ρ_tilde_greater = exp(Lbar_τm*P.δt)*ρ_tilde_greater
                Grn_greater_test= -im*tr(c_mat[site]*unvectorise_ρ(ρ_tilde_greater,false))
                Grn_lesser_test = im*tr(c_mat[site]*unvectorise_ρ(ρ_tilde_lesser,false))
                Grn_t_test[start_ind+i] = Grn_greater_test - Grn_lesser_test
                ρ_tilde_lesser_vec_test[start_ind+i] = ρ_tilde_lesser
                ρ_tilde_greater_vec_test[start_ind+i] = ρ_tilde_greater
            end
            ρ_lesser_diff = [norm(ρ_tilde_lesser_vec[i]-ρ_tilde_lesser_vec_test[i]) for i=start_ind:length(ρ_tilde_lesser_vec)]
            ρ_greater_diff = [norm(ρ_tilde_greater_vec[i]-ρ_tilde_greater_vec_test[i]) for i=start_ind:length(ρ_tilde_greater_vec)]
            corr_diff = [norm(Grn_t[i]-Grn_t_test[i]) for i=start_ind:length(Grn_t)]
            ρ_lesser_diff_average_vec[i] = sum(ρ_lesser_diff)/(length(ρ_lesser_diff))
            ρ_greater_diff_average_vec[i] = sum(ρ_greater_diff)/(length(ρ_greater_diff))
            corr_diff_average_vec[i] = sum(corr_diff)/(length(corr_diff))
            corr_diff_max_vec[i] = maximum(corr_diff)
        end
        return ρ_lesser_diff_average_vec,ρ_greater_diff_average_vec,corr_diff_average_vec,corr_diff_max_vec
    end

    function ρ_diff_average_SB(ρ_perturbed_vec,spin_corr_t,NESS_times,L_vec,operator_mat,P)
        (;δt,n,sys_mode_type) = P
        ρ_diff_average_vec = similar(NESS_times)
        corr_diff_average_vec = similar(NESS_times)
        corr_diff_max_vec = similar(NESS_times)
        @show(length(NESS_times))
        for (i,extrapolation_time) in enumerate(NESS_times)
            start_ind = Int(round(extrapolation_time/P.δt))
            start_L_ind = Int(round(extrapolation_time/(P.n*P.δt)))
            times2 = range(start_ind/10,stop=end_time,step=0.1)

            Lτm = L_vec[start_L_ind]
            

            ρ_perturbed = ρ_perturbed_vec[start_ind]
            ρ_perturbed_vec_test = Vector{Any}(undef,length(times))
            spin_corr_t_test = 0*similar(spin_corr_t)
            spin_corr_t_test[1:start_ind] = spin_corr_t[1:start_ind]
            ρ_perturbed_vec_test[1:start_ind] = ρ_perturbed_vec[1:start_ind]

            for i=1:length(times2)
                ρ_perturbed = exp(Lτm*P.δt)*ρ_perturbed
                spin_corr_t_test[start_ind+i] = tr(operator_mat*unvectorise_ρ(ρ_perturbed,false))
                ρ_perturbed_vec_test[start_ind+i] = ρ_perturbed
            end
            ρ_diff = [norm(ρ_perturbed_vec[i]-ρ_perturbed_vec_test[i]) for i=start_ind:length(ρ_perturbed_vec)]
            corr_diff = [norm(spin_corr_t[i]-spin_corr_t_test[i]) for i=start_ind:length(spin_corr_t)]
            ρ_diff_average_vec[i] = sum(ρ_diff)/(length(ρ_diff))
            corr_diff_average_vec[i] = sum(corr_diff)/(length(corr_diff))
            corr_diff_max_vec[i] = maximum(corr_diff)
        end
        return ρ_diff_average_vec,corr_diff_average_vec,corr_diff_max_vec
    end

    function extrapolation_vs_map_propagation_spin_boson(ρ_init,L_vec,Λ_vec,memory_time,NESS_times)

        memory_time_ind = Int(memory_time/(P.n*P.δt))

        #ρ_init = vectorise_ρ(DP.ψ_init,P,DP;use_spin_operators=true)
        #ρ_init = [0,0,0,1]#[1 0;0 0]


        ρ_vec1 = Vector{Any}(undef,length(NESS_times)+1)
        σz_vec1 = zeros(length(ρ_vec1))
        σx_vec1= zeros(length(ρ_vec1))
        coherence_vec1 = complex(similar(σz_vec1))
        coherence_vec2 = complex(similar(σz_vec1))

        ρ_vec2 = Vector{Any}(undef,length(NESS_times)+1)
        σz_vec2 = zeros(length(ρ_vec1))
        σx_vec2= zeros(length(ρ_vec1))

        ρ_diff = similar(ρ_vec1)

        ρ_vec1[1] = ρ_init
        ρ_vec2[1] = ρ_init
        ρ_diff[1] = 0
        σz_vec1[1] = 0.5*real(ρ_init[1]-ρ_init[4])
        σx_vec1[1] = 0.5*real(ρ_init[2]+ρ_init[3])
        σz_vec2[1] = 0.5*real(ρ_init[1]-ρ_init[4])
        σx_vec2[1] = 0.5*real(ρ_init[2]+ρ_init[3])
        coherence_vec1[1] = ρ_init[2]
        coherence_vec2[1] = ρ_init[2]
        for i =1:length(NESS_times)
            
            ρ1= Λ_vec[i]*ρ_init
            if NESS_times[i] >=memory_time
                ρ2 = exp(L_vec[memory_time_ind]*P.n*P.δt)*ρ_vec2[i]
            else
                ρ2 = Λ_vec[i]*ρ_init
                
            end

            σz_vec1[i+1] = 0.5*real(ρ1[1]-ρ1[4])
            σx_vec1[i+1] = 0.5*real(ρ1[2]+ρ1[3])
            σz_vec2[i+1] = 0.5*real(ρ2[1]-ρ2[4])
            σx_vec2[i+1] = 0.5*real(ρ2[2]+ρ2[3])
            coherence_vec1[i+1] = ρ1[2]
            coherence_vec2[i+1] = ρ2[2]
            ρ_diff[i+1] = norm(ρ1-ρ2)
            ρ_vec1[i+1] = ρ1
            ρ_vec2[i+1] = ρ2
        end
        Plots.plot([0;NESS_times],σz_vec1,label="σz1")
        Plots.plot!([0;NESS_times],σz_vec2,label="σz2")
        # Plots.plot!([0;NESS_times],σx_vec1,label="σx1")
        # display(Plots.plot!([0;NESS_times],σx_vec2,label="σx2",legend = :right))
        # display(Plots.plot([0;NESS_times],ρ_diff))
        return σz_vec1,coherence_vec1,coherence_vec2,ρ_vec1
    end
        
    function extrapolation_vs_map_propagation_fermions(ρ_init,site,L_vec,Λ_vec,memory_time,NESS_times,P)
        """
        Lm is the converged propagator, where m denotes the memory.
        ρ_in is the input state
        
        """
        (;n,δt,Ns) = P

        memory_time_ind = Int(memory_time/(n*δt))

        
        ρ_vec1 = Vector{Any}(undef,length(NESS_times)+1)
        corr_vec1 = similar(ρ_vec1)
        ρ_vec2 = Vector{Any}(undef,length(NESS_times)+1)
        corr_vec2 = similar(ρ_vec2)
        
        ρ_diff = similar(ρ_vec1)

        ρ_vec1[1] = ρ_init
        ρ_vec2[1] = ρ_init
        ρ_diff[1] = 0

        corr_vec1[1] = ρ_system_corr(unvectorise_ρ(ρ_init,true),Ns,P)
        corr_vec2[1] = ρ_system_corr(unvectorise_ρ(ρ_init,true),Ns,P)

        for i =1:length(NESS_times)
                ##propagate by Δt

            ρ1= Λ_vec[i]*ρ_init
            if NESS_times[i] >=memory_time
                ρ2 = exp(L_vec[memory_time_ind]*n*δt)*ρ_vec2[i]
            else
                ρ2 = Λ_vec[i]*ρ_init
                
            end
            corr1 = ρ_system_corr(unvectorise_ρ(ρ1,true),Ns,P)
            corr2 = ρ_system_corr(unvectorise_ρ(ρ2,true),Ns,P)
            corr_vec1[i+1] = corr1
            corr_vec2[i+1] = corr2
            ρ_vec1[i+1] = ρ1
            ρ_vec2[i+1] = ρ2

        end
        ρ_diff = norm.(ρ_vec1.-ρ_vec2)
        den1 = [real(corr[site,site]) for corr in corr_vec1]
        den2 =[real(corr[site,site]) for corr in corr_vec2]
        Plots.plot([0;NESS_times],den1,label="den1")
        display(Plots.plot!([0;NESS_times],den2,label="den2"))
        display(Plots.plot([0;NESS_times],ρ_diff))
    end


    function extrapolation_vs_map_propagation_with_incremental_maps(ρ_init,L_vec,Λ_vec,dΛ_vec,memory_time,NESS_times)

        memory_time_ind = Int(memory_time/(P.n*P.δt))

        #ρ_init = vectorise_ρ(DP.ψ_init,P,DP;use_spin_operators=true)
        #ρ_init = [0,0,0,1]#[1 0;0 0]


        ρ_vec_Λ,ρ_vec_L,ρ_vec_dΛ= Vector{Any}(undef,length(NESS_times)+1),Vector{Any}(undef,length(NESS_times)+1),Vector{Any}(undef,length(NESS_times)+1)
        σz_vec_Λ,σz_vec_L,σz_vec_dΛ = zeros(length(ρ_vec_Λ)),zeros(length(ρ_vec_Λ)),zeros(length(ρ_vec_Λ))
        σx_vec_Λ,σx_vec_L,σx_vec_dΛ= zeros(length(ρ_vec_Λ)),zeros(length(ρ_vec_Λ)),zeros(length(ρ_vec_Λ))

        ρ_diff_L = similar(ρ_vec_Λ)
        ρ_diff_dΛ = similar(ρ_vec_Λ)

        ρ_vec_Λ[1] = ρ_init
        ρ_vec_L[1] = ρ_init
        ρ_vec_dΛ[1] = ρ_init

        ρ_diff_L[1] = 0
        ρ_diff_dΛ[1] = 0

        σz_vec_Λ[1] = 0.5*real(ρ_init[1]-ρ_init[4])
        σx_vec_Λ[1] = 0.5*real(ρ_init[2]+ρ_init[3])

        σz_vec_L[1] = 0.5*real(ρ_init[1]-ρ_init[4])
        σx_vec_L[1] = 0.5*real(ρ_init[2]+ρ_init[3])

        σz_vec_dΛ[1] = 0.5*real(ρ_init[1]-ρ_init[4])
        σx_vec_dΛ[1] = 0.5*real(ρ_init[2]+ρ_init[3])

        for i =1:length(NESS_times)
            
            ρΛ= Λ_vec[i]*ρ_init
            if NESS_times[i] >=memory_time
                ρL = exp(L_vec[memory_time_ind]*P.n*P.δt)*ρ_vec_L[i]
                ρdΛ = dΛ_vec[memory_time_ind]*ρ_vec_dΛ[i]
            else
                ρL= Λ_vec[i]*ρ_init
                ρdΛ= Λ_vec[i]*ρ_init
            end

            σz_vec_Λ[i+1] = 0.5*real(ρΛ[1]-ρΛ[4])
            σx_vec_Λ[i+1] = 0.5*real(ρΛ[2]+ρΛ[3])

            σz_vec_L[i+1] = 0.5*real(ρL[1]-ρL[4])
            σx_vec_L[i+1] = 0.5*real(ρL[2]+ρL[3])

            σz_vec_dΛ[i+1] = 0.5*real(ρdΛ[1]-ρdΛ[4])
            σx_vec_dΛ[i+1] = 0.5*real(ρdΛ[2]+ρdΛ[3])

            ρ_diff_L[i+1] = norm(ρΛ-ρL)
            ρ_diff_dΛ[i+1] = norm(ρΛ-ρdΛ)

            ρ_vec_Λ[i+1] = ρΛ
            ρ_vec_L[i+1] = ρL
            ρ_vec_dΛ[i+1] = ρdΛ
        end
        Plots.plot([0;NESS_times],σz_vec_Λ,label="σz_Λ",lw=2)
        Plots.plot!([0;NESS_times],σz_vec_L,label="σz_L",lw=2)
        Plots.plot!([0;NESS_times],σz_vec_dΛ,label="σz_dΛ",lw=2,linestyle=:dash)
        Plots.plot!([0;NESS_times],σx_vec_Λ,label="σx_Λ",lw=2)
        Plots.plot!([0;NESS_times],σx_vec_L,label="σx_L",lw=2)
        display(Plots.plot!([0;NESS_times],σx_vec_dΛ,label="σx_dΛ",legend = :right,lw=2,linestyle=:dash))
        Plots.plot([0;NESS_times],ρ_diff_L,label="error for L extrapolation",lw=2)
        display(Plots.plot!([0;NESS_times],ρ_diff_dΛ,label="error for dΛ extrapolation",lw=2))

    end
    function incremental_maps(Λ_vec)
        Λinv_vec = [get_inv(Λ) for Λ in Λ_vec]
        dΛ_vec = similar(Λ_vec)
        dΛ_vec[1] = Λ_vec[1]
        for i=2:length(Λ_vec)
            dΛ_vec[i] = Λ_vec[i]*Λinv_vec[i-1]
        end
        return dΛ_vec
    end





    function markovian_green_function_calculation(L_markovian,Ldag_markovian,ρ_initial,times,sys_site,P;kwargs...)
    
        (;Ns) = P
        
        d = 2^Ns
        Id = 1* Matrix(I, d, d)
        Left_vac = vectorise_mat(Id)
        cdag_mat,c_mat = matrix_operators(Ns,P)
        inds_ = get(kwargs,:inds_,1:Int(d^2))
    
        Greens_function = complex(zeros(length(times)))
        Greens_function_tilde_approach = complex(zeros(length(times)))
        ρ_tilde_lesser_vec = Vector{Any}(undef,length(times))
        ρ_tilde_greater_vec = Vector{Any}(undef,length(times))
       # ρ_markovian_vec = Vector{Any}(undef,length(times))
    
        c_vec = vectorise_mat(c_mat[sys_site])
        ρ_initial_mat = unvectorise_ρ(ρ_initial,true)
        ρ_tilde_greater = (lmult(cdag_mat[sys_site])*ρ_initial)[inds_]
        ρ_tilde_lesser = (rmult(cdag_mat[sys_site])*ρ_initial)[inds_]
        #ρ_init = vectorise_ρ(DP.ψ_init,P,DP;use_spin_operators=true,take_symmetry_subset=false)
        
    
        for (i,t) in enumerate(times)
            c_evolved = exp(Ldag_markovian*t)*c_vec
            G_greater = sum(Left_vac .*(rmult(cdag_mat[sys_site]*ρ_initial_mat)*c_evolved))
            G_lesser = sum(Left_vac .*(rmult(ρ_initial_mat*cdag_mat[sys_site])*c_evolved))
            Greens_function[i] = -im*(G_greater+G_lesser)
            
            ρ_tilde_greater_evolved = exp(L_markovian*t)[inds_,inds_]*ρ_tilde_greater
            ρ_tilde_lesser_evolved = exp(L_markovian*t)[inds_,inds_]*ρ_tilde_lesser
            #ρ_evolved = exp(L_markovian*t)*ρ_initial
            ρ_tilde_lesser_vec[i] = ρ_tilde_lesser_evolved
            ρ_tilde_greater_vec[i] = ρ_tilde_greater_evolved
          #  ρ_markovian_vec[i] = ρ_evolved
            
            G_greater = sum(Left_vac[inds_] .*(rmult(c_mat[sys_site])[inds_,inds_]*ρ_tilde_greater_evolved))
            G_lesser = sum(Left_vac[inds_] .*(rmult(c_mat[sys_site])[inds_,inds_]*ρ_tilde_lesser_evolved))
            Greens_function_tilde_approach[i] = -im*(G_greater+G_lesser)
        end
        return Greens_function_tilde_approach,Greens_function,ρ_tilde_greater_vec,ρ_tilde_lesser_vec
    end

    function markovian_Lindbladian(parity_factor,P,DP)
    
        (;Ns,ϵ,tc,Γ_R,β_R,μ_R,Γ_L,β_L,μ_L) = P
        (;ϵi,ti) = DP
    
        d = 2^Ns
        Id = 1* Matrix(I, d, d)
        Left_vac = vectorise_mat(Id)
    
        ##Defining the creation and annihilation operators
        cdag_mat,c_mat = matrix_operators(Ns,P)
    
        ##Emission and absorption. Factor of 4/π
        ##is due to the different definitions of Γ in our construction 
        ## and the construction used for the master equation 
        Γd_R = (4/π)*Γ_R*(1-thermal_factor(ϵi[end],β_R,μ_R,"Fermion"))
        Γe_R = (4/π)*Γ_R*(thermal_factor(ϵi[end],β_R,μ_R,"Fermion"))
        
        Γd_L = (4/π)*Γ_L*(1-thermal_factor(ϵi[1],β_L,μ_L,"Fermion"))
        Γe_L = (4/π)*Γ_L*(thermal_factor(ϵi[1],β_L,μ_L,"Fermion"))
    
    
        ##Unitary part
        HS = complex(0*similar(cdag_mat[1]))
        for n in 1:Ns
            HS += ϵi[n]*cdag_mat[n]*c_mat[n]
            if n <Ns
                HS += ti[n]*(cdag_mat[n+1]*c_mat[n]+cdag_mat[n]*c_mat[n+1])
            end
        end
        L_unitary = -im*(rmult(HS)- lmult(HS))
    
        ##Dissipative part
        emission_R = Γd_R*(parity_factor*rmult(cdag_mat[end])*lmult(c_mat[end]) - (1/2)*rmult(cdag_mat[end]*c_mat[end]) - (1/2)*lmult(cdag_mat[end]*c_mat[end]))
        absorption_R = Γe_R*(parity_factor*rmult(c_mat[end])*lmult(cdag_mat[end]) - (1/2)*rmult(c_mat[end]*cdag_mat[end]) - (1/2)*lmult(c_mat[end]*cdag_mat[end]))
        emission_L = Γd_L*(parity_factor*rmult(cdag_mat[1])*lmult(c_mat[1]) - (1/2)*rmult(cdag_mat[1]*c_mat[1]) - (1/2)*lmult(cdag_mat[1]*c_mat[1]))
        absorption_L = Γe_L*(parity_factor*rmult(c_mat[1])*lmult(cdag_mat[1]) - (1/2)*rmult(c_mat[1]*cdag_mat[1]) - (1/2)*lmult(c_mat[1]*cdag_mat[1]))
        L_markovian =L_unitary+absorption_R+emission_R+absorption_L+emission_L
    
        ##Creating Ldag
        dag_emission_R = Γd_R*(parity_factor*rmult(c_mat[end])*lmult(cdag_mat[end]) - (1/2)*rmult(cdag_mat[end]*c_mat[end]) - (1/2)*lmult(cdag_mat[end]*c_mat[end]))
        dag_absorption_R = Γe_R*(parity_factor*rmult(cdag_mat[end])*lmult(c_mat[end]) - (1/2)*rmult(c_mat[end]*cdag_mat[end]) - (1/2)*lmult(c_mat[end]*cdag_mat[end]))
        dag_emission_L = Γd_L*(parity_factor*rmult(c_mat[1])*lmult(cdag_mat[1]) - (1/2)*rmult(cdag_mat[1]*c_mat[1]) - (1/2)*lmult(cdag_mat[1]*c_mat[1]))
        dag_absorption_L = Γe_L*(parity_factor*rmult(cdag_mat[1])*lmult(c_mat[1]) - (1/2)*rmult(c_mat[1]*cdag_mat[1]) - (1/2)*lmult(c_mat[1]*cdag_mat[1]))
        Ldag_markovian = -L_unitary+dag_emission_R+dag_absorption_R+dag_emission_L+dag_absorption_L
    
        ρeq = eigen(L_markovian).vectors[:,end]
        trace = sum(Left_vac .*ρeq)
        ρeq = ρeq/trace
        return L_markovian,Ldag_markovian,ρeq,Left_vac
    end

    function SB_lindblad_rates(L_vec)
        #Using formula A7 in https://journals.aps.org/pra/pdf/10.1103/PhysRevA.89.042120

        #For the spin boson, the G matrices are given by the pauli operators and the identity
        G0 = (1/√2)*[1 0;0 1]
        G1 = [1 0;0 -1]
        G2 = [0 1;1 0]
        G3 = [0 -im;im 0]
        G_matrices = [G0,G1,G2,G3]
        decoherence_matrix_vector = Vector{Any}(undef,length(L_vec))
        lindblad_rates_vector = complex(zeros(length(L_vec),3))
        @showprogress for (n,L) in enumerate(L_vec)
            decoherence_matrix = complex(zeros(3,3))
            for i = 1:3
                for j=1:3
                    dij = 0
                    for m=1:4
                        ##apply propagator to Gm
                        op_t = unvectorise_ρ(L*vectorise_mat(G_matrices[m]),false)
                        operator = G_matrices[m]*G_matrices[i+1]*op_t*G_matrices[j+1]
                        dij += tr(operator)
                    end
                    decoherence_matrix[i,j] = dij
                end
            end
            decoherence_matrix_vector[n] = decoherence_matrix
            lindblad_rates_vector[n,:] = eigen(decoherence_matrix).values
        end
        return lindblad_rates_vector
    end

    function calculate_ρ_sys_using_G(corr,DP,P)
        """
        G is a the single particle correlation matrix covering the system modes. 
        ρ = det(1-G)*e^A,
        A = ∑_ij [log(G(1-G)^-1)]_ij cdag[i]c[j].
        log(G(1-G)^(-1))_ij = α_ij
        """
        (;Ns,sys_mode_type,ordering_choice,symmetry_subspace) = P
        (;q) = DP
        G = transpose(corr[q,q])

        if ordering_choice =="interleaved"
            G = interleaved_to_separated(G)
        end
        G = G[1:Ns,1:Ns]

        ###Implements equation connecting the reduced density matrix to the single particle correlation matrix.
        Id = Diagonal(ones(Float64,Ns))    
        α = matrix_log(G*pinv(Id-G))

        A = complex(zeros(2^(Ns),2^(Ns)))
        cdag_mat,c_mat = matrix_operators(Ns,P)

        for (i, creator_i) in enumerate(cdag_mat)
            for (j, annihilator_j) in enumerate(c_mat)
                corr_op = Matrix(creator_i)*Matrix(annihilator_j)
                A += α[i, j] * corr_op
            end
        end
        ρ = det(Id-G)*exp(A)
        return ρ
    end

    function propagate_with_converged_map(τm,Λ_vec,apply_no,NESS_times,site,use_corr_or_Λ,corrs,P,DP)

        """
        This function produces dynamics for τ>τm by using Λ(τm) repeatedly.
        """
    
        (;Ns,U,sys_mode_type) = P
        (;ψ_init) = DP
    
        Δt = NESS_times[1]
        t_end = τm*(apply_no+1)
        Nsteps = Int((t_end)/Δt) 
        t_vec = LinRange(Δt,t_end,Nsteps)
    
    
        ρ_vec = Vector{Any}(undef,Nsteps)
        corr_vec = similar(ρ_vec)
        JR_vec = Vector{Any}(undef,length(t_vec))
        JL_vec,den_vec = similar(JR_vec),similar(JR_vec)
        diag_list = zeros(Nsteps,2^Ns)
    
        
        Λ_vec = [expand_Λ(Λ,Ns) for Λ in Λ_vec]
        index = findfirst(x -> x < 1e-10,abs.(t_vec .-τm))
    
    
    
        Λm = Λ_vec[index]
        ρ_init = vectorise_ρ(ψ_init,P,DP)
        ρ_vec[1] = ρ_init
        ##Initialising the state for times NESS_times[1]:τm
        for (i,Λ) in enumerate(Λ_vec[1:index])
            if use_corr_or_Λ == "corr"
                if U == 0 && sys_mode_type == "Fermion"
                    ρ_out =  transpose(calculate_ρ_sys_using_G(corrs[i],DP,P))
                    ρ_vec[i] = vectorise_mat(ρ_out/tr(ρ_out))
                end
            else    
                ρ_out = Λ*ρ_init
                ρ_vec[i] = vectorise_mat(unvectorise_ρ(ρ_out,true))
            end
        end
    
    
        #the outer loop iterates over the number of applications of the map
        for j = 1:apply_no
            for k = 1:index
                ρ_out = Λm*ρ_vec[(j-1)*index+k]
                ρ_vec[j*index+k] = vectorise_mat(unvectorise_ρ(ρ_out,true))
            end
        end
    
        for (i,ρ) in enumerate(ρ_vec)
    
            diag_list[i,:] = real.(diag(unvectorise_ρ(ρ,true)))
    
            corr = ρ_system_corr(unvectorise_ρ(ρ,true),Ns,P)
            JL_vec[i],JR_vec[i],den_vec[i] = current_operator(corr,site,P,DP)
    
            corr_vec[i] = corr
        end
    
        return t_vec,corr_vec,ρ_vec,JL_vec,JR_vec,den_vec,diag_list
    end


    

        
    function propagation_with_converged_L(ρ_in,Lm,t_start,t_end,Δt,site,P,DP)
        """
        Lm is the converged propagator, where m denotes the memory.
        ρ_in is the input state
        
        """
        (;Ns) = P
        (;times) = DP
        qN = extract_physical_modes(Ns)
        if size(Lm)[1] == length(qN)
            Lm = expand_Λ(Lm,Ns)
        end

        Nsteps = Int((t_end-t_start)/Δt) + 1
        t_vec = LinRange(t_start,t_end,Nsteps)
        ρ_vec = Vector{Any}(undef,Nsteps)
        corr_vec = similar(ρ_vec)
        diag_list = zeros(Nsteps,2^Ns)
        
        corr_vec[1] = ρ_system_corr(unvectorise_ρ(ρ_in,true),Ns,P)
        ρ_vec[1] = ρ_in
        diag_list[1,:] = real.(diag(unvectorise_ρ(ρ_in,true)))
        U = exp(Lm*Δt)
        for i=2:Nsteps
            ##propagate by Δt
            ρ_out = U*ρ_in
            ρ_out = vectorise_mat(unvectorise_ρ(ρ_out,true))
            corr = ρ_system_corr(unvectorise_ρ(ρ_out,true),Ns,P)
            ##save to ρ_vec
            ρ_vec[i] = ρ_out
            corr_vec[i] = corr
            diag_list[i,:] = real.(diag(unvectorise_ρ(ρ_out,true)))
            ##iterate
            ρ_in = ρ_out
        end
        
        JL,JR,den = calculate_currents(corr_vec,site,P,DP)
        return ρ_vec,corr_vec,JL,JR,den,t_vec,diag_list
    end

    function convergence_analysis(L_vec,Λ_vec,P,DP;kwargs...)

        (;Ns) = P
        (;times) = DP
    
        L_NESS_vec = similar(L_vec)
        Λ_NESS_vec = similar(Λ_vec)
        @showprogress for (i,L) in enumerate(L_vec)
            vec = NESS_extraction(L,"L",P,DP;kwargs...)[2]
            mat = unvectorise_ρ(vec,true)
            L_NESS_vec[i] = mat
        end
        @showprogress for (i,Λ) in enumerate(Λ_vec)
            vec = NESS_extraction(Λ,"Λ",P,DP;kwargs...)[2]
            mat = unvectorise_ρ(vec,true)
            Λ_NESS_vec[i] = mat
        end
    
    
        period = 2*π
        period_ind = findmin(abs.(times .-period))[2]
        L_average_vec = similar(L_vec)[1:(length(L_vec)-period_ind)]
        for i =1:length(L_average_vec)
            L_average_vec[i] = sum(L_vec[i:i+period_ind])/(period)
        end
    
    
        dLdt = similar(L_vec)[2:end-1]
        dρLdt = similar(L_vec)[2:end-1]
        dρΛdt = similar(Λ_vec)[2:end-1]
        dL_average_dt = similar(L_average_vec)[2:end-1]
        dΛdt = similar(Λ_vec)[2:end-1]
        dt = times[1]
    
        for i =1:length(dΛdt)
            dΛdt[i] = norm((Λ_vec[i+2]-Λ_vec[i])/(2*dt))
            dρΛdt[i] = norm((Λ_NESS_vec[i+2]-Λ_NESS_vec[i])/(2*dt))
            if i <=length(dLdt)
                dLdt[i] = norm((L_vec[i+2]-L_vec[i])/(2*dt))
    
                dρLdt[i] = norm((L_NESS_vec[i+2]-L_NESS_vec[i])/(2*dt))
            end
            if i <= length(dL_average_dt)
                dL_average_dt[i] = norm((L_average_vec[i+2]-L_average_vec[i])/(2*dt))
            end
        end
        return dLdt,dρLdt,dρΛdt,dL_average_dt,dΛdt
    end



















    function propagate_MPS_comparing_matrix_vs_MPO(boundary_test_bool,P,DP; kwargs...)
        (;n,δt,tdvp_cutoff,compute_maps_bool,minbonddim,maxbonddim,map_as_matrix_or_MPO,Ns) = P
        (;H_MPO,nframe_en,nframe) = DP

        L_vec,Λ_vec,NESS_times = Any[],Any[],Any[]
        Lmat_vec,Λmat_vec = Any[],Any[]
    
        ψ = deepcopy(DP.ψ_init)
        Krylov = Krylov_states(ψ,P,DP)
        ψ[DP.N]= ψ[DP.N]/norm(ψ)
        ψ = enrich_generic3(ψ, Krylov; P);
        ψ[DP.N]= ψ[DP.N]/norm(ψ)

        if P.sys_mode_type == "Fermion"
            obs = Observer("times" => current_time,"corr" => measure_correlation_matrix,"SvN" =>measure_SvN,"Mem" =>measure_mem,"GC" => perform_GC)
        elseif P.sys_mode_type == "S=1/2"
            obs = Observer("times" => current_time,"corr" => measure_spin_correlation_matrix,
            "σz" => measure_pauli_z,"σx" => measure_pauli_x,"σy" => measure_pauli_y,"SvN" =>measure_SvN,"Mem" =>measure_mem,"GC" => perform_GC)
        end
        global sim_t = 0

        for i =1:nframe
            @time ψ = tdvp(H_MPO,-im * n*δt,ψ; time_step = -im * δt, cutoff = tdvp_cutoff,mindim=minbonddim,maxdim=maxbonddim,
                outputlevel=1, normalize=false, (observer!)=obs)    

            global sim_t += n*δt
            @show(sim_t)

            if i <=nframe_en
                println("Time taken for enrichment")
                @time begin
                    Krylov_i = Krylov_states(ψ,P,DP);
                    ψ2 = enrich_generic3(ψ, Krylov_i; P)
                    ψ = ψ2
                end
            end
            if compute_maps_bool == true
                if map_as_matrix_or_MPO == "Matrix"
                    if Ns <= 3
                        println("Time taken for map and louvillian extraction")
                        @time begin
                            MPS_vec = MPS_extraction(ψ,P,DP)
                            L,Λ = compute_Louv(MPS_vec,true,P,DP)
                            push!(L_vec,L)
                            push!(Λ_vec,Λ)
                        end
                    end
                else
                    println("Time taken for map extraction")
                    @time Λ = extract_map_as_MPO(ψ,false,P,DP; kwargs...)
                    push!(Λ_vec,Λ)
                    if Ns <= 3
                        MPS_vec = MPS_extraction(ψ,P,DP)
                        Lmat,Λmat= compute_Louv(MPS_vec,true,P,DP)
                        push!(Lmat_vec,Lmat)
                        push!(Λmat_vec,Λmat)
                    end
                
                end
                push!(NESS_times,sim_t)
            end

            if boundary_test_bool == true
                if sum(boundary_test(ψ,1e-3,DP,P))>0
                    println("boundary reached at t="*string(sim_t))
                    break 
                end
            end
            GC.gc()
        end
        
        corr = obs.corr#res["corr"]
        SvN = obs.SvN#res["SvN"]; 
        
        
        if P.sys_mode_type == "S=1/2"
            σz = obs.σz
            σy = obs.σy
            σx = obs.σx
            return ψ,corr,σz,σy,σx,SvN,NESS_times,L_vec,Λ_vec
        else 
            return ψ,corr,SvN,NESS_times,L_vec,Λ_vec,Lmat_vec,Λmat_vec 
        end
    end


    



function NESS_extraction_through_power_method_comparison(swapping_cutoff,Λ,Λmat,convergence_cutoff,apply_no,DP,P)
    """
    Takes Λ and applies it to the initial state until the output converges up
    to "convergence_cutoff", or its been applied "apply_no" times.
    """


    (;Ns,N_L) = P
    (;ψ_init,s,q) = DP
    ρ_init = efficient_system_state_extraction(DP.ψ_init,P,DP)

    if Ns <= 3
        ρvec_init = vectorise_ρ(ψ_init,P,DP)
        ρ1vec = ρvec_init
    else
        ρvec_init = 0
        ρ1vec = 0
    end

    ρ1 = ρ_init    
    ρNESS = 0
    
    
    try
        Λmat = expand_Λ(Λmat,P.Ns)
    catch 
        Λmat = Λmat
    end
        
    den_op_vec = [op("N",s[i]) for i in q[1:Ns]]
    den_NESS = zeros(apply_no,Ns)
    denvec_NESS = zeros(apply_no,Ns)
    
    @showprogress for i=1:apply_no
        if i == 1
            ρNESS = efficient_map_contraction(swapping_cutoff,ρ1,Λ,false,P,DP)
        else
            ρNESS = efficient_map_contraction(swapping_cutoff,ρ1,Λ,true,P,DP)
        end
        ρNESS = ρNESS/tr(ρNESS)
    
        if Ns <= 3
            ρmatNESS = unvectorise_ρ(Λmat*ρ1vec,true)
            corrNESS = ρ_system_corr(ρmatNESS,Ns,P)
            denvec_NESS[i,:] = real.(diag(corrNESS))
        end
        den_NESS[i,:] = real.([expect_op(ρNESS,den_op,P) for den_op in den_op_vec])

        if norm(ρNESS-ρ1)<convergence_cutoff
            break
        else
            ρ1vec = vectorise_mat(ρmatNESS)
            ρ1 = ρNESS
        end
    end
    denvec_labels = ["From matrix, particle $i" for i = 1:Ns]
    den_labels = ["From MPO, particle $i" for i = 1:Ns]
    
    Plots.plot()
    [Plots.plot!(1:apply_no,real.(den_NESS[:,i]),label = den_labels[i]) for i =1:Ns]
    [display(Plots.plot!(1:apply_no,real.(denvec_NESS[:,i]),label = denvec_labels[i],linestyle=:dash,linewidth=2))
    for i =1:Ns]
    


    ρNESS = ρNESS/tr(ρNESS)

    @assert(typeof(ρNESS)==MPO)
    return ρNESS,denvec_labels,den_labels,denvec_NESS,den_NESS
end




    function time_evolution(boundary_test_bool,P,DP)
        (;sys_mode_type,U) = P
        if sys_mode_type == "Fermion" && U == 0
            corr = propagate_correlations(DP)
            Λ_vec,L_vec = corr_to_Λ(corr,DP,P)
            return corr,L_vec,Λ_vec
        else  
            return propagate_MPS(boundary_test_bool,P,DP)
        end
    end

    function propagate_correlations(DP,P)
        """
        The correlation matrix C_ij = expect(cdag[j]*c[i]) propagates according to
        C_ij(t) =U*C_ij(0)*U', but G_ij = expect(cdag[i]*c[j]) doesn't.This is why 
        the correlation matrices are transposed before calculating the reduced density
        matrix as the formula uses the second definition.
        """
        (;sys_mode_type,bath_mode_type) = P
        (;Ci, H_single,times,N) = DP
        δt = times[2] - times[1]
        
        if sys_mode_type == "Electron"
            @assert(sys_mode_type == bath_mode_type)
            H_single = [H_single zeros(N,N);zeros(N,N) H_single]
        end
        U_step = exp(-im*δt*H_single)
    
        corrs = Vector{Any}(undef,length(times))
        corrs[1] = U_step*Ci*U_step'
        for i in 2:length(times)
            corrs[i] = U_step*corrs[i-1]*U_step'
        end
        return corrs
    end


    function enrich_state!(ψ, P, DP;kwargs...)
        (;method_type) = P
        (;s) = DP
        normalise = get(kwargs,:normalise,true)

        if method_type == "SFMC"
            left_vac = leftvacuum(s)
            Krylov = Krylov_states(ψ, P, DP; ishermitian = false)
            try 
                ψ2 = expand_(ψ, Krylov; cutoff = P.Kr_cutoff)
            catch 
                ψ2 = enrich_generic3(ψ, Krylov; P,normalise=normalise)
            end
            if normalise
                nrm = inner(left_vac, ψ2)
                @show(inner(left_vac, ψ2))
                ψ2 = ψ2 / nrm
            end
        else
            Krylov = Krylov_states(ψ, P, DP)
            try
                ψ2 = enrich_generic3(ψ, Krylov; P,normalise=normalise)
            catch 
                ψ2 = expand_(ψ, Krylov; cutoff = P.Kr_cutoff)
            end
            if normalise
                ψ[DP.N] = ψ[DP.N] / norm(ψ)
            end
        end
        @show(1 - inner(ψ2, ψ))
        ψ = ψ2
        return ψ
    end

    # Define a function to initialize the observer
    function initialise_observer(P, DP, obs)
        if obs != false
            return obs
        else
            obs_map = Dict(
                "TFCM_Fermion_SIAM" => Observer("times" => current_time, "corr" => measure_correlation_matrix, 
                                                "SvN" => measure_SvN, "diag_elements" => measure_SIAM_diag_elements, 
                                                "Mem" => measure_mem, "GC" => perform_GC),
                "TFCM_Fermion" => Observer("times" => current_time, "corr" => measure_correlation_matrix, 
                                                "SvN" => measure_SvN, "GC" => perform_GC),
                "TFCM_Electron" => Observer("times" => current_time, "corr_dn" => measure_spin_dn_electron_correlation_matrix,
                                                "corr_up" => measure_spin_up_electron_correlation_matrix,"SvN" => measure_SvN,
                                                "diag_elements" => measure_SIAM_diag_elements_spinful),
                # "TFCM_S=1/2" => Observer("times" => current_time, "corr" => measure_spin_correlation_matrix, 
                #                         "σz" => measure_pauli_z, "σx" => measure_pauli_x, "σy" => measure_pauli_y, 
                #                         "SvN" => measure_SvN, "Mem" => measure_mem, "GC" => perform_GC),
                "TFCM_S=1/2" => Observer("times" => current_time,"σz" => measure_pauli_z, "σx" => measure_pauli_x, "σy" => measure_pauli_y, 
                                    "SvN" => measure_SvN),
                "SFMC" => Observer("den" => measure_den_SF, "norm" => measure_norm, "times" => current_time)
            )
    
            key = P.method_type == "TFCM" ? join(["TFCM", P.sys_mode_type], "_") : "SFMC"
            # if P.Ns == 2 && P.sys_mode_type == "Fermion"
            #     key = join([key,"SIAM"],"_")
            # end
            return get(obs_map, key, Observer("den" => measure_den_SF, "norm" => measure_norm, "times" => current_time))
        end
    end


    function propagate_MPS_2(P, DP; kwargs...)
        # Parameter unpacking
        (; n, δt, tdvp_cutoff, minbonddim, maxbonddim, map_as_matrix_or_MPO, method_type) = P
        (; H_MPO, nframe_en, nframe, s) = DP
    
        obs = initialise_observer(P, DP, get(kwargs, :obs, false))
        enrich_bool = get(kwargs,:enrich_bool,true)
        TDVP_nsite = get(kwargs,:TDVP_nsite,2)
        boundary_test_bool = get(kwargs,:boundary_test_bool,true)
        compute_maps_bool = get(kwargs,:compute_maps_bool,P.compute_maps_bool)
        partial_energies_bool = get(kwargs,:partial_energies_bool,false)
        partial_energy_length = get(kwargs,:partial_energy_length,1)
        times = get(kwargs,:times,DP.times)
        ψ = get(kwargs,:ψ_init,deepcopy(DP.ψ_init))    
        L_vec,Λ_vec,ρ_vec,NESS_times = Any[],Any[],Any[],Any[]
    
        # Configure updater parameters
        updater_kwargs = Dict(:ishermitian => P.method_type == "TFCM", :issymmetric => P.method_type == "TFCM", :eager => true)
        normalize = P.method_type == "TFCM"
    
        #Normalisation
        (method_type == "TFCM") && (ψ = ψ/norm(ψ))
        (method_type == "SFMC") && (ψ = ψ/inner(leftvacuum(s),ψ))

        ##Enrichment/Grow MPS for single-site TDVP
        TDVP_nsite == 2 && enrich_bool && (ψ = enrich_state!(ψ, P, DP))
        TDVP_nsite == 1 && growMPS!(ψ, minbonddim)
    
        global sim_t = 0
    
        partial_energies_bool && (partial_energies_vec =complex(zeros(length(times))))

        # Main time evolution loop
        for i in 1:length(times)
            @time ψ = ITensorMPS.tdvp(H_MPO, -im * δt, ψ; time_step = -im * δt, cutoff = tdvp_cutoff, 
                                    mindim = minbonddim, maxdim = maxbonddim, outputlevel = 1, 
                                    normalize = normalize, observer! = obs, updater_kwargs, 
                                    nsite = TDVP_nsite, reverse_step = true)
    
            # Normalize state for SFMC
            if P.method_type == "SFMC"
                left_vac = leftvacuum(s)
                nrm = inner(left_vac, ψ)
                @show(inner(left_vac, ψ))
                ψ .= ψ / nrm
                @show(inner(left_vac, ψ))
            end
    
            global sim_t += δt
            @show(sim_t)
    
            partial_energies_bool && (partial_energies_vec[i] = partial_energies(ψ,partial_energy_length,DP))

            if (i % n == 0)
                # Enrichment logic
                if (i / n <= nframe_en) && TDVP_nsite == 2 && enrich_bool
                    println("Time taken for enrichment")
                    @time ψ = enrich_state!(ψ, P, DP)
                end
        
                # Compute maps if needed
                if compute_maps_bool
                    println("Time taken for map extraction")
                    if map_as_matrix_or_MPO == "Matrix"
                        @time begin
                            MPS_vec = MPS_extraction(ψ, P, DP)
                            L, Λ = compute_Louv(MPS_vec, true, P, DP;kwargs...)
                            ρ = vectorise_ρ(ψ, P, DP;kwargs...)
                            push!(ρ_vec, ρ)
                            push!(L_vec, L)
                            push!(Λ_vec, Λ)
                        end
                    else
                        @time Λ = extract_map_as_MPO(ψ, false, P, DP; kwargs...)
                        push!(Λ_vec, Λ)
                    end
                    push!(NESS_times, sim_t)
                end
            end
            # Boundary condition check
            if P.method_type == "TFCM" && boundary_test_bool && sum(boundary_test(ψ,1e-3, DP, P)) > 0
                println("Boundary reached at t = $sim_t")
                break
            end
        end
        if partial_energies_bool
            if compute_maps_bool
                return  ψ,obs,L_vec,Λ_vec,ρ_vec,NESS_times,partial_energies_vec
            else
                return ψ, obs,partial_energies_vec
            end
        else
            if compute_maps_bool
                return  ψ,obs,L_vec,Λ_vec,ρ_vec,NESS_times
            else
                return ψ, obs
            end
        end
    end

    function propagate_MPS(boundary_test_bool,P,DP; kwargs...)
        (;n,δt,tdvp_cutoff,compute_maps_bool,minbonddim,maxbonddim,map_as_matrix_or_MPO) = P
        (;H_MPO,nframe_en,nframe) = DP

        L_vec,Λ_vec,ρ_vec,NESS_times = Any[],Any[],Any[],Any[]

        ψ = deepcopy(DP.ψ_init)
        Krylov = Krylov_states(ψ,P,DP)
        
        ψ = enrich_generic3(ψ, Krylov; P);
        
        ψ[DP.N]= ψ[DP.N]/norm(ψ)

        if P.sys_mode_type == "Fermion"
            if P.Ns == 2
                obs = Observer("times" => current_time,"corr" => measure_correlation_matrix,"SvN" =>measure_SvN,
                "diag_elements" => measure_SIAM_diag_elements,"Mem" =>measure_mem,"GC" => perform_GC)
            else
                obs = Observer("times" => current_time,"corr" => measure_correlation_matrix,"SvN" =>measure_SvN,"GC" => perform_GC)
            end
        elseif P.sys_mode_type == "S=1/2"
            obs = Observer("times" => current_time,"corr" => measure_spin_correlation_matrix,
            "σz" => measure_pauli_z,"σx" => measure_pauli_x,"σy" => measure_pauli_y,"SvN" =>measure_SvN,"Mem" =>measure_mem,"GC" => perform_GC)
        end
        global sim_t = 0

        for i =1:nframe
            @time ψ = tdvp(H_MPO,-im * n*δt,ψ; time_step = -im * δt, cutoff = tdvp_cutoff,mindim=minbonddim,maxdim=maxbonddim,
                outputlevel=1, normalize=false, (observer!)=obs)    

            global sim_t += n*δt
            @show(sim_t)

            if i <=nframe_en
                println("Time taken for enrichment")
                @time begin
                    Krylov_i = Krylov_states(ψ,P,DP);
                    ψ2 = enrich_generic3(ψ, Krylov_i; P)
                    ψ = ψ2
                end
            end
            if compute_maps_bool == true
                if map_as_matrix_or_MPO == "Matrix"
                    println("Time taken for map and louvillian extraction")
                    @time begin
                        MPS_vec = MPS_extraction(ψ,P,DP)
                        L,Λ = compute_Louv(MPS_vec,true,P,DP)
                        ρ = vectorise_ρ(ψ,P,DP)  
                        push!(ρ_vec,ρ)
                        push!(L_vec,L)
                        push!(Λ_vec,Λ)
                    end
                else
                    println("Time taken for map extraction")
                    @time Λ = extract_map_as_MPO(ψ,false,P,DP; kwargs...)
                    push!(Λ_vec,Λ)
                end
                push!(NESS_times,sim_t)
            end

            if boundary_test_bool == true
                if sum(boundary_test(ψ,1e-3,DP,P))>0
                    println("boundary reached at t="*string(sim_t))
                    break
                end
            end
            GC.gc()
        end
        
        corr = obs.corr#res["corr"]
        SvN = obs.SvN#res["SvN"]; 
        if P.Ns == 2
            diag_elements = obs.diag_elements
        end
        
        if P.sys_mode_type == "S=1/2"
            σz = obs.σz
            σy = obs.σy
            σx = obs.σx
            return ψ,corr,σz,σy,σx,SvN,NESS_times,L_vec,Λ_vec
        else 
            if P.Ns == 2
                return ψ,corr,SvN,NESS_times,L_vec,Λ_vec,diag_elements,ρ_vec
            else  
                return ψ,corr,SvN,NESS_times,L_vec,Λ_vec,ρ_vec
            end
        end
    end

    



    function MPS_extraction(ψ,P,DP;kwargs...)

        (;δt,tdvp_cutoff,minbonddim,maxbonddim,method_type) = P
        (;H_MPO) = DP
        ψ1 = deepcopy(ψ)

        # Configure parameters
        updater_kwargs = Dict(:ishermitian => method_type == "TFCM", :issymmetric => method_type == "TFCM", :eager => true)
        normalize = method_type == "TFCM"
        TDVP_nsite = get(kwargs,:TDVP_nsite,2)

        ψ_prev = ITensorMPS.tdvp(H_MPO, im * δt, ψ; time_step = im * δt, cutoff = tdvp_cutoff, 
        mindim = minbonddim, maxdim = maxbonddim, outputlevel = 1, 
        normalize = normalize, updater_kwargs, 
        nsite = TDVP_nsite, reverse_step = true)

        ψ_next = ITensorMPS.tdvp(H_MPO, -im * δt, ψ; time_step = -im * δt, cutoff = tdvp_cutoff, 
        mindim = minbonddim, maxdim = maxbonddim, outputlevel = 1, 
        normalize = normalize, updater_kwargs, 
        nsite = TDVP_nsite, reverse_step = true)

       
        MPS_vector = ψ_prev,ψ1,ψ_next
        return MPS_vector
    end

    function map_extraction(MPS_vec,perturb_bool,P,DP)    
        
        Lτ,Λτ = compute_Louv(MPS_vec,true,P,DP)
        Λτ_δtL,Λτ_δtR  = [],[]
        if perturb_bool
            Λτ_δtL,Λτ_δtR = single_bath_map_extraction(ψ,P,DP)
        end
        maps = [Λτ,Lτ,Λτ_δtL,Λτ_δtR]
        return maps
    end













    "Λ and L extraction from correlation matrix, only valid for quadratic systems
    ---------------------------------------------------------------------------------------------------------------------------------------"

    
    function calculate_ρ_using_G(corr,DP,P)
        """
        G is a the single particle correlation matrix covering the system and ancilla modes. 
        ρ = det(1-G)*e^A,
        A = ∑_ij [log(G(1-G)^-1)]_ij cdag[i]c[j].
        log(G(1-G)^(-1))_ij = α_ij

        This is defined via the system and ancilla modes being in separated ordering. In this ordering,
        the initial correlation matrix of the choi state with have minus signs due to the long range correlations 
        in this ordering. Then the spin particle hole transform can be performed to get the map. If the initial
        correlation matrix is defined for an interleaved setup then there won't be these minus signs, and using the spin 
        particle hole transform will give the wrong answer. If the fermionic particle hole transform is used, this is corrected for,
        but I'm not sure why. In summary, the old initial state gate function should be used with a spin 
        particle hole transform. Currently, using the initial_state_gates_separated
        and the fermionic particle hole transform will work, but is wrong.  As the interleaved_to_separated function
        is equivalent to a set of qubit swap gates, the phase correction isn't needed.

        """
        (;Ns,sys_mode_type,ordering_choice,symmetry_subspace) = P
        (;q) = DP
        G = transpose(corr[q,q])

        qA =  Ns+1:2*Ns
        qS =  1:Ns
        if ordering_choice =="interleaved"
            G = interleaved_to_separated(G)
        end

        if sys_mode_type == "Fermion"
            ###Implements equation connecting the reduced density matrix to the single particle correlation matrix.
            Id = Diagonal(ones(Float64,2*Ns))    
            α = matrix_log(G*pinv(Id-G))

            A = complex(zeros(2^(2*Ns),2^(2*Ns)))
            cdag_mat,c_mat = matrix_operators(2*Ns,P)

            for (i, creator_i) in enumerate(cdag_mat)
                for (j, annihilator_j) in enumerate(c_mat)
                    corr_op = Matrix(creator_i)*Matrix(annihilator_j)
                    A += α[i, j] * corr_op
                end
            end

            ρ = det(Id-G)*exp(A)

            # if ordering_choice == "interleaved"
            #     ##swap phases
            #     gates =  ancilla_phase_gate_swap(qA,false,DP,P)
            #     ρ = apply_gates_to_ρmat(ρ,gates)
            # end
            # #PH phases
            # if isodd(Ns)
            #     gates = ancilla_phase_gate_PH(qA,false,DP,P)
            #     ρ = apply_gates_to_ρmat(ρ,gates)
            # end
        else
            ρ = G
        end
        ##PH transform
        gates =  particle_hole_transform(qA,false,DP,P)
        ρ = apply_gates_to_ρmat(ρ,gates)
        ρ_test(ρ,1e-5)
        Λ = ρ_to_Λ(ρ,Ns)
        if symmetry_subspace =="Number conserving"
            qN = extract_physical_modes(Ns)
            Λ = Λ[qN,qN]
        end
        return Λ
    end

   function corr_to_Λ(corrs,DP,P)

    Λ_vec = similar(corrs)
    println("Calculating Λ(t) from the single particle correlation matrix.")
    @showprogress for i =1:length(corrs)
         Λ_vec[i]  = calculate_ρ_using_G(corrs[i],DP,P)
    end

    L_vec = Λs_to_Louv(Λ_vec,P,DP)[1]
    return Λ_vec,L_vec
end




    function N_superoperator(Ns)
        d = 2^Ns
        Id = Diagonal(ones(Float64,d))
        Num = spin_operators(Ns)[4]
        NS = kronecker(Id,sum(Num)) - kronecker(sum(Num),Id)
        return NS
    end


    function apply_gates_to_ρmat(ρ,gates)
        for i=1:length(gates)    
            ρ = gates[i]*ρ*gates[i]'
        end
        return ρ
    end

    function ρ_to_Λ(ρ,Ns)
        d = 2^Ns
        Λ = zeros(ComplexF64, d^2, d^2) 
        for i_s=1:d, j_s=1:d
            for i_a=1:d, j_a=1:d  
                Λ[(i_s-1)*d + j_s, (i_a-1)*d + j_a] = 
                    conj(d*ρ[(i_s-1)*d + i_a, (j_s-1)*d + j_a])
            end
        end
        return Λ
    end





















    "Λ and L extraction from the MPS as a full Tensor. 
    This can only be applied to small systems
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    

    function NESS_calculations(maps,site,Λ_or_L,P,DP;kwargs...)
        """
        This functions takes a list of maps in matrix form and for each entry
        it calculates the spectrum, the NESS state in vector form, the NESS state in matrix form,
        the NESS currents and density, and the convergence of the NESS state using trace distance between
        consecutive NESS states. 
        """
        (;Ns,sys_mode_type) = P
        (;s) = DP
        d = size(maps[1])[1]
        calculate_NESS_observables = get(kwargs,:calculate_NESS_observables,true)
        site_dimension = NDTensors.dim(s[1])
        spectra = complex(zeros(length(maps),d)) 
        corr_list = Any[]
        NESS_list = Any[]
        diag_list = zeros(length(maps),site_dimension^Ns)
        vec = complex(zeros(site_dimension^(2*Ns)))
    
        println("Doing NESS calculations.")
        @showprogress for
             i =1:length(maps)
            ##extracting NESS state
            spec,vec = NESS_extraction(maps[i],Λ_or_L,P,DP;kwargs...)
            mat = unvectorise_ρ(vec,true)
            spectra[i,:] = spec
            push!(NESS_list,mat)
            if Ns == 2 && sys_mode_type == "Fermion"
                diag_list[i,:] = real.(diag(mat))
            end
            
            ##Calculating correlation matrix for NESS state
            if calculate_NESS_observables
                corr_NESS = ρ_system_corr(mat,Ns,P;kwargs...)
                push!(corr_list,corr_NESS)
            end
            
        end
        if calculate_NESS_observables
            if sys_mode_type == "S=1/2"
                σx_NESS,σy_NESS,σz_NESS = calculate_spin_values(NESS_list)
                return spectra,NESS_list,σx_NESS,σy_NESS,σz_NESS
            else
                JL_NESS_l,JR_NESS_l,den_NESS_l = calculate_currents(corr_list,site,P,DP)
    
                # if Ns == 2 && sys_mode_type == "Fermion"
                #     return spectra,JL_NESS_l,JR_NESS_l,den_NESS_l,diag_list
                # else
                return spectra,NESS_list,JL_NESS_l,JR_NESS_l,den_NESS_l
                
            end
        else 
            return spectra,NESS_list
        end
    end


    function kth_mode_extraction(k,map,Λ_or_L,P)
        """
        k=1 is the NESS state, k=2 is the slowest decaying mode etc.
        """

        (;symmetry_subspace,Ns) = P

        spec = eigen(map).values
        vecs = eigen(map).vectors
        spec = map_to_principal.(spec)

        vec = complex(zeros(2^(2*Ns)))
        qN = extract_physical_modes(Ns)

        if Λ_or_L == "Λ"
            sorted_indx = reverse(sortperm(real.(spec)))
            ind = sorted_indx[k]
        elseif Λ_or_L == "L"
            sorted_indx = sortperm(abs.(spec))
            ind = sorted_indx[k]
        end

        if symmetry_subspace == "Number conserving"
            vec[qN] = vecs[:,ind]
        else
            vec = vecs[:,ind]
        end
        return spec,vec,spec[ind],vecs[:,sorted_indx]
    end



    # function NESS_extraction(map,Λ_or_L,P;kwargs...)
    #     (;symmetry_subspace,Ns) = P

    
    #     spec = eigen(map).values
    #     vecs = eigen(map).vectors
    #     spec = map_to_principal.(spec)
    
    #     vec = complex(zeros(2^(2*Ns)))
    #     qN = extract_physical_modes(Ns)
    
    #     if Λ_or_L == "Λ"
    #         ind = argmax(real.(spec))
    #     elseif Λ_or_L == "L"
    #         ind = argmin(abs.(spec))
    #     end
    
    #     if symmetry_subspace != "Full"
    #         vec[qN] = vecs[:,ind]
    #     else
    #         vec = vecs[:,ind]
    #     end
    #     return spec,vec
    # end
    function NESS_extraction(map,Λ_or_L,P,DP;kwargs...)
        (;symmetry_subspace,Ns) = P
        (;s) = DP
        take_symmetry_subset = get(kwargs,:take_symmetry_subset,true)
        site_dimension = NDTensors.dim(s[1])
    
        spec = eigen(map).values
        vecs = eigen(map).vectors
        spec = map_to_principal.(spec)
    
        vec = complex(zeros(site_dimension^(2*Ns)))
        qN = extract_physical_modes(Ns)
    
        if Λ_or_L == "Λ"
            ind = argmax(real.(spec))
        elseif Λ_or_L == "L"
            ind = argmin(abs.(spec))
        end
    
        if symmetry_subspace != "Full" && take_symmetry_subset
            vec[qN] = vecs[:,ind]
        else
            vec = vecs[:,ind]
        end
        return spec,vec
    end

    function unvectorise_ρ(ρvec,tr_bool)

        d =  Int(sqrt(length(ρvec)))
        ρ = complex(zeros(d,d))
        for i =1:d
            for j=1:d
                ρ[j,i] = ρvec[Int((i-1)*d +j)]
            end
        end
        if tr_bool
            ρ = ρ/tr(ρ) ##ensures correct normalisation
        end
        return ρ
    end





















    "Λ and L extraction from the MPS as an MPO. 
    #This is a poor implementation as the ancilla modes 
    #are separated from their entangled system modes, causing 
    #a rainbow state and a large bond dimension
    ------------------------------------------------------------------------------------------"



    function NESS_extraction_through_power_method(Λ,swapping_cutoff,convergence_cutoff,apply_no,DP,P)
        """
        Takes Λ and applies it to the initial state until the output converges up
        to "convergence_cutoff", or its been applied "apply_no" times.
        """
        
        
        (;Ns) = P
        (;ψ_init) = DP
        
        ρ_init = efficient_system_state_extraction(ψ_init,P,DP)
        ρ1 = ρ_init
        ρNESS = 0

        @showprogress for i=1:apply_no
            ρNESS = efficient_map_contraction(swapping_cutoff,ρ1,Λ,true,P,DP)
            ρNESS = ρNESS/tr(ρNESS)
            
            if norm(ρNESS-ρ1)<convergence_cutoff
                break
            else
                ρ1 = ρNESS
            end
        end
        ρNESS = ρNESS/tr(ρNESS)
        
        @assert(typeof(ρNESS)==MPO)
        return ρNESS
    end
    
    function NESS_MPO_calculations(Λ_vec,site,swapping_cutoff,convergence_cutoff,apply_no,DP,P)
        (;N_L) = P
        (;s,qS,qA) = DP
        
        """
        This converts the index from interleaved ordering to separated
        """
        
        ind = indexin(site,qS)[1]
        den_op = op("N",s[ind+2*N_L])
        
        den_NESS = []
        for i = 1:length(Λ_vec)
            ρNESS = NESS_extraction_through_power_method(Λ_vec[i],swapping_cutoff,convergence_cutoff,apply_no,DP,P)
            den = expect_op(ρNESS,den_op,P)
            push!(den_NESS,den)
        end
        return den_NESS
    end



















    "Various helper functions
    -----------------------------------------------------------------------------------"
   

    function calculate_spin_values(NESS_list)
        σx = []
        σy = []
        σz = []
        σx_op = [[0,1] [1,0]]
        σy_op = [[0,im] [-im,0]]
        σz_op = [[1,0] [0,-1]]
        for NESS in NESS_list
            push!(σx,tr(σx_op*NESS))
            push!(σy,tr(σy_op*NESS))
            push!(σz,tr(σz_op*NESS))
        end
        return σx,σy,σz
    end

    function Λs_to_Louv(Λ,P,DP)
        Ls = similar(Λ)[1:(end-2)]
        dΛτdt_list = similar(Ls)
        n = length(Ls)
        for i =1:n
            Λs = [Λ[i],Λ[i+1],Λ[i+2]]
            L,dΛτdt = compute_Louv(Λs,false,P,DP)
            Ls[i],dΛτdt_list[i] = L,dΛτdt
        end
        return Ls,dΛτdt_list
    end


    function compute_Louv(Λs,ITensor_bool,P,DP;kwargs...)
        (;δt,symmetry_subspace) = P
        if ITensor_bool
            Λs = [NESS_fn(Λs[1], P, DP;kwargs...)[2], 
                  NESS_fn(Λs[2], P, DP;kwargs...)[2],
                  NESS_fn(Λs[3],P,DP;kwargs...)[2]]
        end

        take_symmetry_subset = get(kwargs,:take_symmetry_subset,true)

        dΛτdt = compute_dΛdt(Λs,δt) 
        Λ_inv = get_inv(Λs[2])
        L =  dΛτdt*Λ_inv
        qN = extract_physical_modes(P.Ns)
        if take_symmetry_subset && symmetry_subspace == "Number conserving" && size(Λs[2])[1] != length(qN)
            Λs[2] = Λs[2][qN,qN]
            L = L[qN,qN]
        end

        return L,Λs[2]
    end

    function reduced_L_extraction(d,Λ_vec,δt)
        """
        d is the number of dimensions of the space we want to project out.
        For example, if Λ has n fast decaying modes and m other modes, Λ will become  a 
        rank m projector. Inverting this will become numerically unstable, to overcome this
        we manually remove the n fast decaying modes and only work in reduced space of dimension m.
        
        Dependencies: compute_dΛdt, get_inv
        
        Returns both spectra and the resulting propagators in the 
        basis of the corresponding map.
        """
    
        L_vec_reduced = Vector{Any}(undef,length(Λ_vec)-2)
        spectra_L_reduced = complex(zeros(length(L_vec_reduced),6-d)) 
        spectra_Λ_reduced = similar(spectra_L_reduced)
    
        for i =2:length(Λ_vec)-1
    
            Λs = [Λ_vec[i-1],Λ_vec[i],Λ_vec[i+1]]
            dΛτdt = compute_dΛdt(Λs,δt) 
            vals,U = eigen(Λs[2]).values,eigen(Λs[2]).vectors
            Λ_diag = diagm(vals)
            Λ_diag_reduced = Λ_diag[d+1:6,d+1:6]
            dΛτdt_D = inv(U)*dΛτdt*U
            L = dΛτdt_D[d+1:6,d+1:6]*get_inv(Λ_diag_reduced)
    
            spec_L = eigen(L).values
            spec_L = map_to_principal.(spec_L)
            spectra_L_reduced[i-1,:] = spec_L
    
            spectra_Λ_reduced[i-1,:] = map_to_principal.(vals[d+1:6])
            L_vec_reduced[i-1] = L
        end
        return spectra_L_reduced,spectra_Λ_reduced,L_vec_reduced
    end

    function ρ_system_corr(ρ,Ns,P;kwargs...)
        """
        Note that at this point, only system sites remain. 
        """
        cdag_mat,c_mat = matrix_operators(Ns,P)
        corr = complex(zeros(Ns,Ns))
        for i=1:Ns
            for j=1:Ns
                corr[i,j] = tr(ρ*cdag_mat[j]*c_mat[i])
            end
        end
        return corr
    end



    function Λs_to_dΛdt(Λ,δt)
        dΛτdt_list = similar(Λ)[2:end-1]
        n = length(dΛτdt_list)
        for i =1:n
            Λs = [Λ[i],Λ[i+1],Λ[i+2]]
            dΛτdt = compute_dΛdt(Λs,δt)
            dΛτdt_list[i] = dΛτdt
        end
        return dΛτdt_list
    end

    function extract_physical_modes(Ns)
        diag_vals = diag(N_superoperator(Ns))
        qN = findall(x->x==0,diag_vals)
        return qN
    end


    function expand_Λ(Λ,Ns)
        qN = extract_physical_modes(Ns)
        Nλ = 2^(2*Ns)
        A = complex(zeros(Nλ,Nλ))
        A[qN,qN] = Λ
        return A
    end


    function plot_currents(t,xlim,JL,JR,JL_NESS,JR_NESS,den,den_NESS,P,DP)
        (;D) = P
        (;ϵi) = DP
        fac = 100/D
        J_LB,_,den_LB = LB_current(P,DP)
        J_LB = J_LB*ones(length(t))
        den_LB = den_LB*ones(length(t))
        # println("Save current plot? [y/n]")
        # save_choice = readline()

        plot(xlim=xlim)
        plot!(t,real.(fac*JL),label="JL")
        plot!(t,real.(fac*JL_NESS),label="JL NESS")
        plot!(t,real.(fac*JR),label="JR")
        plot!(t,real.(fac*JR_NESS),label="JR NESS")
        display(plot!(t,real.(fac*J_LB),label="LB result",
                ylabel="Particle current Jp/D (×100)",xlabel="Time",
                title="impurity energy="*string(ϵi[1])))
        # if save_choice =="y"
        #     println("Filename:")
        #     name = readline()
        #     savefig(name)
        # end

        # println("Save density plot? [y/n]")
        # save_choice = readline()
        plot(xlim=xlim)
        plot!(t,real.(den),label="density")
        plot!(t,real.(den_NESS),label="NESS density")
        display(plot!(t,real.(den_LB),label="LB result",xlabel="Time",ylabel="density"))
        # if save_choice =="y"
        #     println("Filename:")
        #     name = readline()
        #     savefig(name)
        # end
    end

    function plot_spectra(t,spec_l,xlim)

        # println("Save real spectral plot? [y/n]")
        # save_choice = readline()
        display(plot(t,real.(spec_l),title="Spectrum (real part)",label =false,xlabel="Time",xlim=xlim))
        # if save_choice =="y"
        #     println("Filename:")
        #     name = readline()
        #     savefig(name)
        # end

        # println("Save imaginary spectral plot? [y/n]")
        # save_choice = readline()
        display(scatter(t,imag.(spec_l),title="Spectrum (imaginary part)",label = false,xlabel="Time",xlim=xlim))
        # if save_choice =="y"
        #     println("Filename:")
        #     name = readline()
        #     savefig(name)
        # end
        display(plot(t,log10.(abs.(real.(spec_l))),title="log of spectrum (real part)",label =false,xlim=xlim))
        display(plot(t,log10.(abs.(imag.(spec_l))),title=title="log of spectrum (imaginary part)",label = false,xlim=xlim))
    end





















    "Current calculations
    ---------------------------------------------------------------------------------------------------------------"
    
    function calculate_currents(corr,site,P,DP)
        """
        Returns the currents and density for the site chosen, calculated
        from the correlation matrix.
        """
        JL_list = []
        JR_list = []
        den_list = []
        println("Calculating exact currents.")
        @showprogress for i=1:length(corr)
            JL,JR,den = current_operator(corr[i],site,P,DP)
            push!(JL_list,JL)
            push!(JR_list,JR)
            push!(den_list,den)
        end
        return JL_list,JR_list,den_list
    end


    function current_operator(corr,site,P,DP)
        """
        Calculates the currents and density for the site chosen. There are two cases for this,
        corr = N by N matrix for the entire system+baths, or corr=Ns by Ns for just the system modes.
        """
        (;Ns,N_L,N_R,sys_mode_type) = P
        (;N,qS,H_single) = DP



        ind = indexin(site,qS)[1]
        corr_length = size(corr)[1]
        if corr_length == Ns
            corr_qS = 1:Ns
        elseif corr_length == 2*Ns
            corr_qS = 1:2:2*Ns
        elseif corr_length == N
            corr_qS = qS
        end

        if N_L > 0
            if ind != 1   
                t1 = H_single[qS[ind-1],site]
                JL = im*(t1*corr[corr_qS[ind],corr_qS[ind-1]] - conj(t1)*corr[corr_qS[ind-1],corr_qS[ind]])
            else
                if corr_length == N
                    JL = left_boundary_current(corr,site,DP)
                else
                    side = "left"
                    JL = 0#perturb_J_v2(Λτ,Λτ_δtL,site,side,P,DP)
                end
            end 
        else 
            JL = 0
        end
        if N_R >0 
            if ind != Ns
                t1 = H_single[site,qS[ind+1]]
                JR = im*(t1*corr[corr_qS[ind+1],corr_qS[ind]] - conj(t1)*corr[corr_qS[ind],corr_qS[ind+1]])
            else
                if corr_length == N
                    JR = right_boundary_current(corr,site,DP)
                else 
                    side = "right"
                    JR = 0#perturb_J_v2(Λτ,Λτ_δtR,site,side,P,DP)
                end
            end
        else
            JR = 0
        end
        if sys_mode_type =="Fermion"
            n = corr[corr_qS[ind],corr_qS[ind]]
        elseif sys_mode_type == "S=1/2"
            n = corr[corr_qS[ind],corr_qS[ind]]-0.5
        end
        return JL,JR,n
    end



    function left_boundary_current(corr,site,DP)
        """
        Special case of the boundary where current comes from 
        both the filled and empty chains.
        """
        (;H_single) = DP
        t1,t2 = H_single[site-2,site],H_single[site-1,site]
        JB_L = t1*corr[site,site-2] - conj(t1)*corr[site-2,site]
        JA_L = t2*corr[site,site-1] - conj(t2)*corr[site-1,site]
        JL = im*(JB_L + JA_L)
        return JL
    end
    function right_boundary_current(corr,site,DP)
        """
        Special case of the boundary where current comes from 
        both the filled and empty chains.
        """
        (;H_single) = DP
        t1,t2 = H_single[site+2,site],H_single[site+3,site]
        JB_R = t1*corr[site,site+2] - conj(t1)*corr[site+2,site]
        JA_R = t2*corr[site,site+3] - conj(t2)*corr[site+3,site]
        JR = -im*(JB_R + JA_R)
        return JR
    end

    function LB_current_general(P,DP)
        """
        Calculate the particle current Jp, energy current Je, and density n of a system 
        using the Landauer-Büttiker formalism.
        """
    
        (;Γ_L,Γ_R,μ_L,μ_R,β_L,β_R,Ns,spec_fun_type,D,eta,N_L,N_R) = P
        (;ϵi,ti,qS,q,H_single) = DP
    
    
        wsamp = 10000; # Frequency sampling.
        w = range(-10*D,10*D,wsamp); # Frequency axis for Landauer calculations (slightly larger than the band).
        dw = w[2] - w[1]; # Frequency increment.
    
        JR = spectral_function(w,0,"right",P)
        JL = spectral_function(w,0,"left",P)
        Δ_L_test = -im*π*hilbert(JL)
        Δ_R_test = -im*π*hilbert(JR)
    
    
        if spec_fun_type == "box"
            ρ = (1/(2*D))*(heaviside(w .+ D) .- heaviside(w .- D))
            Δ_L = (-Γ_L*D/(2*pi*D))*log.((w.-D .+im*eta)./(w .+D .+im*eta))
            Δ_R = (-Γ_R*D/(2*pi*D))*log.((w.-D .+im*eta)./(w .+D .+im*eta))
            #Λ = -im*D*Γ*hilbert(ρ)
    
        elseif spec_fun_type =="ellipse"
            ρ = (2/(π*D))*real(sqrt.(Complex.(1 .-(w/D).^2)))
    
            Δ_L = -im*D*Γ_L*hilbert(ρ)
            Δ_R = -im*D*Γ_R*hilbert(ρ)
        end
    
        M = complex(zeros(Ns,Ns))
        transmission_fn = complex(zeros(length(w)))
        density_fn = complex(zeros(length(w)))
    
        coupling_factor = prod(abs.(ti).^2)
        HS = H_single[qS,qS]
        detM_vec = []
        for (i,E) in enumerate(w)
            M = E*I - HS
            M[1,1] += -Δ_L[i]
            M[Ns,Ns] += -Δ_R[i]
            detM = det(M)
            push!(detM_vec,detM)
            transmission_fn[i] = coupling_factor*4*(π^2)*JL[i]*JR[i]/(abs(detM)^2)
            #density_fn[i] = 0.5*coupling_factor*4*(π^2)*(JL[i]+JR[i])/(abs(detM)^2)
            
        end
        
        f_L = 1 ./ (1 .+exp.((w .- μ_L)*β_L)); 
        f_R = 1 ./ (1 .+exp.((w .- μ_R)*β_R));
    
    
        # Compute the particle-current, energy-current and density:
        Jp = (1/(2*π))*sum(transmission_fn.*(f_L - f_R))*dw;
        Je = (1/(2*π))*sum(w.*transmission_fn.*(f_L - f_R))*dw;
        #n = (1/(2*π))*sum(density_fn.*(f_L + f_R))*dw;
        
        display(Plots.plot(w,real.(density_fn)))
        return [Jp,Je,n] 
    end
    

    function LB_current(P,DP)
        """
        Calculate the particle current Jp, energy current Je, and density n of a system 
        using the Landauer-Büttiker formalism.
        """

        (;Γ_L,Γ_R,μ_L,μ_R,β_L,β_R,Ns,spec_fun_type,D,eta) = P
        (;ϵi,ti) = DP


        wsamp = 10000; # Frequency sampling.
        w = range(-10*D,10*D,wsamp); # Frequency axis for Landauer calculations (slightly larger than the band).
        Γ = Γ_L+Γ_R
        dw = w[2] - w[1]; # Frequency increment.
        if spec_fun_type == "box"
            ρ = (1/(2*D))*(heaviside(w .+ D) .- heaviside(w .- D))
            Δ_L = (-Γ_L*D/(2*pi*D))*log.((w.-D .+im*eta)./(w .+D .+im*eta))
            Δ_R = (-Γ_R*D/(2*pi*D))*log.((w.-D .+im*eta)./(w .+D .+im*eta))
            #Λ = -im*D*Γ*hilbert(ρ)

        elseif spec_fun_type =="ellipse"
            ρ = (2/(π*D))*real(sqrt.(Complex.(1 .-(w/D).^2)))
            Δ_L = -im*D*Γ_L*hilbert(ρ)
            Δ_R = -im*D*Γ_R*hilbert(ρ)
        end

        f_L = 1 ./ (1 .+exp.((w .- μ_L)*β_L)); 
        f_R = 1 ./ (1 .+exp.((w .- μ_R)*β_R));

        ####Calculate determinant of M matrix (Ns x Ns)
        if Ns == 1
            Δ = Δ_L + Δ_R
            detM = w .- ϵi[1] .- Δ
            G = 1 ./detM
            A_den = (-1/π).*imag.(G)    
        elseif Ns == 2
            detM = (w .-ϵi[1] .-Δ_L).*(w.-ϵi[2].-Δ_R)
            detM = detM .- abs.(ti[1]^2) 
            G = ((w .-ϵi[2] .-Δ_R))./detM
            A_den = (-1/π).*imag.(G)    
        elseif Ns == 3
            detM = (w .-ϵi[1] .-Δ_L).*(w.-ϵi[3].-Δ_R).*(w.-ϵi[2])
            detM = detM - (abs.(ti[2])^2)*(w.-ϵi[1].-Δ_L)
            detM = detM - (abs.(ti[1])^2)*(w.-ϵi[3].-Δ_R)
            G =  (w .-ϵi[1] .-Δ_L).*(w.-ϵi[3].-Δ_R)./detM
            A_den = (-1/π)*imag.(G)
        end

        A = (D*Γ*ρ/π) ./(abs.(detM).^2)
        if length(ti)>0
            coupling_factor = prod(abs.(ti).^2)
            A = A.*coupling_factor
        end    
        f_L = 1 ./ (1 .+exp.((w .- μ_L)*β_L)); 
        f_R = 1 ./ (1 .+exp.((w .- μ_R)*β_R));

        prefactor = (4*π*D*Γ_L*D*Γ_R)/(2*π*D*Γ)
        # Compute the particle-current, energy-current and density:
        Jp = prefactor*sum(ρ.*A.*(f_L - f_R))*dw;
        Je = prefactor*sum(ρ.*w.*A.*(f_L - f_R))*dw;
        n = 0.5*sum(A_den.*(f_L + f_R))*dw; 
        return [Jp,Je,n] 
    end


