

   
    export Base_params
    export dependent_params
    export heaviside
    export Default_Fermion_P_initialisation
    export Default_Spin_Boson_P_initialisation
    export DP_initialisation
    export DP_initialisation_NEQ
    export Reinitialise_Hamiltonian
    export initialise_indices
    
    
    export thermofield_renormalisation
    export thermal_factor
    export spectral_function
    export mode_operators
    export approx_J
    
    
    export direct_mapping
    export reaction_mapping
    export orthopol_chain
    export orthopol_chain_NEQ
    export stieltjes_mod
    export removeZeroWeights
    
    
    export initial_correlation_matrix
    export initialise_bath
    export create_H_single
    export Full_hamiltonian_NEQ
    #export non_interacting_sys_H
    export system_bath_couplings
    export system_Hamiltonian
    export thermofield_chain_mapping_H_matrix


    ###These functions are for the non-equilibrium SIAM.
    export initialise_bath_NEQ_SIAM
    export NEQ_SIAM_DP_initialisation
    export NEQ_SIAM_hamiltonian

    ##Markovian closure functions
    export super_fermion_solver
    export super_fermion_solver_factorised_initial_state
    export Markovian_closure
    export combine_thermofield_with_markovian_closure
    export interleave_matrix
    #export leftvacuum

        



    mutable struct Base_params
        ###Bath params
        N_L ::Int64                      #Number of left bath sites
        N_R ::Int64                      #Number of right bath sites
        Ns  ::Int64                      #Number of system sites (not including ancilla)  
        N_chain ::Int64       
        Nc ::Int64                       #Number of sites for Markovian closure (if using)    
        β_L ::Float64                    #inverse temperature of right bath
        β_R ::Float64                    #inverse temperature of left bath
        β_cutoff ::Float64               ##cutoff above which the temperature is treated as zero (β=∞)
        μ_L ::Float64                   #chemical potential of left bath
        μ_R ::Float64                   #chemical potential of right bath
        Γ_L ::Float64                    #Left coupling strength
        Γ_R ::Float64                    #Right coupling strength
        SB_coupling_op ::String
        D ::Float64                      #Half the bandwidth 
        spec_fun_type ::String           #Spectral function choice
        "Choice: ellipse or box"
        disc_choice ::String             #discretization choice
        "Choice: reaction, direct or orthopol" 
        use_stieltjes_mod ::Tuple{Bool,Float64}
        compute_maps_bool :: Bool
        map_as_matrix_or_MPO ::String
        symmetry_subspace ::String       #Choice of whether to only consider modes in a given symmetry subspace of the Hilbert space
        method ::Int64
        bath_mode_type ::String          #Type of bath mode e.g. Fermions, Bosons, spins etc
        sys_mode_type ::String
        bosonic_cutoff ::Int64
        ohmic_cutoff ::String            #Whether to have a hard or soft cutoff
        method_type ::String             #Choices are "SFMC" or "TFCM", i.e. superfermion with markovian closure and thermofield chain mapping
        
                  
        ϵ ::Float64                      #Energy of system modes
        B ::Float64                      #Magnetic field for spinful electrons
        tc ::Float64                     #System hopping
        init_occ ::Float64               #Initial occupation of system modes
        ωq ::Float64                     #Z Magnetic field for spin
        S_ohmic ::Float64                #Ohmic_factor
        ωc :: Float64                    ##Freqency cutoff
        Δ ::Float64                      #X magnetic field for spin
        U ::Float64
        n ::Int64                        #Number of timesteps between each calculation of L,Λ
        δt ::Float64                     #timestep
        T ::Float64                      #Evolution time
        ordering_choice ::String         #Choose whether to interleave system and ancilla modes or separate them, 
        "Choice: interleaved or separated" 
        Kr_cutoff ::Float64              #Numerical cutoff used for Krylov enrichment
        k1 ::Int64                       #Number of Krylov states used
        τ_Krylov ::Float64
        tdvp_cutoff ::Float64            #Numerical cutoff for tdvp
        minbonddim ::Int64               #Minimum bond dimension for tdvp 
        maxbonddim ::Int64               #Maximum bond dimension for tdvpp
        T_enrich ::Float64               #Time in simulation where enrichment is used
        eta ::Float64                    #Numerical parameter used for LB calculation            
        using_system_ancilla ::Bool      #Boolean for whether to use ancillas or not for system modes
        Base_params() = new()
    end



    mutable struct dependent_params
        """
        Struct holding dependent parameters defined by Base_params.
        """

        N ::Int64                        #number of modes including ancillas
        ϵi ::Vector{Float64}             #self energies of system modes
        ti ::Vector{Float64}             #coupling of system modes
        Ui ::Vector{Float64}             #system interaction strength
        q ::UnitRange{Int64}             #system and ancilla sites
        qB_L ::UnitRange{Int64}  
        qB_R ::UnitRange{Int64}  
        qS ::StepRange{Int64, Int64}     #system sites
        qA ::StepRange{Int64, Int64}     #ancilla sites
        qtot ::UnitRange{Int64}  
        times ::Vector{Float64}          #Evolution time array
        bath_ann_op ::Any
        bath_cre_op ::Any
        sys_ann_op ::Any
        sys_cre_op ::Any
        s ::Any
        c ::Any
        cdag ::Any
        Id ::Any
        F ::Any
        Ci ::Matrix{ComplexF64}
        H_single ::Matrix{ComplexF64}
        H_MPO ::MPO
        MPO_terms ::Any
        ψ_init ::MPS

        T_unenriched ::Float64
        nframe_en ::Int64
        nframe_un ::Int64
        nframe ::Int64
        times1 ::Vector{Float64}

        Λ :: Any

        dependent_params() = new()
    end


   
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    heaviside(t) = 0.5 * (sign.(t) .+ 1)

    function Default_Fermion_P_initialisation() 
        """
        Some default parameters for a Fermionic NEQ setup.
        """

        P = Base_params()
        P.N_L = 30
        P.N_R = 30
        P.Ns = 2
        P.N_chain = 30
        P.β_L = 0.2
        P.β_R = 0.2
        P.β_cutoff = 100
        P.μ_L = 0.1
        P.μ_R = -0.1
        P.Γ_L = 0.01 
        P.Γ_R = 0.01
        P.D = 1
        P.spec_fun_type = "ellipse"
        P.disc_choice = "orthopol" 
        P.use_stieltjes_mod = (false,0)
        P.bath_mode_type = "Fermion"
        P.sys_mode_type = "Fermion"
        P.method = 0
        P.map_as_matrix_or_MPO = "matrix"
        P.ϵ = 0.2
        P.B = 0
        P.tc = 0.1
        P.U = 0
                   
        P.n = 5
        P.δt = 0.1
        P.T = 30
       
        P.symmetry_subspace = "Number conserving"
        P.ordering_choice = "interleaved"
        P.compute_maps_bool = false
        P.using_system_ancilla = true
        P.Kr_cutoff = 1e-12
        P.k1 = 3 
        P.τ_Krylov = 1 
        P.tdvp_cutoff =1e-12
        P.minbonddim = 10
        P.maxbonddim = 100
        P.T_enrich = 5
        P.eta = 0.005

        P.method_type = "TFCM"
        P.Nc = 6
        return P
    end

    function Default_Spin_Boson_P_initialisation() 
        """
        Some default parameters for a SB eq setup.
        """

        P = Base_params()
        P.N_L = 30
        P.N_R = 30
        P.Ns = 2
        P.N_chain = 30
        P.β_L = 0.2
        P.β_R = 0.2
        P.β_cutoff = 100
        P.μ_L = 0
        P.μ_R = 0
        P.Γ_L = 0.1 
        P.Γ_R = 0.1
        P.SB_coupling_op = "Sz"
        P.D = 1
        P.spec_fun_type = "Ohmic"
        P.S_ohmic = 1
        P.ohmic_cutoff = "exponential"
        P.disc_choice = "orthopol" 
        P.use_stieltjes_mod = (false,0)
        P.bath_mode_type = "Boson"
        P.sys_mode_type = "S=1/2"
        P.method = 0
        P.map_as_matrix_or_MPO = "Matrix"
        P.bosonic_cutoff = 5

        P.ωq = 0.5                     
        P.ωc = 1                    
        P.Δ = 1                      
        
        
        P.n = 5
        P.δt = 0.1
        P.T = 30

        P.symmetry_subspace = "Full"
        P.ordering_choice = "interleaved"
        P.compute_maps_bool = true
        P.using_system_ancilla = true
        P.Kr_cutoff = 1e-10
        P.k1 = 3 
        P.τ_Krylov = 1 
        P.tdvp_cutoff =1e-12
        P.minbonddim = 10
        P.maxbonddim = 100
        P.T_enrich = 5
        P.eta = 0.005
        P.method_type = "TFCM"
        return P
    end
    
    
        
        
    function DP_initialisation(P;kwargs...)
        """
        Initialises all dependent variables defined by P.
        """
        (;μ_L,μ_R,using_system_ancilla,sys_mode_type,bath_mode_type,method_type,compute_maps_bool,
        Ns) = P
        DP = dependent_params()
        
        if sys_mode_type == "Electron" && bath_mode_type != sys_mode_type
            println("Haven't implemented this combination of parameters")
        end
        if sys_mode_type == "Electron" && method_type == "SFMC"
            println("Haven't implemented spinful fermions for superfermion method")
        end

        (using_system_ancilla ==false) && (compute_maps_bool = false)


        initialise_indices(P,DP)

        DP.times = range(P.δt,stop=P.T,step =P.δt)           #Simulation time vector
        if P.bath_mode_type == "Boson"
            DP.bath_ann_op = ["A"]
            DP.bath_cre_op = ["Adag"]
        elseif P.bath_mode_type == "Fermion"
            DP.bath_ann_op = ["C"]
            DP.bath_cre_op = ["Cdag"]
        elseif P.bath_mode_type == "Electron"
            DP.bath_ann_op = ["Cdn","Cup"]
            DP.bath_cre_op = ["Cdagdn","Cdagup"]
        end
        if P.sys_mode_type == "Boson"
            DP.sys_ann_op = ["A"]
            DP.sys_cre_op = ["Adag"]
        elseif P.sys_mode_type == "Fermion"
            DP.sys_ann_op = ["C"]
            DP.sys_cre_op = ["Cdag"]
        elseif P.sys_mode_type == "S=1/2"
            DP.sys_ann_op = ["S+"]
            DP.sys_cre_op = ["S-"]
        elseif P.bath_mode_type == "Electron"
            DP.sys_ann_op = ["Cdn","Cup"]
            DP.sys_cre_op = ["Cdagdn","Cdagup"]
        end

        DP.s,DP.cdag,DP.c = mode_operators(DP,P;kwargs...)    # Array of site indices                      # identity matrix

        DP.ϵi = get(kwargs,:ϵi,P.ϵ*ones(P.Ns));                              #self energies of system modes
        DP.ti = get(kwargs,:ti,P.tc*ones(P.Ns-1));                           #coupling of system modes
        if P.sys_mode_type == "Electron"
            DP.Ui = P.U*ones(P.Ns);
        else 
            DP.Ui = P.U*ones(P.Ns-1)
        end                            #system interaction strengths
        
        DP.Id = Vector{ITensor}(undef,length(DP.s))                    #List of MPS identities
    
        # if P.sys_mode_type == "Electron"
        #     DP.F = [ops(DP.s,[("F$spin", n) for spin in  ["dn", "up"]]) for n in DP.qtot]
        # else
        #     DP.F = [ops(DP.s, [("F", n) for spin in 1:length(DP.c[1])]) for n in DP.qtot]
        # end
        DP.F = ops(DP.s, [("F", n) for n in DP.qtot])
        
        for i =1:length(DP.s)
            iv = DP.s[i]
            ID = ITensor(iv', dag(iv));
            for j in 1:ITensors.dim(iv)
                ID[iv' => j, iv => j] = 1.0
            end
            DP.Id[i] = ID
        end

        if P.method_type == "TFCM"
            """
            Note if separated ordering is used and you want to use correlation matrix map extraction method, initial_state_gates should be
            used along with the spin transform in the map calculation function. Alternatively, initial_state_gates_separated can be used
            with the fermionic particle hole transform, but I'm not sure why this works. 
            """

            DP.ψ_init = initialise_psi(P,DP;kwargs...)
            DP.Ci = initial_correlation_matrix(DP,P)
        else
            ##Currently have a patchy way of calculating the initial correlation matrix for SFMC
            DP.ψ_init,Ci_physical = initialise_psi(P,DP;kwargs...)
        end
        if P.method_type == "SFMC"
            @assert(P.sys_mode_type == "Fermion")
            @assert(P.bath_mode_type== "Fermion")
        
            H,Γd,Γe,Ci = combine_thermofield_with_markovian_closure(Ci_physical,P,DP)
        
            # Incoherent driving matrix:
            Γe = diagm(vec(Γe));
            Γd = diagm(vec(Γd));
            Ω  = 0.5*(Γd - Γe);
            DP.Λ  = 0.5*(Γd + Γe); 
            global const_offset = tr(H+im*DP.Λ)
            @show(const_offset)

            # Build the "super" (2N) x (2N) effective Hamiltonian matrix:   
            DP.H_single = [(H - im*Ω) (im*Γe); (im*Γd) (H + im*Ω)];
            DP.Ci = complex([Ci I-Ci;Ci I-Ci])
            
            N = expect(DP.ψ_init,"n")
            # @show(real(const_offset))
            #DP.H_single -= (const_offset/N)*I

            if P.ordering_choice == "interleaved"
                DP.H_single = interleave_matrix(DP.H_single)
                DP.Ci= interleave_matrix(DP.Ci)
            end

            DP.MPO_terms = build_from_matrix(DP.H_single,DP.bath_cre_op,DP.bath_ann_op)

            for i =1:Ns
                if sys_mode_type == "Fermion"
                    if i<Ns
                        DP.MPO_terms += DP.Ui[i],"n",DP.qS[i],"n",DP.qS[i+1]
                        DP.MPO_terms -= DP.Ui[i],"C",DP.qA[i],"Cdag",DP.qA[i],"C",DP.qA[i+1],"Cdag",DP.qA[i+1]
                    end
                elseif sys_mode_type == "Electron"
                    DP.MPO_terms += DP.Ui[i],"Nup",DP.qS[i],"Ndn",DP.qS[i+1]
                    DP.MPO_terms -= DP.Ui[i],"Cup",DP.qA[i],"Cdagup",DP.qA[i],"Cdn",DP.qA[i],"Cdagdn",DP.qA[i]
                end
            end

            #MPO_terms -= const_offset,"Id",1
        
        else
            DP.H_single = create_H_single(P,DP)
            HS_os,DP.H_single = system_Hamiltonian(DP.H_single,P,DP)
            couplings,DP.H_single =  system_bath_couplings(DP.H_single,P,DP)
            DP.MPO_terms = HS_os + couplings
            if P.N_L >0
                [DP.MPO_terms += build_from_matrix(DP.H_single[DP.qB_L,DP.qB_L],DP.bath_cre_op[i],DP.bath_ann_op[i]) for i in 1:length(DP.bath_ann_op)]
            end
        
            if P.N_R >0
                if P.using_system_ancilla
                    offset = 2*P.N_L+2*P.Ns
                else
                    offset = 2*P.N_L+P.Ns
                end
                [DP.MPO_terms += build_from_matrix(DP.H_single[DP.qB_R,DP.qB_R],DP.bath_cre_op[i],DP.bath_ann_op[i];offset= offset) for i in 1:length(DP.bath_ann_op)]
            end
        end
        DP.H_MPO = MPO(DP.MPO_terms,DP.s)     
        if P.method_type == "SFMC"  
            DP.H_MPO -= const_offset*MPO(DP.s,"Id")
        end
        
        DP.T_unenriched = round(P.T-P.T_enrich,digits=10)       # Time when the state is no longer enriched each step                                       
        DP.nframe_en = Int(P.T_enrich/(P.n*P.δt))               #Number of timesteps between each enrichment and calculation of map
        DP.nframe_un = Int(DP.T_unenriched/(P.n*P.δt))          #Number of timesteps between each calculation of map after enrichment
        DP.nframe = DP.nframe_en + DP.nframe_un                 #Number of maps calculated
        DP.times1 = range(P.δt,stop = P.T_enrich,step = P.δt)

        

        return DP
    end

    function DP_initialisation_NEQ(P;kwargs...)
        """
        Initialises all dependent variables defined by P.
        """
        (;μ_L,μ_R,using_system_ancilla,sys_mode_type,bath_mode_type,method_type,compute_maps_bool,
        Ns,N_L,N_R) = P
        DP = dependent_params()
        
        initialise_indices(P,DP)

        DP.times = range(P.δt,stop=P.T,step =P.δt)           #Simulation time vector
        if P.bath_mode_type == "Boson"
            DP.bath_ann_op = ["A"]
            DP.bath_cre_op = ["Adag"]
        elseif P.bath_mode_type == "Fermion"
            DP.bath_ann_op = ["C"]
            DP.bath_cre_op = ["Cdag"]
        elseif P.bath_mode_type == "Electron"
            DP.bath_ann_op = ["Cdn","Cup"]
            DP.bath_cre_op = ["Cdagdn","Cdagup"]
        end
        if P.sys_mode_type == "Boson"
            DP.sys_ann_op = ["A"]
            DP.sys_cre_op = ["Adag"]
        elseif P.sys_mode_type == "Fermion"
            DP.sys_ann_op = ["C"]
            DP.sys_cre_op = ["Cdag"]
        elseif P.sys_mode_type == "S=1/2"
            DP.sys_ann_op = ["S+"]
            DP.sys_cre_op = ["S-"]
        elseif P.bath_mode_type == "Electron"
            DP.sys_ann_op = ["Cdn","Cup"]
            DP.sys_cre_op = ["Cdagdn","Cdagup"]
        end

        DP.s,DP.cdag,DP.c = mode_operators(DP,P;kwargs...)    # Array of site indices                      # identity matrix

        DP.ϵi = get(kwargs,:ϵi,P.ϵ*ones(P.Ns));                              #self energies of system modes
        DP.ti = get(kwargs,:ti,P.tc*ones(P.Ns-1));                           #coupling of system modes
        if P.sys_mode_type == "Electron"
            DP.Ui = P.U*ones(P.Ns);
        else 
            DP.Ui = P.U*ones(P.Ns-1)
        end                            #system interaction strengths
        
        DP.Id = Vector{ITensor}(undef,length(DP.s))                    #List of MPS identities
        DP.F = ops(DP.s, [("F", n) for n in DP.qtot])
        
        for i =1:length(DP.s)
            iv = DP.s[i]
            ID = ITensor(iv', dag(iv));
            for j in 1:ITensors.dim(iv)
                ID[iv' => j, iv => j] = 1.0
            end
            DP.Id[i] = ID
        end
        
        DP.ψ_init = initialise_psi(P,DP;kwargs...)
        DP.Ci = initial_correlation_matrix(DP,P)

        DP.H_single = Full_hamiltonian_NEQ(P,DP)
        DP.MPO_terms = build_from_matrix(DP.H_single,"Cdag","C")

        DP.H_MPO = MPO(DP.MPO_terms,DP.s)     

        DP.T_unenriched = round(P.T-P.T_enrich,digits=10)       # Time when the state is no longer enriched each step                                       
        DP.nframe_en = Int(P.T_enrich/(P.n*P.δt))               #Number of timesteps between each enrichment and calculation of map
        DP.nframe_un = Int(DP.T_unenriched/(P.n*P.δt))          #Number of timesteps between each calculation of map after enrichment
        DP.nframe = DP.nframe_en + DP.nframe_un                 #Number of maps calculated
        DP.times1 = range(P.δt,stop = P.T_enrich,step = P.δt)

        

        return DP
    end

    function Reinitialise_Hamiltonian(P,DP)

        """
        This function reinitialises the hamiltonian in separated ordering
        """
        (;Ns) = P
        P.ordering_choice = "separated"
        DP.qS = DP.q[1:Ns]
        DP.qA = DP.q[Ns+1:2*Ns]
    
        DP.H_single = create_H_single(P,DP)
        HS_os,DP.H_single = system_Hamiltonian(DP.H_single,P,DP)
        couplings,DP.H_single =  system_bath_couplings(DP.H_single,P,DP)
        MPO_terms = HS_os + couplings
        if P.N_L >0
            [MPO_terms += build_from_matrix(DP.H_single[DP.qB_L,DP.qB_L],DP.bath_cre_op[i],DP.bath_ann_op[i]) for i in 1:length(DP.bath_ann_op)]
        end
    
        if P.N_R >0
            if P.using_system_ancilla
                offset = 2*P.N_L+2*P.Ns
            else
                offset = 2*P.N_L+P.Ns
            end
            [MPO_terms += build_from_matrix(DP.H_single[DP.qB_R,DP.qB_R],DP.bath_cre_op[i],DP.bath_ann_op[i];offset= offset) for i in 1:length(DP.bath_ann_op)]
        end
    
        DP.H_MPO = MPO(MPO_terms,DP.s)   
        return DP
    end  

    function initialise_indices(P,DP)

        (;N_L,N_R,N_chain,Nc,β_L,β_R,β_cutoff,bath_mode_type,
        μ_L,μ_R,using_system_ancilla,Ns,method_type) = P
        
        N_R >0 && @assert(N_chain <= N_R)
        N_L >0 && @assert(N_chain <= P.N_L)
        
        if bath_mode_type == "Boson"
            """
            For bosons, one thermofield bath is removed at low temperatures,
            so if the temperature is low enough we don't purify the bath.
            """

            @assert(μ_R<= 0)
            @assert(μ_R<= 0)
            DP.N = N_L + N_R     

            DP.qB_L = 1:(β_L < β_cutoff ? 2 * N_L : N_L)
            DP.N += β_L < β_cutoff ? N_L : 0
            DP.N += β_R < β_cutoff ? N_R : 0
        else
            DP.N = 2*N_L+2*N_R 
            DP.qB_L = DP.qB_L = 1:2*N_L 
        end
        DP.qB_L = N_L == 0 ? (0:0) : DP.qB_L
        
        if method_type == "SFMC"
            """
            Each bath splits into an empty and filled bath, which
            then each split into a real and a superfermion bath. I assume an interleaved 
            ordering.
            """
            DP.N = 4*(N_L+N_R) + 2 * Ns
            DP.N += N_L == 0 ? 0 : 4*Nc
            DP.N += N_R == 0 ? 0 : 4*Nc  
            DP.qB_L = N_L == 0 ? (0:0) : 1:4*(N_L+Nc)
            DP.qB_R = N_R == 0 ? (0:0) : DP.qB_L[end] + 2*Ns + 1:DP.N
            DP.q = DP.qB_L[end] + 1:(DP.qB_L[end] + 2*Ns)
            DP.qS = DP.q[1:2:(2*Ns-1)]
            DP.qA = DP.q[2:2:2*Ns]
        else
            DP.N += using_system_ancilla ? 2 * Ns : Ns
            DP.qB_R = N_R == 0 ? (0:0) : DP.qB_L[end] + (using_system_ancilla ? 2 * Ns : Ns) + 1:DP.N
            DP.q = DP.qB_L[end] + 1:(DP.qB_L[end] + (using_system_ancilla ? 2 * Ns : Ns))
            if using_system_ancilla
                if P.ordering_choice == "separated"
                    DP.qS = DP.q[1:P.Ns]
                    DP.qA = DP.q[P.Ns+1:2*P.Ns]
                elseif P.ordering_choice == "interleaved"   
                    DP.qS = DP.q[1:2:(2*P.Ns-1)]
                    DP.qA = DP.q[2:2:2*P.Ns]
                end
            else
                DP.qS = DP.q
                DP.qA = 0:0
            end
        end

        DP.qtot = 1:DP.N 
    end


    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    Spectral density and mappings
    """

    function thermal_factor(w,β,μ,mode_type)
        if mode_type == "Fermion" || mode_type == "Electron"
            f = 1 ./(exp.(β*(w .- μ)).+ 1)
        elseif mode_type == "Boson"
            f = 1 ./(exp.(β*(w .- μ)).-1)
        end
        return f
    end
        
    function thermofield_renormalisation(w,β,μ,thermo_chain_number,mode_type)
        fw = thermal_factor(w,β,μ,mode_type)
        
        if mode_type == "Fermion" || mode_type == "Electron"
            symmetry_fac = -1
        elseif mode_type == "Boson"
            symmetry_fac = 1
        end
        renorm = 1
        if thermo_chain_number == 0
            renorm = 1
        elseif thermo_chain_number == 1
            renorm = 1 .+(symmetry_fac*fw)
        elseif thermo_chain_number == 2 
            renorm = fw
        else 
            error("no chain chosen for thermofield renormalisation")
        end
        return renorm
    end

    function spectral_function(w,thermo_chain_number,side,P;kwargs...)
        """
        This function creates the spectral function for either the left or right bath 
        depending on "side". spec_therm denotes whether the bath is a filled chain, empty chain 
        or the full chain with no thermofield transform.
        """
        (;bath_mode_type,ωc,S_ohmic) = P

        ###Choosing which bath
        if side =="left"
            spec_fun_type,Γ,β,μ,D= P.spec_fun_type,P.Γ_L,P.β_L,P.μ_L,P.D
        elseif side == "right"
            spec_fun_type,Γ,β,μ,D = P.spec_fun_type,P.Γ_R,P.β_R,P.μ_R,P.D
        else 
            error("neither side chosen")
        end
        renorm = thermofield_renormalisation(w,β,μ,thermo_chain_number,bath_mode_type)
        
        ##Choosing which spectral function to use
        if spec_fun_type == "box"
            ρ = (1/(2*D))*(heaviside(w .+ D) .- heaviside(w .- D)).*renorm
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type =="ellipse"
            ρ = real((2/(π*D))*sqrt.(Complex.(1 .-(w/D).^2)).*renorm)
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type =="Lorentzian"
            λ = get(kwargs,:λ,0.2)
            α = 1/(2*λ*atan(D/λ))
            ρ = 1 .+(w./λ).^2
            ρ = (α./ρ).*renorm
            J =  (Γ*D/π)*ρ
        elseif spec_fun_type == "symmetric ohmic"
            ρ = (w.^2).*exp.(-abs.(w) ./ωc)
            factor = 2*2*ωc^3-ωc*(2*ωc^2+2*ωc*D+D^2)*exp(-D/ωc)
            ρ = ρ/factor
            ρ = ρ .*renorm
            J =  (Γ*D/π)*ρ
        elseif contains(spec_fun_type,"Ohmic")
            if P.ohmic_cutoff == "hard"
                support = heaviside(w) .-heaviside(w .-ωc)
            else
                support = exp.(-w/ωc)
            end
            ρ = (w.^(S_ohmic) .*support).*renorm
            #ρ = ρ.*(ωc^(1-S_ohmic))
            J =  Γ*(ωc^(1-S_ohmic))*ρ
            #J[1] = 0
            replace_nan(J)
        elseif spec_fun_type == "Drude Lorentz"
            ##From https://arxiv.org/pdf/2505.21017
            γ = 1
            J = 2*Γ*γ.*w./(γ^2 .+ w.^2)
        elseif spec_fun_type == "Ohmic Plenio definition"
            #Using the definition from https://arxiv.org/abs/1510.03100
            support = exp.(-w/ωc)
            J = Γ*(w.^(S_ohmic) .*support).*renorm
        elseif spec_fun_type == "GaAs quantum dot"
            ce = 0.1271
            ωe = 2.555
            ωh = 2.938
            ch = 0.0635
            J1 = ce*exp.(-(w.^2)/ωe^2)
            J2 = ch*exp.(-(w.^2)/ωh^2)
            J = (w.^3) .*(J1 .+ J2)
        elseif spec_fun_type == "GaAs quantum dot 2"
            ce = 0.1271
            ωe = 2.555
            ωh = 2.938
            ch = 0.0635
            J1 = ce*exp.(-(w.^2)/ωe^2)
            J2 = ch*exp.(-(w.^2)/ωh^2)
            J = (w.^3) .*((J1 .+ J2).^2)
        elseif spec_fun_type == "smoothed box"

            """
            Based on Influence functional paper by Abanin: An efficient method for quantum impurity problems out of equilibrium
            """

            ν =  100
            ωc = 1
            denominator = (1 .+exp.(ν*(w .-ωc))).*(1 .+exp.(-ν*(w .+ωc)))
            J = (Γ/(2*π)) ./denominator
            J = J.*renorm
        elseif spec_fun_type == "WSCP"
            #This spectral function is taken from https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.123.090402/thermalizedTEDOPA_Supplemental.pdf
            rescale = 100
            
            S_vec = [0.39,0.23,0.23]
            sigma_vec = [0.4,0.25,0.2]
            ω_vec = [26,51,85]./rescale ##cm-1
            Omega_vec = [181,221,240]./rescale ##cm=1
            g_vec = [0.0173,0.0246,0.0182]
            γ = 5/rescale ##cm-1
        
            ##Low freqency
            J = 0*w 
            for k=1:3
                exponent = -(log.(w./ω_vec[k])).^2
                exponent = exponent./(2*(sigma_vec[k]^2))
                Jk = w .* exp.(exponent)
                Jk *= S_vec[k]/(sigma_vec[k]*sqrt(2*π))
                J += Jk
            end
            for m=1:3
                numerator = 4*γ*Omega_vec[m]*g_vec[m]*(Omega_vec[m]^2+γ^2) .*w
                denominator = π*(γ^2 .+ (w .+ Omega_vec[m]).^2).*(γ^2 .+ (w .- Omega_vec[m]).^2)
                Jm = numerator ./ denominator
                J += Jm
            end
            J = J.*renorm
        else
            error("No spectral density chosen")
        end
        return J
    end

     
    function mode_operators(DP,P;kwargs...)
        (;Ns,N_L,N_R,sys_mode_type,bath_mode_type,bosonic_cutoff,symmetry_subspace,
        using_system_ancilla,method_type,Nc) = P
        (;q,qB_L,qB_R,bath_ann_op,bath_cre_op,sys_ann_op,sys_cre_op) = DP  

        ##Site dependent cutoff, used in https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.123.090402
        left_cutoff_dims = get(kwargs, :left_cutoff_dims,fill(bosonic_cutoff,length(qB_L)))
        right_cutoff_dims = get(kwargs, :right_cutoff_dims,fill(bosonic_cutoff,length(qB_R)))


        if using_system_ancilla == false && method_type == "TFCM"
            system_length = Ns
        else
            system_length = 2*Ns
        end
    
        left_bath_bool = N_L>0
        right_bath_bool = N_R>0
    
        if symmetry_subspace == "Number conserving"
            sys_modes = siteinds(sys_mode_type,system_length;conserve_nf=true)
        elseif symmetry_subspace == "Number and Sz conserving"
            sys_modes = siteinds(sys_mode_type,system_length;conserve_qns=true)
        else
            sys_modes = siteinds(sys_mode_type,system_length)
        end
        if method_type == "SFMC"
            @assert(using_system_ancilla==false)
        end
        
        sys_modes = [settags(mode,sys_mode_type*",System Site,n="*string(i+qB_L[end])) for (i,mode) in enumerate(sys_modes)]
        s = sys_modes
        if left_bath_bool
            if bath_mode_type == "Boson"
                NL_modes = siteinds(bath_mode_type,length(qB_L),dim = bosonic_cutoff)
            elseif bath_mode_type == "Fermion" || bath_mode_type == "Electron"
                if symmetry_subspace == "Number conserving"
                    NL_modes = siteinds(bath_mode_type,length(qB_L);conserve_nf=true)
                elseif symmetry_subspace == "Number and Sz conserving"
                    NL_modes = siteinds(bath_mode_type,length(qB_L);conserve_qns=true)
                else
                    NL_modes = siteinds(bath_mode_type,length(qB_L))
                end
            end
            NL_modes =[settags(mode,bath_mode_type*",Left bath site,n="*string(i)) for (i,mode) in enumerate(NL_modes)]
            s = append!(NL_modes,s)
        end
    
        if right_bath_bool 
            if bath_mode_type == "Boson"
                NR_modes = siteinds(bath_mode_type,length(qB_R),dim = bosonic_cutoff)
            elseif bath_mode_type == "Fermion" || bath_mode_type == "Electron"
                if symmetry_subspace == "Number conserving"
                    NR_modes = siteinds(bath_mode_type,length(qB_R);conserve_nf=true)
                elseif symmetry_subspace == "Number and Sz conserving"
                    NR_modes = siteinds(bath_mode_type,length(qB_R);conserve_qns=true)
                else
                    NR_modes = siteinds(bath_mode_type,length(qB_R))
                end
            end
            NR_modes = [settags(mode,bath_mode_type*",Right bath site,n="*string(i+q[end])) for (i,mode) in enumerate(NR_modes)]
            s = append!(s,NR_modes)
        end
        ann_ops = [ops(s, [(sys_ann_op[i], n) for i in 1:length(sys_ann_op)]) for n in q]
        cre_ops = [ops(s, [(sys_cre_op[i], n) for i in 1:length(sys_ann_op)]) for n in q]
        
        if left_bath_bool
            ann_NL_modes = [ops(s, [(bath_ann_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_L]
            cre_NL_modes = [ops(s, [(bath_cre_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_L]
            ann_ops = append!(ann_NL_modes,ann_ops)
            cre_ops = append!(cre_NL_modes,cre_ops)
        end 
    
        if right_bath_bool
            ann_NR_modes = [ops(s, [(bath_ann_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_R]
            cre_NR_modes = [ops(s, [(bath_cre_op[i], n) for i in 1:length(bath_ann_op)]) for n in qB_R]
            ann_ops = append!(ann_ops,ann_NR_modes)
            cre_ops = append!(cre_ops,cre_NR_modes)
        end
    
        return s,cre_ops,ann_ops
    end

    function approx_J(J,ϵ_b,V_k,Γ)
        """
        Plots the true spectral function
        with the discretized version for comparison.
        """
        N = length(ϵ_b)
        η = 1/N
        samp = 10000; # Frequency sampling.
        w = range(-1.5,1.5,samp); # Frequency axis for Landauer calculations (slightly larger than the band).
        J_approx = zeros(samp)
        for i=1:N
            denom = (w .-ϵ_b[i]).^2 .+(η)^2
            L = η./(π*denom)
            delta = (V_k[i]^2)*L
        # display(plot(w,delta)) 
            J_approx += delta
        end
        J_exact = J(w,Γ)
        plot(w,J_exact)
        display(plot!(w,J_approx))
    end


    function direct_mapping(side,DP,P)
        """
        Implements direct discretisation, using simple trapezium integration.
        """
        (;D) = P
        (;ϵi) = DP
        if side=="left"
            Nb = P.N_L
            ϵ_ref = ϵi[1]
        elseif side=="right"
            Nb,Ns = P.N_R,P.Ns
            ϵ_ref = ϵi[Ns]
        else
            error("No side chosen.")
        end

        samp = 1000 # Number of points in mesh.
        y = LinRange(-D,D,Nb+1)
        tsq =  Vector{Float64}(undef,Nb)
        en =  Vector{Float64}(undef,Nb)
        for i =1:Nb
            x = LinRange(y[i],y[i+1],samp)
            Jx = spectral_function(x,thermo_chain_number,side,P)
            
            tsq[i] = trapz(x,Jx); 
            en[i] = (1/tsq[i])*trapz(x,x.*Jx);
            
        end
        ind = sortperm(abs.(en.-ϵ_ref)) 
        tsq, en = tsq[ind], en[ind];  
        tk = sqrt.(tsq)
        return [tk,en] 
    end


    function reaction_mapping(side,P)
        """
        Implements the reaction coordinate transformation.
        """

        if side=="left"
            Nb,method = P.N_L,P.method
        elseif side=="right"
            Nb,method = P.N_R,P.method
        else
            error("No side chosen.")
        end
        #Define fixed  mesh over [-2,2] to capture spectral function and
        # its hilbert transform correctly within [-1,1].
        samp = 100000 # Number of points in mesh.
        x = LinRange(-2,2,samp);

        Jx = spectral_function(x,thermo_chain_number,side,P)# Evaluate symbolic input function over the grid.

        Vsq =  Vector{Float64}(undef,Nb)
        en =  Vector{Float64}(undef,Nb)
        # Loop over the omega intervals and perform integrations:
        Jcur = Jx; # Current bath spectral function.
        for s=1:Nb
        # Simple trapezoid integration for hopping squared and on-site energy:
        Vsq[s] = trapz(x,Jcur); 
        en[s] = (1/Vsq[s])*trapz(x,x.*Jcur);

        Jprev = Jcur;
        JH = imag(hilbert(Jprev)); # Hilbert transform.
        Jcur = (Vsq[s]/(pi^2))*Jprev./(JH.^2+Jprev.^2);
        end

        Vk = sqrt.(Vsq)
        if method != 0
            #Now we form rotate the bath modes such that we have a star geometry. This is done
            #by diagonalising the bath modes.

            A,U = zeros(Nb+1,Nb+1),zeros(Nb+1,Nb+1)
            A[1,1],U[1,1] = 1,1

            for i =2:Nb+1
                A[i,i] = en[i-1]
                A[i-1,i] = Vk[i-1]
                A[i,i-1] = conj(Vk[i-1])
            end
            A_sub = A[2:Nb+1,2:Nb+1]
            U_sub = eigen(A_sub).vectors
            U[2:Nb+1,2:Nb+1] = U_sub
            A_star = U'*A*U

            ###Pretty sure its this way and not [1,2:Nb+1]
            Vk = A_star[2:Nb+1,1]
            en = diag(A_star)[2:Nb+1]
        end
        return [Vk, en]
    end


    function orthopol_chain(thermo_chain_number,side,P)
        """
        Implements chain mapping using orthogonal polynomials.
        thermo_chain_number:
        - 0 full spectral density, no thermofield
        - 1 occupation_number*spectral density (filled for fermions, empty for bosons)
        -2 empty chain for both
        """
        (;D,bath_mode_type,N_L,N_R,use_stieltjes_mod,N_chain,β_cutoff) = P
        
        if side=="left"
            Nb = N_L
            β = P.β_L
        elseif side=="right"
            Nb = N_R
            β = P.β_R
        else
            error("No side chosen.")
        end
        couplings,energies = complex(zeros(Nb)),complex(zeros(Nb))
        if thermo_chain_number == 2 && β >β_cutoff && bath_mode_type == "Boson"
            return couplings,energies
        else 
            w(t) = spectral_function(t, thermo_chain_number, side, P)
            if bath_mode_type == "Fermion" || bath_mode_type =="Electron"
                supp = (-2D,2D)
            elseif bath_mode_type == "Boson"
                supp = (0,D)
            end

            if use_stieltjes_mod[1]
                cutoff = use_stieltjes_mod[2]
                nodes = LinRange(supp[1],supp[2],10000)
                weights = w(nodes)
                α_coeffs,β_coeffs = stieltjes_mod(N_chain,nodes,weights,cutoff;removezeroweights=true)
            else
                #degree = Nb-1
                my_meas = Measure("my_meas", w, supp, false, Dict())
                my_op = OrthoPoly("my_op", N_chain-1, my_meas; Nquad=100000);
                α_coeffs,β_coeffs = coeffs(my_op)[:,1],coeffs(my_op)[:,2]
            end
            if length(α_coeffs)<Nb
                energies[1:N_chain] = α_coeffs
                couplings[1:N_chain] = sqrt.(β_coeffs)
                energies[N_chain+1:end] .= α_coeffs[N_chain]
                couplings[N_chain+1:end] .= sqrt(β_coeffs[N_chain])
            else
                energies = α_coeffs
                couplings = sqrt.(β_coeffs)
            end

            return couplings,energies
        end

    end


    function orthopol_chain_NEQ(thermo_chain_number,side,P)
        """
        Implements chain mapping using orthogonal polynomials.
        thermo_chain_number:
        - 0 full spectral density, no thermofield
        - 1 occupation_number*spectral density (filled for fermions, empty for bosons)
        -2 empty chain for both
        """
        (;D,bath_mode_type,N_L,N_R,use_stieltjes_mod,N_chain,β_cutoff) = P
        
        if side=="left"
            β = P.β_L
        elseif side=="right"
            β = P.β_R
        else
            error("No side chosen.")
        end

        ##Treating N_R as N_L+N_R
        Nb = Int(N_R/2)

        couplings,energies = complex(zeros(Nb)),complex(zeros(Nb))
        if thermo_chain_number == 2 && β >β_cutoff && bath_mode_type == "Boson"
            return couplings,energies
        else 
            w(t) = spectral_function(t, thermo_chain_number, side, P)
            if bath_mode_type == "Fermion" || bath_mode_type =="Electron"
                supp = (-2D,2D)
            elseif bath_mode_type == "Boson"
                supp = (0,D)
            end

            if use_stieltjes_mod[1]
                cutoff = use_stieltjes_mod[2]
                nodes = LinRange(supp[1],supp[2],10000)
                weights = w(nodes)
                α_coeffs,β_coeffs = stieltjes_mod(N_chain,nodes,weights,cutoff;removezeroweights=true)
            else
                #degree = Nb-1
                my_meas = Measure("my_meas", w, supp, false, Dict())
                my_op = OrthoPoly("my_op", N_chain-1, my_meas; Nquad=100000);
                α_coeffs,β_coeffs = coeffs(my_op)[:,1],coeffs(my_op)[:,2]
            end
            if length(α_coeffs)<Nb
                energies[1:N_chain] = α_coeffs
                couplings[1:N_chain] = sqrt.(β_coeffs)
                energies[N_chain+1:end] .= α_coeffs[N_chain]
                couplings[N_chain+1:end] .= sqrt(β_coeffs[N_chain])
            else
                energies = α_coeffs
                couplings = sqrt.(β_coeffs)
            end

            return couplings,energies
        end

    end

    function stieltjes_mod(N::Int,nodes_::AbstractVector{<:Real},weights_::AbstractVector{<:Real},cutoff::Float64;removezeroweights::Bool=true)
        tiny = 10*floatmin()
        huge = 0.1*floatmax()
        α, β = zeros(Float64,N), zeros(Float64,N)
        nodes, weights = removezeroweights ? removeZeroWeights(nodes_,weights_) : (nodes_, weights_)
        Ncap = length(nodes)
        @assert N > 0 && N <= Ncap "N is out of range."
        s0::Float64 = sum(weights)
        @inbounds α[1] = dot(nodes,weights)/s0
        @inbounds β[1] = s0
        N == 1 && return α, β
        p0, p1, p2 = zeros(Float64,Ncap), zeros(Float64,Ncap), ones(Float64,Ncap)
    
        for k=1:N-1
            prev_vals = [α[k],β[k]]
            p_order_of_magnitude = floor(log10(mean(abs.(p2))))
            
            p0 = p1/(10^(p_order_of_magnitude))
            p1 = p2/(10^(p_order_of_magnitude))
            @inbounds p2 = (nodes .- α[k]) .* p1 - β[k]*p0
    
            s1 = dot(weights, p2.^2)
            s2 = dot(nodes, weights.*p2.^2)
    
            abs(s1) < tiny && throw(DomainError(tiny, "Underflow in stieltjes() for k=$k; try using `removeZeroWeights`"))
            !(maximum(abs.(p2))<=huge && abs(s2)<=huge) && throw(DomainError(huge, "Overflow in stieltjes for k=$k"))
            
            if abs(sum(prev_vals .-[s2/s1,s1/s0])/sum(prev_vals))>cutoff
                s0 = s0/(10^(2*p_order_of_magnitude))
                @inbounds α[k+1] = s2/s1
                @inbounds β[k+1] = s1/s0
                s0 = s1
            else
                println("α and β are stationary up to cutoff at mode,"*string(k)*", remaining modes are treated as identical.")
                α[k+1:end] .= α[k]
                β[k+1:end] .= β[k]
                break
            end
        end
        α, β
    end

    function removeZeroWeights(n::AbstractVector{<:Real},w::AbstractVector{<:Real})
        nw = [n w]
        inds = findall(w->abs(w)>=eps(), nw[:,2])
        # nw = sortrows(nw, by=x->x[1])
        nw = nw[inds,:]
        nw = sortslices(nw, dims=1, by=x->x[1])
        nw[:,1], nw[:,2]
    end


    #------------------------------------------------------------------------------------------------------------------------------------
    """
    Creation of Hamiltonian and initial correlation matrix.
    """

    function initial_correlation_matrix(DP,P)
        """
        Initial correlation matrix for maximally entangled system-ancilla pairs.
        """
        (;bath_mode_type,sys_mode_type,N_L,N_R,method_type) = P
        (;q,qB_L,qB_R,qtot,bath_cre_op,bath_ann_op,
        sys_ann_op,sys_cre_op,ψ_init,cdag,c,s,N) = DP
        
        C = complex(zeros(length(bath_ann_op)*N,length(bath_ann_op)*N))
        if P.sys_mode_type != "S=1/2"
            if method_type == "TFCM"
                if bath_mode_type == sys_mode_type
                    for (i, cre) in enumerate(bath_cre_op), (j, ann) in enumerate(bath_ann_op)
                        C[qtot .+ (i-1)*N, qtot .+ (j-1)*N] = transpose(correlation_matrix(ψ_init, cre, ann))
                    end
                else        
                    println("Correlation matrix initialisation for the system having a different mode type
                    to the bath not implemented rigorously")
                    if N_L >0
                        C[qB_L,qB_L] = transpose(correlation_matrix(ψ_init,bath_cre_op[1],bath_ann_op[1],sites=qB_L))
                    end
                
                    C[q,q] = transpose(correlation_matrix(ψ_init,sys_cre_op[1],sys_ann_op[1],sites=q))
                    if N_R >0
                        C[qB_R,qB_R] = transpose(correlation_matrix(ψ_init,bath_cre_op[1],bath_ann_op[1],sites=qB_R))
                    end
                end    
            else
                left_vac = leftvacuum(s)
                for i =1:length(s)
                    for j =1:length(s)
                        os = OpSum()
                        os += 1,"Cdag",i,"C",j
                        Op = MPO(os,s) 
                        C[i,j] = inner(left_vac',Op,ψ_init)
                    end
                end
            end
        else
            println("Initial correlation matrix not calculated for SB model")
        end
        return C
        
    end
        

    
    function initialise_bath(side,P,DP)
        (;disc_choice) = P
        

        """
        This block creates the terms that go into 
        building the bath hamiltonian.
        """
        (;method) = P
        if method == 0
            ch_l = [1,2]
        else
            ch_l = [0,0]
        end
        if disc_choice == "direct"
            Vk_emp, ϵb_emp = direct_mapping(side,DP,P)
            Vk_fill, ϵb_fill = direct_mapping(side,DP,P)
        elseif disc_choice == "reaction"
            Vk_emp, ϵb_emp = reaction_mapping(side,P)
            Vk_fill, ϵb_fill = reaction_mapping(side,P)
        elseif disc_choice == "orthopol"
            Vk_emp, ϵb_emp = orthopol_chain(ch_l[1],side,P)
            Vk_fill, ϵb_fill = orthopol_chain(ch_l[2],side,P)
        else
            error("no discretization chosen")
        end
        
        if side =="left"
            if method != 3
                """
                The reason for this condition is that for the tridiagonal case, the bath modes don't need to be put in 
                the right order for the band diagonalisation of fill_mat and emp_mat. The order is handled after
                this is done using the reflection_diag function. These arrays don't affect the initialisation of the state
                for the thermofield case so they don't need to be ordered correctly yet.
                """
                Vk_emp = reverse(Vk_emp)
                Vk_fill = reverse(Vk_fill)
                ϵb_emp = reverse(ϵb_emp)
                ϵb_fill = reverse(ϵb_fill)
            end
        end
        return Vk_emp,ϵb_emp,Vk_fill,ϵb_fill
    end

    function create_H_single(P,DP;kwargs...)
        (;N_L,N_R,β_L,β_R,β_cutoff) = P
        (;N) = DP
    
        """
        N_physical refers to the number of modes before the double to the superfermion description. 
        It also doesn't include the markovian closure modes as these are dealt with separately.
        """
        N_physical = get(kwargs,:N_physical,N)

    
        left_bath_bool = N_L>0
        right_bath_bool = N_R>0
       
    
        H_single = complex(zeros(N_physical,N_physical))
        if left_bath_bool
            if β_L >β_cutoff && P.bath_mode_type == "Boson"
                H_single = zero_temperature_thermofield_chain_mapping(H_single,"left",P,DP)
            else
                H_single = thermofield_chain_mapping_H_matrix(H_single,"left",P,DP)
            end
        end
        if right_bath_bool
            if β_R >β_cutoff && P.bath_mode_type == "Boson"
                H_single = zero_temperature_thermofield_chain_mapping(H_single,"right",P,DP)
            else
                H_single = thermofield_chain_mapping_H_matrix(H_single,"right",P,DP)
            end
        end
        return H_single
    end

    function Full_hamiltonian_NEQ(P,DP)
        (;N_R,Ns,using_system_ancilla,ordering_choice) = P
        (;N,qS,ϵi,ti) = DP

        sys_modes = qS

        if using_system_ancilla
            if ordering_choice == "separated"
                ind_difference = P.Ns+1
            else
                ind_difference = 2
            end
            right_bath_mode_filled_inds = 2*Ns+1:4:N 
        else
            right_bath_mode_filled_inds = Ns+1:4:N
            ind_difference = 1
        end

        H_single = complex(zeros(N,N))
        sym_factor = 1
        sys_modes = qS


        ##System_hamiltonian
        for i =1:length(sys_modes)
            H_single[sys_modes[i],sys_modes[i]] = ϵi[i]
            if i<Ns
                H_single[sys_modes[i],sys_modes[i+1]] = ti[i]
                H_single[sys_modes[i+1],sys_modes[i]] = conj(ti[i])
            end
        end


        Vk_emp_L, ϵb_emp_L = orthopol_chain_NEQ(1,"left",P)
        Vk_fill_L, ϵb_fill_L = orthopol_chain_NEQ(2,"left",P)

        Vk_emp_R, ϵb_emp_R = orthopol_chain_NEQ(1,"right",P)
        Vk_fill_R, ϵb_fill_R = orthopol_chain_NEQ(2,"right",P)

        b = 0
        for j in right_bath_mode_filled_inds
            b += 1   
            H_single[j,j] = sym_factor*ϵb_fill_R[b]
            H_single[j+1,j+1] = ϵb_emp_R[b]
            H_single[j+2,j+2] = sym_factor*ϵb_fill_L[b]
            H_single[j+3,j+3] = ϵb_emp_L[b]
            if j>right_bath_mode_filled_inds[1]
                H_single[j-4,j] = sym_factor*conj(Vk_fill_R[b])
                H_single[j,j-4] = sym_factor*Vk_fill_R[b]
                H_single[j-3,j+1] =  conj(Vk_emp_R[b])
                H_single[j+1,j-3] =  Vk_emp_R[b]

                H_single[j-2,j+2] = sym_factor*conj(Vk_fill_L[b])
                H_single[j+2,j-2] = sym_factor*Vk_fill_L[b]
                H_single[j-1,j+3] =  conj(Vk_emp_L[b])
                H_single[j+3,j-1] =  Vk_emp_L[b]
            end
        end


        """
        System bath coupling
        """

        @show(ind_difference)
        @show(ind_difference+1)
        @show(ind_difference+3)


        """Thermofield right empty bath to system"""
        H_single[sys_modes[end]+ind_difference+1,sys_modes[end]] = Vk_emp_R[1]
        H_single[sys_modes[end],sys_modes[end]+ind_difference+1] = conj(Vk_emp_R[1])

        """Thermofield right full bath to system"""
        H_single[sys_modes[end]+ind_difference,sys_modes[end]] =  sym_factor*Vk_fill_R[1]
        H_single[sys_modes[end],sys_modes[end]+ind_difference] =  sym_factor*conj(Vk_fill_R[1])

        """Thermofield left empty bath to system"""
        H_single[sys_modes[1]+(P.Ns-1)+ind_difference+3,sys_modes[1]] = Vk_emp_L[1]
        H_single[sys_modes[1],sys_modes[1]+(P.Ns-1)+ind_difference+3] = conj(Vk_emp_L[1])

        """Thermofield left full bath to system"""
        H_single[sys_modes[1]+(P.Ns-1)+ind_difference+2,sys_modes[1]] =  sym_factor*Vk_fill_L[1]
        H_single[sys_modes[1],sys_modes[1]+(P.Ns-1)+ind_difference+2] =  sym_factor*conj(Vk_fill_L[1])

        return H_single
    end

    function zero_temperature_thermofield_chain_mapping(H_single,side,P,DP)
        (;N,qB_L,qB_R) = DP

        if side == "left" 
            couplings, energies = orthopol_chain(0,"left",P)
            b = 0
            for j in qB_L
                b += 1
                H_single[j,j] = energies[b]
                if j<qB_L[end]
                    H_single[j,j+1] = couplings[b]
                    H_single[j+1,j] = conj(couplings[b])
                end
            end
        end
        if side == "right"
            couplings, energies = orthopol_chain(0,"right",P)
            b = 0
            for j in qB_R
                b += 1
                H_single[j,j] = energies[b]
                if j<N
                    H_single[j,j+1] = couplings[b+1]
                    H_single[j+1,j] = conj(couplings[b+1])
                end
            end
        end
        return H_single
    end

    function thermofield_chain_mapping_H_matrix(H_single,side,P,DP)
        (;N_L,N_R,Ns,bath_mode_type,using_system_ancilla) = P
     
        N_physical = size(H_single)[1]
    
        if bath_mode_type == "Fermion" || bath_mode_type == "Electron"
            sym_factor = 1
        elseif bath_mode_type == "Boson"
            sym_factor = -1
        end
        if side == "left" 
            Vk_emp_L,ϵb_emp_L,Vk_fill_L,ϵb_fill_L = initialise_bath("left",P,DP)
            
            left_bath_mode_inds = 1:2:2*(N_L)
            b = 0
            for j in left_bath_mode_inds
                b += 1
                H_single[j,j] = sym_factor*ϵb_fill_L[b]
                H_single[j+1,j+1] =  ϵb_emp_L[b]
                if j<(2*N_L-1)
                    H_single[j,j+2] = sym_factor*Vk_fill_L[b]
                    H_single[j+2,j] = sym_factor*conj(Vk_fill_L[b])
                    H_single[j+1,j+3] =  Vk_emp_L[b]
                    H_single[j+3,j+1] =  conj(Vk_emp_L[b])
                end
            end
        end
        if side == "right"
            Vk_emp_R,ϵb_emp_R,Vk_fill_R,ϵb_fill_R = initialise_bath("right",P,DP)
            if using_system_ancilla
                right_bath_mode_inds = 2*(N_L+Ns)+1:2:N_physical
            else
                right_bath_mode_inds = 2*N_L+Ns+1:2:N_physical
            end
            b = 0
            for j in right_bath_mode_inds
                b += 1   
                H_single[j,j] = sym_factor*ϵb_fill_R[b]
                H_single[j+1,j+1] = ϵb_emp_R[b]
                if j>right_bath_mode_inds[1]
                    H_single[j-2,j] = sym_factor*conj(Vk_fill_R[b])
                    H_single[j,j-2] = sym_factor*Vk_fill_R[b]
                    H_single[j-1,j+1] =  conj(Vk_emp_R[b])
                    H_single[j+1,j-1] =  Vk_emp_R[b]
                end
            end
        end
        return H_single
    end

    function system_bath_couplings(H_single,P,DP)
        (;N_L,N_R,sys_mode_type,bath_mode_type,method_type,using_system_ancilla,Ns,ordering_choice,
        β_L,β_R,β_cutoff) = P
        (;qB_L,qB_R,qS,bath_cre_op,bath_ann_op,sys_ann_op,
        sys_cre_op) = DP

        """
        os isn't used for the superfermion setup. The OpSum is created
        through build_from_matrix in HamiltonianBuilding.jl.
        """
        
        # Determine symmetry factor
        sym_factor = bath_mode_type == "Boson" ? -1 : 1

        left_bath_bool = N_L>0
        right_bath_bool = N_R>0
        
        os = OpSum()
        coupling_ann_op = sys_mode_type == "S=1/2" ? [P.SB_coupling_op] : sys_ann_op
        coupling_cre_op = sys_mode_type == "S=1/2" ? [P.SB_coupling_op] : sys_cre_op

        if P.spec_fun_type == "WSCP"
            coupling_ann_op = ["Proj0"]
            coupling_cre_op = ["Proj0"]
        end

        sys_modes = qS
        if method_type == "SFMC"
            @assert(using_system_ancilla == false)
            sys_modes = 2*N_L+1:2*N_L+Ns
        end

        if left_bath_bool 
            Vk_emp_L,ϵb_emp_L,Vk_fill_L,ϵb_fill_L = initialise_bath("left",P,DP)
            if β_L >β_cutoff && bath_mode_type == "Boson"
                """Thermofield left empty bath to system"""
                H_single[N_L,sys_modes[1]] =  Vk_emp_L[end]
                H_single[sys_modes[1],N_L] =  conj(Vk_emp_L[end])
            else   
                """Thermofield left empty bath to system"""
                H_single[2*N_L,sys_modes[1]] =  Vk_emp_L[end]
                H_single[sys_modes[1],2*N_L] =  conj(Vk_emp_L[end])

                """Thermofield left full (only for fermions, both are empty for bosons) bath to system"""
                H_single[2*N_L-1,sys_modes[1]] =  sym_factor*Vk_fill_L[end]
                H_single[sys_modes[1],2*N_L-1] =  sym_factor*conj(Vk_fill_L[end])
            end

            ##sum takes into account if the site is spinful or not.
            if method_type != "SFMC"
                for i in 1:length(bath_cre_op)
                    #Empty bath
                    os += Vk_emp_L[end],bath_cre_op[i],qB_L[end],coupling_ann_op[i],sys_modes[1]
                    os += conj(Vk_emp_L[end]),coupling_cre_op[i],sys_modes[1],bath_ann_op[i],qB_L[end]
                    if !(β_L >β_cutoff && bath_mode_type == "Boson")
                        #Filled bath
                        os += sym_factor*Vk_fill_L[end],bath_cre_op[i],qB_L[end]-1,coupling_ann_op[i],sys_modes[1]
                        os += sym_factor*conj(Vk_fill_L[end]),coupling_cre_op[i],sys_modes[1],bath_ann_op[i],qB_L[end]-1
                    end
                end
            end
        end

        if right_bath_bool 

            Vk_emp_R,ϵb_emp_R,Vk_fill_R,ϵb_fill_R = initialise_bath("right",P,DP)
            if method_type == "TFCM" && using_system_ancilla
                if ordering_choice == "separated"
                    ind_difference = P.Ns+1
                else
                    ind_difference = 2
                end
            else
                ind_difference = 1
            end
            if β_R >β_cutoff && bath_mode_type == "Boson"
                """Thermofield right empty bath to system"""
                H_single[sys_modes[end]+ind_difference,sys_modes[end]] = Vk_emp_R[1]
                H_single[sys_modes[end],sys_modes[end]+ind_difference] = conj(Vk_emp_R[1])
            else
                """Thermofield right empty bath to system"""
                H_single[sys_modes[end]+ind_difference+1,sys_modes[end]] = Vk_emp_R[1]
                H_single[sys_modes[end],sys_modes[end]+ind_difference+1] = conj(Vk_emp_R[1])

                """Thermofield right full (only for fermions, both are empty for bosons) bath to system"""
                H_single[sys_modes[end]+ind_difference,sys_modes[end]] =  sym_factor*Vk_fill_R[1]
                H_single[sys_modes[end],sys_modes[end]+ind_difference] =  sym_factor*conj(Vk_fill_R[1])
            end

            if method_type != "SFMC"
                for i in 1:length(bath_cre_op)
                    if !(β_R >β_cutoff && bath_mode_type == "Boson")
                        #Empty bath
                        os += Vk_emp_R[1],bath_cre_op[i],qB_R[2],coupling_ann_op[i],sys_modes[end] 
                        os += conj(Vk_emp_R[1]),coupling_cre_op[i],sys_modes[end],bath_ann_op[i],qB_R[2] 
                        #Filled bath
                        os += sym_factor*Vk_fill_R[1],bath_cre_op[i],qB_R[1],coupling_ann_op[i],sys_modes[end]
                        os += sym_factor*conj(Vk_fill_R[1]),coupling_cre_op[i],sys_modes[end],bath_ann_op[i],qB_R[1]
                    else
                        #Empty bath
                        os += Vk_emp_R[1],bath_cre_op[i],qB_R[1],coupling_ann_op[i],sys_modes[end] 
                        os += conj(Vk_emp_R[1]),coupling_cre_op[i],sys_modes[end],bath_ann_op[i],qB_R[1] 
                    end
                end
            end
        end
        return os,H_single
    end
        

    function system_Hamiltonian(H_single,P,DP)
        (;ωq,Δ,B,sys_mode_type,Ns,N_L,method_type,using_system_ancilla) = P
        (;ϵi,ti,Ui,qS,s) = DP     
        os = OpSum()
        
        """
        os isn't used for the superfermion setup. The OpSum is created
        through build_from_matrix in HamiltonianBuilding.jl.
        """
    
    
        sys_modes = qS
        if method_type == "SFMC"
            @assert(using_system_ancilla == false)
            sys_modes = 2*N_L+1:2*N_L+Ns
            
        end
    
        for i = 1:Ns
            if sys_mode_type == "Fermion"
                os += ϵi[i],"n",sys_modes[i]
                H_single[sys_modes[i],sys_modes[i]] = ϵi[i]
                if i<Ns
                    os += ti[i],"Cdag",sys_modes[i],"C",sys_modes[i+1]
                    os += conj(ti[i]),"Cdag",sys_modes[i+1],"C",sys_modes[i]
                    H_single[sys_modes[i],sys_modes[i+1]] = ti[i]
                    H_single[sys_modes[i+1],sys_modes[i]] = conj(ti[i])
                    os += Ui[i],"n",sys_modes[i],"n",sys_modes[i+1]
                end
            elseif sys_mode_type == "S=1/2"
                os += ωq,"Sz",sys_modes[i]
                os += Δ,"Sx",sys_modes[i]
                if i<Ns
                    os += ti[i],"Sz",sys_modes[i],"Sz",sys_modes[i+1]
                    os += conj(ti[i]),"Sz",sys_modes[i+1],"Sz",sys_modes[i]
                end
            elseif sys_mode_type == "Electron"
                os += ϵi[i],"Ntot",sys_modes[i]
                H_single[sys_modes[i],sys_modes[i]] = ϵi[i]
                os += Ui[i],"Nup",sys_modes[i],"Ndn",sys_modes[i]
                os += B,"Ndn",sys_modes[i]
                os += -B,"Nup",sys_modes[i]
                if i<Ns
                    for spin in ["up", "dn"]
                        os += ti[i], "Cdag$spin", sys_modes[i], "C$spin", sys_modes[i+1]
                        os += conj(ti[i]), "Cdag$spin", sys_modes[i+1], "C$spin", sys_modes[i]
                    end
                    H_single[sys_modes[i],sys_modes[i+1]] = ti[i]
                    H_single[sys_modes[i+1],sys_modes[i]] = conj(ti[i])
                end
            end
        end
        return os,H_single
    end

    function initialise_bath_NEQ_SIAM(side,P,DP)
        (;disc_choice) = P
        

        """
        This block creates the terms that go into 
        building the bath hamiltonian.
        """
        (;method) = P
        if method == 0
            ch_l = [1,2]
        else
            ch_l = [0,0]
        end
        if disc_choice == "direct"
            Vk_emp, ϵb_emp = direct_mapping(side,DP,P)
            Vk_fill, ϵb_fill = direct_mapping(side,DP,P)
        elseif disc_choice == "reaction"
            Vk_emp, ϵb_emp = reaction_mapping(side,P)
            Vk_fill, ϵb_fill = reaction_mapping(side,P)
        elseif disc_choice == "orthopol"
            Vk_emp, ϵb_emp = orthopol_chain(ch_l[1],side,P)
            Vk_fill, ϵb_fill = orthopol_chain(ch_l[2],side,P)
        else
            error("no discretization chosen")
        end
        
        return Vk_emp,ϵb_emp,Vk_fill,ϵb_fill
    end
function NEQ_SIAM_hamiltonian(P,DP)
        """
        -Will need to change qB_L,qB_R,qS,qA,q
        
        -spin up baths need to be flipped, not the left bath anymore.
        """
        
        (;N_L,N_R,Ns,bath_mode_type,ϵ) = P
        (;N,qB_R) = DP
        total_mode_number = 4*(N_L+N_R)+2*Ns
        H_single = complex(zeros(total_mode_number,total_mode_number))
        
        Vk_emp_L,ϵb_emp_L,Vk_fill_L,ϵb_fill_L = initialise_bath_NEQ_SIAM("left",P,DP)
        Vk_emp_R,ϵb_emp_R,Vk_fill_R,ϵb_fill_R = initialise_bath_NEQ_SIAM("right",P,DP)

        qB_L_up_filled = 1:4:4*(N_L)
#         qB_R_up_filled = 3:4:4*(N_L)
#         qB_L_up_empty = qB_L_up_filled .+1
#         qB_R_up_empty = qB_R_up_filled .+1
        
        qB_L_dn_filled = (4*N_L+2*Ns+1):4:total_mode_number
#         qB_R_dn_filled = (4*N_L+2*Ns+3):total_mode_number
#         qB_L_dn_empty = qB_L_dn_filled .+1
#         qB_R_dn_empty = qB_R_dn_filled .+1
        
        
        #Spin up, oriented to the left.
        b = 0
        Plots.plot(1:length(ϵb_fill_L),real.(ϵb_fill_L),label="left filled")
        Plots.plot!(1:length(ϵb_fill_R),real.(ϵb_fill_R),label="right filled")
        Plots.plot!(1:length(ϵb_emp_L),real.(ϵb_emp_L),label="left empty")
        display(Plots.plot!(1:length(ϵb_emp_R),real.(ϵb_emp_R),label="right empty",title="Energies"))
    
        Plots.plot(1:length(Vk_fill_L),real.(Vk_fill_L),label="left filled")
        Plots.plot!(1:length(Vk_fill_R),real.(Vk_fill_R),label="right filled")
        Plots.plot!(1:length(Vk_emp_L),real.(Vk_emp_L),label="left empty")
        display(Plots.plot!(1:length(Vk_emp_R),real.(Vk_emp_R),label="right empty",title="Couplings"))
    
        for j in qB_L_up_filled
            b += 1
            #Left bath
            H_single[j,j] = reverse(ϵb_fill_L)[b]
            H_single[j+1,j+1] = reverse(ϵb_emp_L)[b]
            
            #Right bath
            H_single[j+2,j+2] = reverse(ϵb_fill_R)[b]
            H_single[j+3,j+3] = reverse(ϵb_emp_R)[b]
            if j<(4*N_L-3)
                #Left bath
                H_single[j,j+4] = reverse(Vk_fill_L)[b]
                H_single[j+4,j] = conj(reverse(Vk_fill_L)[b])
                H_single[j+1,j+5] =  reverse(Vk_emp_L)[b]
                H_single[j+5,j+1] =  conj(reverse(Vk_emp_L)[b])
                
                #Right bath
                H_single[j+2,j+6] = reverse(Vk_fill_R)[b]
                H_single[j+6,j+2] = conj(reverse(Vk_fill_R)[b])
                H_single[j+3,j+7] =  reverse(Vk_emp_R)[b]
                H_single[j+7,j+3] =  conj(reverse(Vk_emp_R)[b])
            else
                #Left bath
                H_single[j,j+4] = reverse(Vk_fill_L)[b]
                H_single[j+4,j] = conj(reverse(Vk_fill_L)[b])
                H_single[j+1,j+4] =  reverse(Vk_emp_L)[b]
                H_single[j+4,j+1] =  conj(reverse(Vk_emp_L)[b])
                
                #Right bath
                H_single[j+2,j+4] = reverse(Vk_fill_R)[b]
                H_single[j+4,j+2] = conj(reverse(Vk_fill_R)[b])
                H_single[j+3,j+4] =  reverse(Vk_emp_R)[b]
                H_single[j+4,j+3] =  conj(reverse(Vk_emp_R)[b])
            end
        end
    
        #Impurity, split into two modes, left is spin up and right is spin down.
        qS = 4*N_L+1:2:4*N_L+3
        qA = qS .+ 1    
        H_single[qS[1],qS[1]] = ϵ 
        H_single[qS[2],qS[2]] = ϵ

        #Spin down, oriented to the right.
        b = 0

        for j in qB_L_dn_filled
            b += 1
            
            #Left bath
            H_single[j,j] = ϵb_fill_L[b]
            H_single[j+1,j+1] = ϵb_emp_L[b]

            #Right bath
            H_single[j+2,j+2] = ϵb_fill_R[b]
            H_single[j+3,j+3] = ϵb_emp_R[b]
            if j>qB_L_dn_filled[1]
                
                #Left bath
                H_single[j-4,j] = Vk_fill_L[b]
                H_single[j,j-4] = conj(Vk_fill_L[b])
                H_single[j-3,j+1] =  Vk_emp_L[b]
                H_single[j+1,j-3] =  conj(Vk_emp_L[b])
                
                #Right bath
                H_single[j-2,j+2] = Vk_fill_R[b]
                H_single[j+2,j-2] = conj(Vk_fill_R[b])
                H_single[j-1,j+3] =  Vk_emp_R[b]
                H_single[j+3,j-1] =  conj(Vk_emp_R[b])
            else
                #Left bath
                H_single[j-2,j] = Vk_fill_L[b]
                H_single[j,j-2] = conj(Vk_fill_L[b])
                H_single[j-2,j+1] =  Vk_emp_L[b]
                H_single[j+1,j-2] =  conj(Vk_emp_L[b])
                
                #Right bath
                H_single[j-2,j+2] = Vk_fill_R[b]
                H_single[j+2,j-2] = conj(Vk_fill_R[b])
                H_single[j-2,j+3] =  Vk_emp_R[b]
                H_single[j+3,j-2] =  conj(Vk_emp_R[b])
            end
        end
        return H_single
    end
function NEQ_SIAM_DP_initialisation(P)
"""
        Initialises all dependent variables defined by P.
        """
        (;N_L,N_R,Ns) = P
        @assert(Ns == 2)
        DP = dependent_params()
        
        DP.N = 4*(N_L+N_R)+2*Ns
        DP.qtot = 1:DP.N 
        DP.qB_L = 1:4*N_L
        DP.qB_R = DP.qB_L[end]+2*P.Ns+1:DP.N
    
        DP.q =DP.qB_L[end]+1:DP.qB_L[end]+2*P.Ns                       #System sites and ancilla sites   
 
        if P.ordering_choice == "separated"
            DP.qS = DP.q[1:P.Ns]
            DP.qA = DP.q[P.Ns+1:2*P.Ns]
        elseif P.ordering_choice == "interleaved"   
            DP.qS = DP.q[1:2:(2*P.Ns-1)]
            DP.qA = DP.q[2:2:2*P.Ns]
        end
    
        DP.times = range(P.δt,stop=P.T,step =P.δt)           #Simulation time vector

        
        DP.bath_ann_op = "C"
        DP.bath_cre_op = "Cdag"
        DP.sys_ann_op = "C"
        DP.sys_cre_op = "Cdag"
    
        DP.s,DP.cdag,DP.c = mode_operators(DP,P)    # Array of site indices                      # identity matrix
        DP.Id = Vector{ITensor}(undef,DP.N)                    #List of MPS identities
        DP.F = ops(DP.s, [("F", n) for n in DP.qtot]); 
        for i =1:DP.N
            iv = DP.s[i]
            ID = ITensor(iv', dag(iv));
            for j in 1:ITensors.dim(iv)
                ID[iv' => j, iv => j] = 1.0
            end
            DP.Id[i] = ID
        end

     
        DP.H_single = NEQ_SIAM_hamiltonian(P,DP)
        H_os = build_from_matrix(DP.H_single,"Cdag","C")
        DP.H_MPO = MPO(H_os,DP.s)
    
        DP.ψ_init = initialise_psi(P,DP)
        
        DP.Ci = transpose(correlation_matrix(DP.ψ_init,"Cdag","C"))
        
        
        
        DP.T_unenriched = round(P.T-P.T_enrich,digits=10)       # Time when the state is no longer enriched each step                                       
        DP.nframe_en = Int(P.T_enrich/(P.n*P.δt))               #Number of timesteps between each enrichment and calculation of map
        DP.nframe_un = Int(DP.T_unenriched/(P.n*P.δt))          #Number of timesteps between each calculation of map after enrichment
        DP.nframe = DP.nframe_en + DP.nframe_un                 #Number of maps calculated
        DP.times1 = range(P.δt,stop = P.T_enrich,step = P.δt)

        return DP
    end




    # function non_interacting_sys_H(H_single,DP,P)
    #     """
    #     This function initialises the non-interacting parts of the system hamiltonian.
    #     More specifically, it encodes the hopping and self energy terms
    #     """
    #     (;Ns,N_L) = P
    #     (;ϵi,ti,Ui,qS) = DP 
    #     for i=1:Ns
    #         H_single[qS[i],qS[i]] = ϵi[i]
    #         if i<Ns
    #             H_single[qS[i],qS[i+1]] = ti[i]
    #             H_single[qS[i+1],qS[i]] = conj(ti[i])
    #         end
    #     end
    #     return H_single
    # end


    function super_fermion_solver(w,zero_out,DP,P)
        (;ordering_choice) = P
        (;H_single,Ci,times,N,Λ) = DP 
    
        if ordering_choice == "interleaved"
            physical_modes = 1:2:size(H_single,1)
        else
            physical_modes = 1:Int(size(H_single,1)/2)
        end
    
        Nt = size(times,1); # Extract the number of time steps to compute
        Nw = size(w,1); # Extract the number of frequencies to compute
        G_greater = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt); # Storage for G_greater(i,j,t)
        G_lesser = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt); # Storage for G_lesser(i,j,t)
        G_t_retarded = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt) # Storage for retarded Green's function
        Gt_eq = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt); # Storage for G(i,j,t) (equal time greens function)
        Gw = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nw); # Storage for G(w)
    
        # Perform a full diagonalisation of the super Hamiltonian:
        F = eigen(H_single)
        E = F.values
        V = F.vectors
        Vinv = inv(V) # Compute the inverse of the matrix of right eigenvectors
    
        # Form the matrix of normal mode correlations for the stationary state:
        D = diagm(imag(E).>=0);
        ImD = I - D;
    
        energy_matrix = complex(zeros(size(V)))
        [energy_matrix[i,j] =E[i]-E[j] for i in 1:size(V)[1] for j in 1:size(V)[1]]
        G0 = Vinv*transpose(Ci)*V
    
        if zero_out
            G0[imag.(transpose(energy_matrix)) .< 0 ] .= 0 ##zeroing out non-physical elements that lead to numerical instabilities.
        end
        
        # Compute the single-particle Green function for NESS G_ij(t) = -iΘ(t) <{c+_i(t),c_j(0)}> ≡ Θ(t)(G^{>}_{ij}(t)-G^{<}_{ij}(t)), 
        #based on the Stephen's superfermion notes and the paper https://arxiv.org/abs/2012.01424
    
    
        @showprogress 1 "Computing ..." for n=1:Nt
            Gt = conj.(transpose(V*D*diagm(exp.(im*E*times[n]))*Vinv))
            Gt = Gt[physical_modes,physical_modes]; # Extract physical modes
            G_lesser[:,:,n] = im*Gt;
            
            Gt = transpose(V*ImD*diagm(exp.(-im*E*times[n]))*Vinv)
            Gt = Gt[physical_modes,physical_modes]; # Extract physical modes
            G_greater[:,:,n] = -im*Gt;
    
            G_t_retarded[:,:,n] = G_greater[:,:,n] -G_lesser[:,:,n]
    
           #Gt = transpose(V*diagm(exp.(-im*E*t[n]))*G0*diagm(exp.(im*E*t[n]))*Vinv)
            Gt = exp.(im*transpose(energy_matrix)*times[n]).*G0
            Gt = transpose(V*Gt*Vinv)
            Gt_eq[:,:,n] = Gt[physical_modes,physical_modes]
        end
        for n=1:Nw
            
            Gw[:,:,n] = inv(w[n]*I - H_single[physical_modes,physical_modes] + im*Λ)
        end
        G∞ = transpose(V*D*Vinv)
        return Gw,Gt_eq,G_greater,G_lesser,G_t_retarded,G∞ 
    end

    function super_fermion_solver_factorised_initial_state(zero_out,DP,P)
        (;ordering_choice) = P
        (;H_single,Ci,times,N,Λ) = DP 
    
        if ordering_choice == "interleaved"
            physical_modes = 1:2:size(H_single,1)
        else
            physical_modes = 1:Int(size(H_single,1)/2)
        end
    
        Nt = size(times,1); # Extract the number of time steps to compute
        G_greater = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt); # Storage for G_greater(i,j,t)
        G_lesser = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt); # Storage for G_lesser(i,j,t)
        G_t_retarded = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt) # Storage for retarded Green's function
        Gt_eq = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nt); # Storage for G(i,j,t) (equal time greens function)
        #Gw = zeros(ComplexF64,length(physical_modes),length(physical_modes),Nw); # Storage for G(w)
    
        # Perform a full diagonalisation of the super Hamiltonian:
        F = eigen(H_single)
        E = F.values
        V = F.vectors
        Vinv = inv(V) # Compute the inverse of the matrix of right eigenvectors
    
        # Form the matrix of normal mode correlations for the stationary state:
        D = diagm(imag(E).>=0);
        ImD = I - D;
    
        energy_matrix = complex(zeros(size(V)))
        [energy_matrix[i,j] =E[i]-E[j] for i in 1:size(V)[1] for j in 1:size(V)[1]]
        G0 = Vinv*transpose(Ci)*V
    
        if zero_out
            G0[imag.(transpose(energy_matrix)) .< 0 ] .= 0 ##zeroing out non-physical elements that lead to numerical instabilities.
        end
        
        # Compute the single-particle Green function for NESS G_ij(t) = -iΘ(t) <{c+_i(t),c_j(0)}> ≡ Θ(t)(G^{>}_{ij}(t)-G^{<}_{ij}(t)), 
        #based on the Stephen's superfermion notes and the paper https://arxiv.org/abs/2012.01424
    
    
        @showprogress 1 "Computing ..." for n=1:Nt
            Gt = conj.(transpose(V*G0*diagm(exp.(im*E*times[n]))*Vinv))
            Gt = Gt[physical_modes,physical_modes]; # Extract physical modes
            G_lesser[:,:,n] = -im*Gt;
    
            Gt = transpose(V*diagm(exp.(-im*E*times[n]))*(I-G0)*Vinv)
            Gt = Gt[physical_modes,physical_modes]; # Extract physical modes
            G_greater[:,:,n] = -im*Gt;
    
            G_t_retarded[:,:,n] = G_greater[:,:,n] +G_lesser[:,:,n]
    
           #Gt = transpose(V*diagm(exp.(-im*E*t[n]))*G0*diagm(exp.(im*E*t[n]))*Vinv)
            Gt = exp.(im*transpose(energy_matrix)*times[n]).*G0
            Gt = transpose(V*Gt*Vinv)
            Gt_eq[:,:,n] = Gt[physical_modes,physical_modes]
        end
       
        G∞ = transpose(V*D*Vinv)
        return Gt_eq,G_greater,G_lesser,G_t_retarded,G∞
    end
    
    function Markovian_closure(Nc,filled_or_empty)
        if Nc == 6
            # Hopping to pseudo-modes = c/2
            c = [(2.74e-5 - 1.11e-5*im), (-4.79e-1 + 3.99e-1*im), (6.34e-6 - 3.53e-6*im), (4.82e-1 - 3.84e-1*im), (-1.40e-6 + 2.45e-6*im), (3.83e-1 - 2.93e-1*im)]
            # Hopping between pseudo-mods = -g
            g = [0.785, -0.813, -1.08, -0.675, 0.805];
            # Damping of pseudo-modes = -2*G
            G = [-1.60e-2, -1.48e-10, -2.18, -1.44e-11, -4.79e-3, -1.57e-9] 
        elseif Nc == 8
            # Hopping to pseudo-modes = c/2
            c = [(-6.58e-2 - 2.48e-1*im), (-1.31e-1 + 3.47e-2*im), (-1.79e-1 - 6.75e-1*im),
                  (1.92e-2 - 5.08e-3*im), (9.77e-2 + 3.68e-1*im), (-1.36e-1 + 3.6e-2*im),(-1.06e-1 - 4.01e-1*im), (-2.91e-1 + 7.73e-2*im)]
            # Hopping between pseudo-mods = -g
            g = [-0.888, 0.407, -0.996, -1.49, -1.04,-0.455,0.848];
            # Damping of pseudo-modes = -2*G
            G = [-1.06e-9, -1.64e-10, -2.70e-11, -2.98, -1.02e-9, -3.61e-9,-3.53e-11,-3.73e-11]
        elseif Nc == 10
            # Hopping to pseudo-modes = c/2
            c = [(-1.32e-3 + 4.62e-4*im), (3.32e-3 + 5.49e-4*im), (-2.4e-3 - 1.48e-3*im),
                  (1.94e-2 - 3.55e-2*im), (-3.32e-2 - 1.2e-2*im), (1.04e-1 - 3.53e-1*im),
                  (1.21e-1 + 2.08e-2*im), (1.65e-1 - 8.17e-1*im),(-1.21e-1 - 4.45e-3*im),(4.72e-2 - 3.67e-1*im)]
            # Hopping between pseudo-mods = -g
            g = [1.13,1.05,-1.08,0.835,-0.604,-0.509,0.677,0.161,-0.951];
            # Damping of pseudo-modes = -2*G
            G = [-3.43e-1,-8.67e-5,-2.73,-7.09e-1,-3.24e-6,-4.5e-7,-2.79e-6,-9.48e-5,-1.37e-3,-5.95e-6]
        end
        
        if filled_or_empty == "filled"
            g = -g
        end
        
        # Construct single-particle Hamiltonian matrix, with first mode being the mode coupled to the Markovian closure.
        H = zeros(ComplexF64,Nc+1,Nc+1);
        for j=1:(Nc-1)
            H[j+1,j+2] = -g[j]
            H[j+2,j+1] = -g[j]
        end
        for j=1:Nc
            H[1,j+1] = 0.5*c[j]
            H[j+1,1] = 0.5*conj(c[j])
        end
        return H,G
    end

    function combine_thermofield_with_markovian_closure(Ci_physical,P,DP)
        
        (;N_L,N_R,Nc,Ns) = P
        
    
        N_physical = 2*N_L+2*N_R+Ns
        extra_modes = 2*((P.N_L>0)+(P.N_R>0))*Nc
        N_total = N_physical+extra_modes
        H_combined = complex(zeros(N_total,N_total))
        Ci_combined = complex(zeros(N_total,N_total))
        Γd = complex(zeros(N_total))
        Γe = complex(zeros(N_total))
    
        if N_L >0
            Left_filled_MC_inds = 1:2:2*Nc 
            Left_empty_MC_inds = 2:2:2*Nc
            unitary_inds = 1+2*Nc:N_physical+2*Nc
        else
            unitary_inds = 1:N_physical
        end
    
        if N_R >0
            Right_filled_MC_inds = unitary_inds[end]+1:2:unitary_inds[end]+2*Nc
            Right_empty_MC_inds = unitary_inds[end]+2:2:unitary_inds[end]+2*Nc
        end
    
        ##Creates the hamiltonian without the Markovian closure
        H = complex(zeros(N_physical,N_physical))
        H= create_H_single(P,DP;N_physical=N_physical)
        H = system_Hamiltonian(H,P,DP)[2]
        H =  system_bath_couplings(H,P,DP)[2]
    
        ###Baths and system
        H_combined[unitary_inds,unitary_inds] = H
        Ci_combined[unitary_inds,unitary_inds] = Ci_physical
        
        filled_MC_matrix = Markovian_closure(Nc,"filled")[1]
        empty_MC_matrix,G = Markovian_closure(Nc,"empty")
    
        if N_L >0 
            ##self hamiltonian and initial correlation matrix for empty and filled markovian enclosure for the left bath
            H_combined[[Left_filled_MC_inds;unitary_inds[1]],[Left_filled_MC_inds;unitary_inds[1]]] = reverse(filled_MC_matrix,dims=(1,2))
            H_combined[[Left_empty_MC_inds;unitary_inds[2]],[Left_empty_MC_inds;unitary_inds[2]]] = reverse(empty_MC_matrix,dims=(1,2))
    
            ##Reinputting the self energies of the end chain elements, coupled to the Markovian enclosure
            H_combined[unitary_inds[1],unitary_inds[1]] = H[1,1]
            H_combined[unitary_inds[2],unitary_inds[2]] = H[2,2]
            [Ci_combined[i,i] = 1 for i in Left_filled_MC_inds]
        end
        
        if N_R >0 
            ##self hamiltonian and initial correlation matrix for empty and filled markovian enclosure
            H_combined[[unitary_inds[end-1];Right_filled_MC_inds],[unitary_inds[end-1];Right_filled_MC_inds]] = filled_MC_matrix
            H_combined[[unitary_inds[end];Right_empty_MC_inds],[unitary_inds[end];Right_empty_MC_inds]] = empty_MC_matrix
    
            ##Reinputting the self energies of the end chain elements, coupled to the Markovian enclosure
            H_combined[unitary_inds[end-1],unitary_inds[end-1]] = H[end-1,end-1]
            H_combined[unitary_inds[end],unitary_inds[end]] = H[end,end]
            [Ci_combined[i,i] = 1 for i in Right_filled_MC_inds]
        end
    
        N_L >0 && (Γe[Left_filled_MC_inds] = -2*reverse(G))
        N_L >0 && (Γd[Left_empty_MC_inds] = -2*reverse(G))
    
        N_R >0 && (Γe[Right_filled_MC_inds] = -2*G)
        N_R >0 && (Γd[Right_empty_MC_inds] = -2*G)
    
        return H_combined,Γd,Γe,Ci_combined
    end
    
    function combine_thermofield_with_markovian_closure_old(Nc,P,DP)
        
        (;N,Ci) = DP
        H_combined = complex(zeros(N+2*Nc,N+2*Nc))
        Ci_combined = complex(zeros(N+2*Nc,N+2*Nc))
    
        H = complex(zeros(N,N))
        H= create_H_single(P,DP)
        H = system_Hamiltonian(H,P,DP)[2]
        H =  system_bath_couplings(H,P,DP)[2]
    
        ###Baths and system
        H_combined[1:N,1:N] = H
        Ci_combined[1:N,1:N] = Ci
        
        ##self hamiltonian and initial correlation matrix for empty and filled markovian enclosure
        H_combined[N-1:2:N+2*Nc-1,N-1:2:N+2*Nc-1] = Markovian_closure(Nc,"filled")[1]
        H_combined[N:2:N+2*Nc,N:2:N+2*Nc],G = Markovian_closure(Nc,"empty")
        [Ci_combined[i,i] = 1 for i in N+1:2:N+2*Nc-1]
    
        
        ##Reinputting the self energies of the final chain elements, coupled to the Markovian enclosure
        H_combined[N-1,N-1] = H[N-1,N-1]
        H_combined[N,N] = H[N,N]
    
    
        return H_combined,Ci_combined,G
    end

    function interleave_matrix(G)
        """
        Changes the mode ordering of the matrix G from separated to interleaved.
        """
        n = Int((size(G)[1])/2)
        G_int = similar(G)
        for i =1:n
            for j=1:n
                G_int[2*i-1,2*j-1] = G[i,j]
                G_int[2*i,2*j-1] = G[i+n,j]
                G_int[2*i-1,2*j] = G[i,j+n]
                G_int[2*i,2*j] = G[i+n,j+n] 
            end
        end
        return G_int
    end
